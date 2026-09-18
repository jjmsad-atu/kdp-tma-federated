"""FLTrust — Byzantine-robust aggregation via trust bootstrapping.

Cao et al. 2021, "FLTrust: Byzantine-robust Federated Learning via Trust
Bootstrapping" (NDSS 2021).

Idea:
  1. Server maintains a small trusted reference dataset (100 per class).
  2. Each round, server trains θ_t on this reference to compute g_ref.
  3. For each client update Δθ_k, compute cos(Δθ_k, g_ref).
  4. Apply ReLU: negative-similarity clients are rejected (weight = 0).
  5. Normalize updates to reference's norm to prevent scaling attacks.
  6. Weighted average of surviving updates.
"""
from typing import List, Dict, Optional
import copy
import torch
import torch.nn as nn


def _flatten_state(state: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Flatten state_dict into a single 1-D tensor."""
    return torch.cat([v.float().flatten() for v in state.values()])


def _compute_reference_gradient(
    global_state: Dict[str, torch.Tensor],
    reference_dataset,
    build_model_fn,
    device,
    lr: float = 0.01,
    momentum: float = 0.9,
    batch_size: int = 32,
) -> Dict[str, torch.Tensor]:
    """Train the global model on the reference dataset for one epoch.
    Returns the *update* (final - initial), NOT the final weights.
    This update = 'the correct direction' according to server.
    """
    from torch.utils.data import DataLoader

    model = build_model_fn()
    model.load_state_dict(global_state)
    model.to(device)
    model.train()

    loader = DataLoader(reference_dataset, batch_size=batch_size,
                        shuffle=True, num_workers=0)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    criterion = nn.CrossEntropyLoss()

    # Train for exactly one epoch on reference
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        opt.step()

    # Return the update (delta), not the weights
    final = model.state_dict()
    delta = {k: (final[k].detach().cpu() - global_state[k]).float()
             for k in global_state}
    return delta


def aggregate(
    client_states: List[Dict[str, torch.Tensor]],
    client_weights: List[float] = None,
    global_state: Dict[str, torch.Tensor] = None,
    reference_dataset=None,
    build_model_fn=None,
    device=None,
    lr: float = 0.01,
    momentum: float = 0.9,
) -> Dict[str, torch.Tensor]:
    """FLTrust aggregation.

    Args:
        client_states: list of client state_dicts (post-training).
        client_weights: unused (kept for interface compatibility).
        global_state: current global model state (pre-round).
        reference_dataset: server's trusted reference dataset.
        build_model_fn: function that returns a fresh model instance.
        device: torch device.

    Returns:
        Aggregated state_dict.
    """
    if not client_states:
        raise ValueError("No client states to aggregate.")
    if global_state is None or reference_dataset is None or build_model_fn is None:
        raise ValueError(
            "FLTrust requires global_state, reference_dataset, and build_model_fn."
        )

    # Step 1: Compute reference gradient (server-side)
    g_ref = _compute_reference_gradient(
        global_state, reference_dataset, build_model_fn, device, lr, momentum
    )
    g_ref_flat = _flatten_state(g_ref)
    g_ref_norm = g_ref_flat.norm() + 1e-10  # avoid divide by zero

    # Step 2: For each client, compute the update (delta) and its similarity
    client_deltas = []
    similarities = []
    for cs in client_states:
        # Compute delta = client_state - global_state (client's update)
        delta = {k: (cs[k].float() - global_state[k].float())
                 for k in global_state}
        client_deltas.append(delta)

        delta_flat = _flatten_state(delta)

        # Cosine similarity with reference
        cos_sim = torch.dot(delta_flat, g_ref_flat) / (
            delta_flat.norm() * g_ref_norm + 1e-10
        )
        similarities.append(cos_sim.item())

    # Step 3: ReLU on similarities (trust scores)
    trust_scores = torch.tensor([max(0.0, s) for s in similarities])

    # Step 4: If all trust scores are zero (extreme case), fall back to reference
    if trust_scores.sum() < 1e-10:
        # No client is trusted — apply g_ref alone
        aggregated = {
            k: (global_state[k].float() + g_ref[k]).to(global_state[k].dtype)
            for k in global_state
        }
        return aggregated

    # Step 5: Normalize each client's update to have the same norm as g_ref
    # This prevents an attacker from scaling their update to dominate
    normalized_deltas = []
    for delta in client_deltas:
        delta_flat = _flatten_state(delta)
        delta_norm = delta_flat.norm() + 1e-10
        scale = g_ref_norm / delta_norm
        norm_delta = {k: (delta[k] * scale).to(global_state[k].dtype)
                      for k in delta}
        normalized_deltas.append(norm_delta)

    # Step 6: Weighted average
    weights = trust_scores / trust_scores.sum()
    aggregated_delta = {}
    for k in global_state:
        stacked = torch.stack(
            [nd[k].float() for nd in normalized_deltas], dim=0
        )
        w = weights.view(-1, *([1] * (stacked.dim() - 1)))
        aggregated_delta[k] = (stacked * w).sum(dim=0)

    # Step 7: Apply the aggregated update to global state
    new_state = {
        k: (global_state[k].float() + aggregated_delta[k]).to(global_state[k].dtype)
        for k in global_state
    }
    return new_state


if __name__ == "__main__":
    # Quick sanity check without reference dataset (mocked)
    print("FLTrust module loaded successfully.")
    print("Full test requires reference_dataset and build_model_fn from main.py.")
    print("Will be tested via scripts/main.py --defense fltrust")
