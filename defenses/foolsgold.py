"""FoolsGold — Sybil-resilient aggregation via update history similarity.

Fung et al. 2020, "The Limitations of Federated Learning in Sybil Settings"
(RAID 2020).

Idea:
  1. Maintain per-client historical sum of updates (memory).
  2. Compute pairwise cosine similarity across client histories.
  3. Clients with anomalously high similarity to others are down-weighted.
  4. Use a logit-based rescaling: alpha_k = 1 - max_j sim(k, j)
"""
from typing import List, Dict
import torch


# Global memory across rounds: cid -> accumulated update
# This module keeps state; call reset_memory() to clear it (e.g., between runs)
_HISTORY: Dict[int, torch.Tensor] = {}


def reset_memory():
    """Clear the accumulated history (call at start of a new run)."""
    global _HISTORY
    _HISTORY = {}


def _flatten(state: Dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat([v.float().flatten() for v in state.values()])


def aggregate(
    client_states: List[Dict[str, torch.Tensor]],
    client_weights: List[float] = None,
    client_ids: List[int] = None,
    global_state: Dict[str, torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """FoolsGold aggregation.

    Args:
        client_states: list of client state_dicts (post-training).
        client_weights: unused (kept for interface compatibility).
        client_ids: list of the client IDs participating in this round.
        global_state: current global model state (pre-round).

    Returns:
        Aggregated state_dict.
    """
    if not client_states:
        raise ValueError("No client states to aggregate.")
    if client_ids is None:
        raise ValueError("FoolsGold requires client_ids to track history.")
    if global_state is None:
        raise ValueError("FoolsGold requires global_state to compute deltas.")

    n = len(client_states)

    # Step 1: Compute this round's deltas and update history
    deltas_flat = []
    for cs, cid in zip(client_states, client_ids):
        delta = torch.cat(
            [(cs[k].float() - global_state[k].float()).flatten()
             for k in global_state]
        )
        deltas_flat.append(delta)

        # Update accumulated history for this client
        if cid not in _HISTORY:
            _HISTORY[cid] = delta.clone()
        else:
            _HISTORY[cid] = _HISTORY[cid] + delta

    # Step 2: Build history matrix for participating clients
    history_matrix = torch.stack([_HISTORY[cid] for cid in client_ids], dim=0)

    # Step 3: Compute pairwise cosine similarity
    # Normalize each row
    norms = history_matrix.norm(dim=1, keepdim=True) + 1e-10
    normalized = history_matrix / norms
    cs_matrix = torch.mm(normalized, normalized.t())  # (n × n)

    # Zero out self-similarities
    cs_matrix.fill_diagonal_(0)

    # Step 4: For each client, find max similarity to any other client
    max_sim, _ = cs_matrix.max(dim=1)  # shape (n,)

    # Step 5: Pardoning — if client i's max_sim is smaller than client j's,
    # rescale to avoid punishing honest clients that happen to be similar
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if max_sim[j] > max_sim[i] and max_sim[j] > 1e-10:
                cs_matrix[i, j] *= max_sim[i] / max_sim[j]

    max_sim, _ = cs_matrix.max(dim=1)

    # Step 6: Compute learning rate alpha_k = 1 - max_sim_k
    alpha = 1 - max_sim
    alpha = torch.clamp(alpha, min=0, max=1)

    # Step 7: Apply logit rescaling (as in original paper)
    # alpha_k = ln((alpha_k / (1 - alpha_k)) + 1e-5)
    # then normalize to [0, 1]
    eps = 1e-5
    alpha_safe = torch.clamp(alpha, min=eps, max=1 - eps)
    logit = torch.log(alpha_safe / (1 - alpha_safe) + eps)
    logit = torch.clamp(logit, min=-100, max=100)
    # Normalize to [0, 1] via min-max
    if logit.max() > logit.min():
        weights = (logit - logit.min()) / (logit.max() - logit.min())
    else:
        weights = torch.ones_like(logit)

    # If everything ends up zero, fall back to uniform
    if weights.sum() < 1e-10:
        weights = torch.ones_like(weights)
    weights = weights / weights.sum()

    # Step 8: Weighted average of deltas
    aggregated_delta = torch.zeros_like(deltas_flat[0])
    for i, d in enumerate(deltas_flat):
        aggregated_delta = aggregated_delta + weights[i] * d

    # Step 9: Reshape back and apply to global state
    new_state = {}
    idx = 0
    for k, v in global_state.items():
        numel = v.numel()
        new_state[k] = (v.float() + aggregated_delta[idx:idx + numel].view(v.shape)).to(v.dtype)
        idx += numel

    return new_state


if __name__ == "__main__":
    print("FoolsGold module loaded successfully.")
    print("Requires client_ids and global_state — tested via main.py")
