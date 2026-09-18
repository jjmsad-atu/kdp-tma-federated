"""Krum — Byzantine-tolerant aggregation via nearest-neighbor selection.

Blanchard et al. 2017, "Machine Learning with Adversaries: Byzantine Tolerant
Gradient Descent" (NeurIPS 2017).

Idea: For each client, compute the sum of squared distances to its
(N - f - 2) nearest neighbors. Select the client with the smallest sum.
"""
from typing import List, Dict
import torch


def _flatten(state: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Flatten a state_dict into a single 1-D tensor."""
    return torch.cat([v.float().flatten() for v in state.values()])


def _unflatten(flat: torch.Tensor, template: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Reconstruct a state_dict from a flat tensor using template shapes."""
    result = {}
    idx = 0
    for k, v in template.items():
        numel = v.numel()
        result[k] = flat[idx:idx + numel].view(v.shape).to(v.dtype)
        idx += numel
    return result


def aggregate(client_states: List[Dict[str, torch.Tensor]],
              client_weights: List[float] = None,
              num_byzantine: int = 2) -> Dict[str, torch.Tensor]:
    """Krum aggregation.

    Args:
        client_states: list of client state_dicts.
        client_weights: unused (kept for interface compatibility).
        num_byzantine: expected number of Byzantine clients (f).

    Returns:
        The state_dict of the single selected honest client.
    """
    if not client_states:
        raise ValueError("No client states to aggregate.")

    n = len(client_states)
    if n - num_byzantine - 2 < 1:
        raise ValueError(
            f"n={n}, num_byzantine={num_byzantine}: too few clients for Krum. "
            f"Need n >= num_byzantine + 3."
        )

    # Flatten all clients into a matrix (n × d)
    flats = torch.stack([_flatten(s) for s in client_states], dim=0)

    # Compute pairwise squared distances
    # dist[i, j] = ||flats[i] - flats[j]||^2
    dist = torch.cdist(flats, flats, p=2) ** 2  # (n × n)

    # For each client i, sum the (n - f - 2) smallest distances
    # (excluding distance to itself, which is 0)
    k = n - num_byzantine - 2  # number of neighbors to consider
    scores = torch.zeros(n)
    for i in range(n):
        # Sort distances from client i to others
        sorted_dists, _ = torch.sort(dist[i])
        # Skip index 0 (self-distance = 0), take next k smallest
        scores[i] = sorted_dists[1:k + 1].sum()

    # Select the client with the smallest score
    selected_idx = int(torch.argmin(scores).item())

    return client_states[selected_idx]


if __name__ == "__main__":
    # Quick sanity check: 5 clients, 2 attackers, expect an honest one selected
    states = [
        {"w": torch.tensor([0.5, 0.3])},   # honest
        {"w": torch.tensor([0.6, 0.4])},   # honest
        {"w": torch.tensor([0.4, 0.2])},   # honest
        {"w": torch.tensor([-50.0, 100.0])},  # attacker
        {"w": torch.tensor([50.0, -100.0])},  # attacker
    ]
    out = aggregate(states, num_byzantine=1)
    print("Krum selected:", out["w"].tolist())
    print("Expected: one of [0.5, 0.3], [0.6, 0.4], or [0.4, 0.2] (an honest client)")
