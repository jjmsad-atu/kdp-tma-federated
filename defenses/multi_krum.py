"""Multi-Krum — Byzantine-tolerant aggregation via top-m selection.

Blanchard et al. 2017, "Machine Learning with Adversaries: Byzantine Tolerant
Gradient Descent" (NeurIPS 2017).

Idea: Compute Krum scores for all clients, then average the m clients with
the smallest scores (m = n - f).
"""
from typing import List, Dict
import torch


def aggregate(client_states: List[Dict[str, torch.Tensor]],
              client_weights: List[float] = None,
              num_byzantine: int = 2) -> Dict[str, torch.Tensor]:
    """Multi-Krum aggregation.

    Args:
        client_states: list of client state_dicts.
        client_weights: unused (kept for interface compatibility).
        num_byzantine: expected number of Byzantine clients (f).

    Returns:
        Averaged state_dict of the m = n - f best clients.
    """
    if not client_states:
        raise ValueError("No client states to aggregate.")

    n = len(client_states)
    if n - num_byzantine - 2 < 1:
        raise ValueError(
            f"n={n}, num_byzantine={num_byzantine}: too few clients for Multi-Krum."
        )

    # Flatten all clients
    def _flatten(state):
        return torch.cat([v.float().flatten() for v in state.values()])

    flats = torch.stack([_flatten(s) for s in client_states], dim=0)

    # Compute pairwise squared distances
    dist = torch.cdist(flats, flats, p=2) ** 2

    # Krum score for each client
    k = n - num_byzantine - 2
    scores = torch.zeros(n)
    for i in range(n):
        sorted_dists, _ = torch.sort(dist[i])
        scores[i] = sorted_dists[1:k + 1].sum()

    # Select top m = n - f clients (smallest scores)
    m = n - num_byzantine
    _, best_indices = torch.topk(scores, m, largest=False)
    best_indices = best_indices.tolist()

    # Average the selected clients (uniform weights)
    keys = client_states[0].keys()
    aggregated = {}
    for key in keys:
        stacked = torch.stack(
            [client_states[i][key].float() for i in best_indices], dim=0
        )
        aggregated[key] = stacked.mean(dim=0).to(client_states[0][key].dtype)

    return aggregated


if __name__ == "__main__":
    # Sanity check: 5 clients, 2 attackers, expect honest ones averaged
    states = [
        {"w": torch.tensor([0.5, 0.3])},   # honest
        {"w": torch.tensor([0.6, 0.4])},   # honest
        {"w": torch.tensor([0.4, 0.2])},   # honest
        {"w": torch.tensor([-50.0, 100.0])},  # attacker
        {"w": torch.tensor([50.0, -100.0])},  # attacker
    ]
    out = aggregate(states, num_byzantine=2)
    print("Multi-Krum output:", out["w"].tolist())
    print("Expected: average of honest clients ≈ [0.5, 0.3]")
