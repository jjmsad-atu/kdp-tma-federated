"""Trimmed Mean — coordinate-wise trimmed mean aggregation.
Yin et al. 2018 (ICML)."""
from typing import List, Dict
import torch


def aggregate(client_states: List[Dict[str, torch.Tensor]],
              client_weights: List[float] = None,
              trim_ratio: float = 0.2) -> Dict[str, torch.Tensor]:
    if not client_states:
        raise ValueError("No client states to aggregate.")
    n_clients = len(client_states)
    n_trim = int(trim_ratio * n_clients)
    if 2 * n_trim >= n_clients:
        raise ValueError(f"trim_ratio={trim_ratio} too large for {n_clients} clients.")
    keys = client_states[0].keys()
    aggregated = {}
    for k in keys:
        stacked = torch.stack([s[k].float() for s in client_states], dim=0)
        sorted_stack, _ = torch.sort(stacked, dim=0)
        trimmed = sorted_stack[n_trim:n_clients - n_trim]
        aggregated[k] = trimmed.mean(dim=0).to(client_states[0][k].dtype)
    return aggregated


if __name__ == "__main__":
    states = [
        {"w": torch.tensor([1.0, 1.0])},
        {"w": torch.tensor([1.1, 0.9])},
        {"w": torch.tensor([0.9, 1.1])},
        {"w": torch.tensor([-100.0, -100.0])},
        {"w": torch.tensor([100.0, 100.0])},
    ]
    out = aggregate(states, trim_ratio=0.2)
    print("Trimmed mean output:", out["w"].tolist())
