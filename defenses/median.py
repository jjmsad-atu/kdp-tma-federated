"""Coordinate-wise Median aggregation.
Yin et al. 2018 (ICML)."""
from typing import List, Dict
import torch


def aggregate(client_states: List[Dict[str, torch.Tensor]],
              client_weights: List[float] = None) -> Dict[str, torch.Tensor]:
    if not client_states:
        raise ValueError("No client states to aggregate.")
    keys = client_states[0].keys()
    aggregated = {}
    for k in keys:
        stacked = torch.stack([s[k].float() for s in client_states], dim=0)
        median_vals, _ = torch.median(stacked, dim=0)
        aggregated[k] = median_vals.to(client_states[0][k].dtype)
    return aggregated


if __name__ == "__main__":
    states = [
        {"w": torch.tensor([1.0, 1.0])},
        {"w": torch.tensor([1.1, 0.9])},
        {"w": torch.tensor([0.9, 1.1])},
        {"w": torch.tensor([-100.0, -100.0])},
        {"w": torch.tensor([100.0, 100.0])},
    ]
    out = aggregate(states)
    print("Median output:", out["w"].tolist())
