"""FedAvg — vanilla weighted average of client updates.

McMahan et al. 2017, "Communication-Efficient Learning of Deep Networks from
Decentralized Data" (AISTATS 2017).
"""
from typing import List, Dict
import torch


def aggregate(client_states: List[Dict[str, torch.Tensor]],
              client_weights: List[float]) -> Dict[str, torch.Tensor]:
    """Weighted average of state_dicts.

    Args:
        client_states: list of client state_dicts (same keys, same shapes).
        client_weights: list of nonnegative weights (typically num_samples).

    Returns:
        A single aggregated state_dict.
    """
    if not client_states:
        raise ValueError("No client states to aggregate.")

    total = float(sum(client_weights))
    if total <= 0:
        raise ValueError("Sum of client weights must be positive.")
    weights = [w / total for w in client_weights]

    keys = client_states[0].keys()
    aggregated = {}
    for k in keys:
        # Sum weighted tensors
        stacked = torch.stack([s[k].float() for s in client_states], dim=0)
        w = torch.tensor(weights, dtype=stacked.dtype, device=stacked.device)
        # Broadcast weights across all remaining dims of stacked
        while w.dim() < stacked.dim():
            w = w.unsqueeze(-1)
        aggregated[k] = (stacked * w).sum(dim=0).to(client_states[0][k].dtype)
    return aggregated


if __name__ == "__main__":
    # Quick sanity check
    s1 = {"w": torch.tensor([1.0, 2.0])}
    s2 = {"w": torch.tensor([3.0, 4.0])}
    out = aggregate([s1, s2], [1.0, 1.0])
    print("Expected: [2.0, 3.0]  ---  Got:", out["w"].tolist())
