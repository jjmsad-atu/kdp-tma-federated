"""Gaussian Noise Byzantine Attack (Strong Version).

The attacker replaces its update with large Gaussian noise scaled to be
much larger than typical parameter values, causing severe disruption.

Reference: Xie et al. 2020, "Fall of Empires: Breaking Byzantine-tolerant
SGD by Inner Product Manipulation" (UAI 2020).
"""
from typing import Dict
import torch


def apply(client_state: Dict[str, torch.Tensor],
          global_state: Dict[str, torch.Tensor],
          sigma_multiplier: float = 10.0) -> Dict[str, torch.Tensor]:
    """Apply Gaussian noise attack — STRONG version.

    Scales noise to be sigma_multiplier times the standard deviation of
    the model's parameters, ensuring the attack has real impact.

    Args:
        client_state: the honestly-trained state.
        global_state: the current global model state.
        sigma_multiplier: noise strength (default 10.0 = 10x parameter std).

    Returns:
        The poisoned state_dict.
    """
    # Compute the standard deviation of the global model's parameters
    all_params = torch.cat([v.float().flatten() for v in global_state.values()])
    param_std = all_params.std().item() + 1e-8

    # Sigma is now proportional to actual parameter scale
    sigma = sigma_multiplier * param_std

    # Generate large Gaussian noise and add it to global state
    poisoned = {}
    for k in global_state:
        noise = torch.randn_like(global_state[k].float()) * sigma
        poisoned[k] = (global_state[k].float() + noise).to(global_state[k].dtype)

    return poisoned


if __name__ == "__main__":
    torch.manual_seed(42)
    # Simulate more realistic parameters
    global_state = {
        "conv1.weight": torch.randn(32, 1, 5, 5) * 0.1,
        "conv1.bias": torch.zeros(32),
        "fc1.weight": torch.randn(256, 3136) * 0.02,
        "fc1.bias": torch.zeros(256),
    }
    honest_state = {k: v + torch.randn_like(v) * 0.001 for k, v in global_state.items()}

    poisoned = apply(honest_state, global_state, sigma_multiplier=10.0)

    # Compare norms
    honest_delta = torch.cat([
        (honest_state[k] - global_state[k]).flatten() for k in global_state
    ]).norm().item()

    poisoned_delta = torch.cat([
        (poisoned[k] - global_state[k]).flatten() for k in global_state
    ]).norm().item()

    print(f"Honest delta norm:   {honest_delta:.4f}")
    print(f"Poisoned delta norm: {poisoned_delta:.4f}")
    print(f"Amplification:       {poisoned_delta / honest_delta:.1f}x")
    print(f"Expected: Amplification > 100x (much stronger attack)")
