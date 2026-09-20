"""Sign-Flipping Byzantine Attack.

The attacker computes an honest gradient, then flips its sign to push
the model in the opposite direction of correct learning.

Reference: Baruch et al. 2019, "A Little Is Enough: Circumventing Defenses
For Distributed Learning" (NeurIPS 2019).
"""
from typing import Dict
import torch


def apply(client_state: Dict[str, torch.Tensor],
          global_state: Dict[str, torch.Tensor],
          scale: float = 1.0) -> Dict[str, torch.Tensor]:
    """Apply sign-flipping attack to a malicious client's update.

    Instead of returning the client's honest state, return one where
    the update direction is reversed:
        new_state = global_state - scale * (client_state - global_state)

    Args:
        client_state: the honestly-trained state (what the client would have sent).
        global_state: the current global model state.
        scale: multiplier for the flipped update (default 1.0 = full flip).

    Returns:
        The poisoned state_dict to submit to the server.
    """
    poisoned = {}
    for k in global_state:
        # Compute honest delta
        delta = client_state[k].float() - global_state[k].float()
        # Flip and rescale
        flipped_delta = -scale * delta
        # Apply to global state
        poisoned[k] = (global_state[k].float() + flipped_delta).to(global_state[k].dtype)
    return poisoned


if __name__ == "__main__":
    # Quick sanity check
    global_state = {"w": torch.tensor([1.0, 1.0])}
    honest_state = {"w": torch.tensor([1.1, 1.2])}  # honest update: [+0.1, +0.2]

    poisoned = apply(honest_state, global_state, scale=1.0)
    print("Global state:  ", global_state["w"].tolist())
    print("Honest state:  ", honest_state["w"].tolist())
    print("Poisoned state:", poisoned["w"].tolist())
    print("Expected poisoned: [0.9, 0.8] (opposite direction from honest)")
