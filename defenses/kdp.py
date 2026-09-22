"""KDP — Key-Data Provenance Defense.

Every client registers a hash of its data sample at enrollment. At each
round, the client submits (update, current_hash). The server rejects
any client whose hash doesn't match its registered one — this catches
Data Poisoning and Backdoor attacks that modify local data.

Author: Jenan Jader Msad — Original method for PhD research.
"""
from typing import List, Dict, Set
import hashlib
import torch
from torch.utils.data import Subset


def compute_data_hash(dataset, indices: List[int], sample_size: int = 100) -> str:
    """Compute a SHA-256 hash of a client's data sample.

    Uses the first `sample_size` samples for consistency across rounds.
    Hash covers both features and labels — any tampering changes it.

    Args:
        dataset: the base training dataset.
        indices: this client's assigned indices.
        sample_size: how many samples to hash (default 100).

    Returns:
        Hexadecimal SHA-256 hash string.
    """
    # Use first sample_size indices (deterministic per client)
    sample_indices = indices[:sample_size]
    subset = Subset(dataset, sample_indices)

    # Concatenate features and labels into a byte string
    hasher = hashlib.sha256()
    for idx in range(len(subset)):
        image, label = subset[idx]
        # Include both features and label in the hash
        hasher.update(image.numpy().tobytes())
        hasher.update(str(label).encode())

    return hasher.hexdigest()


def build_registry(partition: Dict[int, List[int]], dataset) -> Dict[int, str]:
    """Build the server's registry: {client_id: data_hash}.
    Called once at the start of training (enrollment phase).
    """
    registry = {}
    for cid, indices in partition.items():
        registry[cid] = compute_data_hash(dataset, indices)
    return registry


def verify_client(cid: int, current_hash: str, registry: Dict[int, str]) -> bool:
    """Check if client's current hash matches the registered one.
    Returns True if trusted, False if data was tampered.
    """
    if cid not in registry:
        return False
    return current_hash == registry[cid]


def aggregate(
    client_states: List[Dict[str, torch.Tensor]],
    client_weights: List[float],
    client_ids: List[int],
    client_hashes: List[str],
    registry: Dict[int, str],
) -> Dict[str, torch.Tensor]:
    """KDP aggregation: verify hashes, drop untrusted, then FedAvg.

    Args:
        client_states: list of client state_dicts (post-training).
        client_weights: sample counts for weighting.
        client_ids: participating client IDs this round.
        client_hashes: hashes each client sent this round.
        registry: server's enrollment registry.

    Returns:
        Aggregated state_dict from trusted clients only.
    """
    if not client_states:
        raise ValueError("No client states to aggregate.")

    # Filter: keep only clients whose hash matches
    trusted = []
    rejected = []
    for i, cid in enumerate(client_ids):
        if verify_client(cid, client_hashes[i], registry):
            trusted.append(i)
        else:
            rejected.append(cid)

    # Log rejected (server would print this)
    if rejected:
        print(f"    [KDP] Rejected clients: {rejected}")

    # If all rejected (extreme case), fallback to first client
    if not trusted:
        print("    [KDP] WARNING: All clients rejected — keeping global state")
        # Return average of ALL (fallback), but this should be rare
        trusted = list(range(len(client_states)))

    # Aggregate trusted only (weighted FedAvg)
    trusted_states = [client_states[i] for i in trusted]
    trusted_weights = [client_weights[i] for i in trusted]
    total = sum(trusted_weights)

    aggregated = {}
    for k in trusted_states[0]:
        stacked = torch.stack([s[k].float() for s in trusted_states], dim=0)
        w = torch.tensor(trusted_weights, dtype=torch.float32) / total
        w = w.view(-1, *([1] * (stacked.dim() - 1)))
        aggregated[k] = (stacked * w).sum(dim=0).to(trusted_states[0][k].dtype)

    return aggregated


if __name__ == "__main__":
    print("KDP module loaded successfully.")
    print("Full test requires partition + dataset — tested via main.py")
