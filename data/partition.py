"""
Dirichlet-based non-IID partitioning of image classification datasets.

Follows Hsu, Qi & Brown (2019), "Measuring the effects of non-identical data
distribution for federated visual classification" (arXiv:1909.06335).

Usage:
    python data/partition.py --dataset fashion-mnist --clients 100 --alpha 0.5
    python data/partition.py --dataset cifar10       --clients 100 --alpha 0.5

Output:
    partitions/{dataset}_alpha{alpha}_n{clients}.pkl
    A dict: {client_id: [sample_indices], ...}
    Plus:
    partitions/{dataset}_reference_indices.pkl
    A list of indices held by the server (removed from the client pool).
"""
import argparse
import pickle
import random
from pathlib import Path

import numpy as np
from torchvision import datasets, transforms


def dirichlet_partition(labels, num_clients, alpha, seed=42):
    """
    Partition sample indices by a Dirichlet(alpha) distribution over classes.

    A smaller alpha produces more skewed (more non-IID) partitions.
    """
    rng = np.random.default_rng(seed)
    num_classes = int(labels.max()) + 1
    idx_by_class = [np.where(labels == c)[0] for c in range(num_classes)]

    # For each class, draw a Dirichlet allocation over clients
    client_indices = [[] for _ in range(num_clients)]
    for c in range(num_classes):
        rng.shuffle(idx_by_class[c])
        proportions = rng.dirichlet(alpha * np.ones(num_clients))
        # Convert to integer counts
        split_points = (np.cumsum(proportions) * len(idx_by_class[c])).astype(int)[:-1]
        splits = np.split(idx_by_class[c], split_points)
        for client_id, chunk in enumerate(splits):
            client_indices[client_id].extend(chunk.tolist())

    # Shuffle within each client so the class order is not systematic
    for i in range(num_clients):
        rng.shuffle(client_indices[i])

    return {i: client_indices[i] for i in range(num_clients)}


def sample_reference_set(labels, per_class, seed=42):
    """Sample `per_class` indices from each class for the server reference set."""
    rng = np.random.default_rng(seed + 999)
    num_classes = int(labels.max()) + 1
    reference = []
    for c in range(num_classes):
        idx = np.where(labels == c)[0]
        chosen = rng.choice(idx, size=per_class, replace=False)
        reference.extend(chosen.tolist())
    return reference


def load_labels(dataset_name, root="./data_cache"):
    """Return the training-set label array for the requested dataset."""
    transform = transforms.ToTensor()
    if dataset_name == "fashion-mnist":
        ds = datasets.FashionMNIST(root=root, train=True, download=True,
                                    transform=transform)
        return np.array(ds.targets)
    if dataset_name == "cifar10":
        ds = datasets.CIFAR10(root=root, train=True, download=True,
                              transform=transform)
        return np.array(ds.targets)
    raise ValueError(f"Unknown dataset: {dataset_name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["fashion-mnist", "cifar10"], required=True)
    ap.add_argument("--clients", type=int, default=100)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--reference-per-class", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", type=Path, default=Path("partitions"))
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.dataset} labels ...")
    labels = load_labels(args.dataset)
    print(f"  Total samples: {len(labels)}")

    print(f"Sampling {args.reference_per_class} reference samples per class ...")
    ref_indices = sample_reference_set(labels, args.reference_per_class, args.seed)
    print(f"  Reference set size: {len(ref_indices)}")

    # Remove reference indices from the client pool
    client_pool = np.setdiff1d(np.arange(len(labels)), ref_indices)
    client_labels_map = {i: labels[i] for i in client_pool}

    print(f"Partitioning {len(client_pool)} samples over "
          f"{args.clients} clients with alpha={args.alpha} ...")
    # Re-index: the Dirichlet function works on a dense label array
    remaining_labels = np.array([client_labels_map[i] for i in client_pool])
    partition_by_reduced_idx = dirichlet_partition(
        remaining_labels, args.clients, args.alpha, args.seed
    )
    # Map back to the original dataset indices
    partition = {
        cid: [int(client_pool[j]) for j in idxs]
        for cid, idxs in partition_by_reduced_idx.items()
    }

    ref_path = args.out_dir / f"{args.dataset}_reference_indices.pkl"
    part_path = args.out_dir / f"{args.dataset}_alpha{args.alpha}_n{args.clients}.pkl"

    with open(ref_path, "wb") as f:
        pickle.dump(ref_indices, f)
    with open(part_path, "wb") as f:
        pickle.dump(partition, f)

    sizes = [len(v) for v in partition.values()]
    print(f"  Written {part_path}")
    print(f"  Written {ref_path}")
    print(f"  Per-client size — min: {min(sizes)}, median: {int(np.median(sizes))}, "
          f"max: {max(sizes)}")


if __name__ == "__main__":
    main()
