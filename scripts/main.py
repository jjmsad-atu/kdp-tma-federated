"""
Main experiment entry point.
Supports: FedAvg, Trimmed Mean, Median, Krum, Multi-Krum, FLTrust, FoolsGold.
Attacks: none, sign_flip.
"""
import argparse
import copy
import json
import pickle
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import yaml

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.simple_cnn import SimpleCNN
from models.resnet18 import make_resnet18_cifar
from defenses.fedavg import aggregate as fedavg_aggregate
from defenses.trimmed_mean import aggregate as trimmed_mean_aggregate
from defenses.median import aggregate as median_aggregate
from defenses.krum import aggregate as krum_aggregate
from defenses.multi_krum import aggregate as multi_krum_aggregate
from defenses.fltrust import aggregate as fltrust_aggregate
from defenses.foolsgold import aggregate as foolsgold_aggregate, reset_memory as foolsgold_reset
from attacks.sign_flip import apply as sign_flip_attack
from attacks.gaussian_noise import apply as gaussian_noise_attack


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


DATA_ROOT = Path("./data_cache")


def load_dataset(name):
    if name == "fashion-mnist":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,)),
        ])
        train = datasets.FashionMNIST(DATA_ROOT, train=True, download=True, transform=transform)
        test = datasets.FashionMNIST(DATA_ROOT, train=False, download=True, transform=transform)
        return train, test
    if name == "cifar10":
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ])
        train = datasets.CIFAR10(DATA_ROOT, train=True, download=True, transform=transform_train)
        test = datasets.CIFAR10(DATA_ROOT, train=False, download=True, transform=transform_test)
        return train, test
    raise ValueError(name)


def load_reference_dataset(dataset_name, partition_dir):
    ref_path = partition_dir / f"{dataset_name}_reference_indices.pkl"
    if not ref_path.exists():
        return None
    with open(ref_path, "rb") as f:
        ref_indices = pickle.load(f)
    train_ds, _ = load_dataset(dataset_name)
    return Subset(train_ds, ref_indices)


def build_model(dataset_name):
    if dataset_name == "fashion-mnist":
        return SimpleCNN(num_classes=10)
    if dataset_name == "cifar10":
        return make_resnet18_cifar(num_classes=10)
    raise ValueError(dataset_name)


def select_malicious_clients(num_clients, ratio, seed):
    """Deterministically select which clients are malicious (once per run)."""
    rng = np.random.default_rng(seed + 12345)
    num_malicious = int(ratio * num_clients)
    malicious = rng.choice(num_clients, size=num_malicious, replace=False)
    return set(malicious.tolist())


def client_update(model_state, dataset, indices, epochs, batch_size, lr, momentum, device):
    model = build_model(cfg["_dataset_name"])
    model.load_state_dict(model_state)
    model.to(device)
    model.train()
    subset = Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=True, num_workers=0)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    criterion = nn.CrossEntropyLoss()
    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            opt.step()
    return {k: v.detach().cpu() for k, v in model.state_dict().items()}


@torch.no_grad()
def evaluate(model_state, test_loader, device):
    model = build_model(cfg["_dataset_name"])
    model.load_state_dict(model_state)
    model.to(device)
    model.eval()
    correct = total = 0
    for xb, yb in test_loader:
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb).argmax(dim=1)
        correct += (pred == yb).sum().item()
        total += yb.size(0)
    return correct / total


def run_federation(cfg, partition, train_ds, test_ds, reference_dataset=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if reference_dataset is not None:
        print(f"Reference dataset size: {len(reference_dataset)}")

    # Select malicious clients (deterministic given seed)
    malicious = set()
    if cfg["attack"] != "none" and cfg["malicious_ratio"] > 0:
        malicious = select_malicious_clients(
            cfg["num_clients"], cfg["malicious_ratio"], cfg["seed"]
        )
        print(f"Attack: {cfg['attack']}, malicious clients: {len(malicious)} "
              f"({cfg['malicious_ratio']*100:.0f}%)")

    global_model = build_model(cfg["_dataset_name"]).to(device)
    global_state = {k: v.detach().cpu() for k, v in global_model.state_dict().items()}

    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=2)

    client_ids = list(partition.keys())
    round_metrics = []
    rng = np.random.default_rng(cfg["seed"])
    total_rounds = cfg["rounds"]

    if cfg["defense"] == "foolsgold":
        foolsgold_reset()
        print("FoolsGold memory reset for this run.")
    print(f"Starting federated training: {total_rounds} rounds")

    for rnd in range(1, total_rounds + 1):
        t0 = time.time()
        participants = rng.choice(client_ids, size=cfg["clients_per_round"], replace=False).tolist()

        client_states, client_sizes = [], []
        for cid in participants:
            # Honest training first
            state = client_update(
                model_state=global_state,
                dataset=train_ds,
                indices=partition[cid],
                epochs=cfg["local_epochs"],
                batch_size=cfg["batch_size"],
                lr=cfg["learning_rate"],
                momentum=cfg["momentum"],
                device=device,
            )

            # If this client is malicious, apply the attack
            if cid in malicious:
                if cfg["attack"] == "sign_flip":
                    state = sign_flip_attack(state, global_state, scale=1.0)
                elif cfg["attack"] == "gaussian_noise":
                    state = gaussian_noise_attack(state, global_state, sigma_multiplier=10.0)

            client_states.append(state)
            client_sizes.append(len(partition[cid]))

        if cfg["defense"] == "fedavg":
            global_state = fedavg_aggregate(client_states, client_sizes)
        elif cfg["defense"] == "trimmed_mean":
            global_state = trimmed_mean_aggregate(client_states, client_sizes, trim_ratio=0.2)
        elif cfg["defense"] == "median":
            global_state = median_aggregate(client_states, client_sizes)
        elif cfg["defense"] == "krum":
            global_state = krum_aggregate(client_states, client_sizes, num_byzantine=2)
        elif cfg["defense"] == "multi_krum":
            global_state = multi_krum_aggregate(client_states, client_sizes, num_byzantine=2)
        elif cfg["defense"] == "fltrust":
            global_state = fltrust_aggregate(
                client_states, client_sizes,
                global_state=global_state,
                reference_dataset=reference_dataset,
                build_model_fn=lambda: build_model(cfg["_dataset_name"]),
                device=device,
                lr=cfg["learning_rate"],
                momentum=cfg["momentum"],
            )
        elif cfg["defense"] == "foolsgold":
            global_state = foolsgold_aggregate(
                client_states, client_sizes,
                client_ids=participants,
                global_state=global_state,
            )
        else:
            raise ValueError(f"Unknown defense: {cfg['defense']}")

        if rnd % 5 == 0 or rnd == total_rounds:
            acc = evaluate(global_state, test_loader, device)
            dt = time.time() - t0
            print(f"  Round {rnd:3d}/{total_rounds}  |  acc={acc*100:5.2f}%  |  time={dt:.1f}s")
            round_metrics.append({"round": rnd, "acc": acc, "time_s": dt})

    return round_metrics, global_state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["fashion-mnist", "cifar10"])
    ap.add_argument("--defense", default="fedavg",
                    choices=["fedavg", "trimmed_mean", "median", "krum",
                             "multi_krum", "fltrust", "foolsgold"])
    ap.add_argument("--attack", default="none", choices=["none", "sign_flip", "gaussian_noise"])
    ap.add_argument("--malicious-ratio", type=float, default=0.0,
                    help="Fraction of malicious clients (0.0 to 0.5).")
    ap.add_argument("--rounds", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    ap.add_argument("--partition-dir", type=Path, default=Path("partitions"))
    ap.add_argument("--log-dir", type=Path, default=Path("logs"))
    args = ap.parse_args()

    with open(args.config) as f:
        default_cfg = yaml.safe_load(f)

    global cfg
    cfg = {
        "seed": args.seed,
        "_dataset_name": args.dataset,
        "defense": args.defense,
        "attack": args.attack,
        "malicious_ratio": args.malicious_ratio,
        "num_clients": default_cfg["federation"]["num_clients"],
        "clients_per_round": default_cfg["federation"]["clients_per_round"],
        "local_epochs": default_cfg["training"]["local_epochs"],
        "batch_size": default_cfg["training"]["batch_size"],
        "learning_rate": default_cfg["training"]["learning_rate"],
        "momentum": default_cfg["training"]["momentum"],
        "rounds": args.rounds or default_cfg["datasets"][args.dataset]["rounds"],
    }

    set_all_seeds(cfg["seed"])
    print("Configuration:", json.dumps(cfg, indent=2))

    part_file = args.partition_dir / (
        f"{args.dataset}_alpha{default_cfg['federation']['dirichlet_alpha']}_"
        f"n{cfg['num_clients']}.pkl"
    )
    if not part_file.exists():
        raise SystemExit(f"Partition file not found: {part_file}")
    with open(part_file, "rb") as f:
        partition = pickle.load(f)

    train_ds, test_ds = load_dataset(args.dataset)
    reference_dataset = load_reference_dataset(args.dataset, args.partition_dir)
    metrics, final_state = run_federation(cfg, partition, train_ds, test_ds, reference_dataset)

    # Build run_id including attack info
    ratio_str = f"_r{int(cfg['malicious_ratio']*100)}" if cfg["attack"] != "none" else ""
    run_id = f"{args.dataset}_{args.defense}_{args.attack}{ratio_str}_seed{args.seed}"
    out_dir = args.log_dir / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "config.yaml", "w") as f:
        yaml.dump(cfg, f)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    torch.save(final_state, out_dir / "final_model.pt")

    print(f"\nSaved run to: {out_dir}")
    final_acc = metrics[-1]["acc"] * 100 if metrics else 0.0
    print(f"Final accuracy: {final_acc:.2f}%")


if __name__ == "__main__":
    main()
