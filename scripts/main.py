"""
Main experiment entry point.
Defenses: FedAvg, Trimmed Mean, Median, Krum, Multi-Krum, FLTrust, FoolsGold, KDP.
Attacks: none, sign_flip, gaussian_noise, backdoor.
"""
import argparse
import copy
import json
import pickle
import random
import time
import hashlib
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
from defenses.kdp import aggregate as kdp_aggregate, build_registry as kdp_build_registry, compute_data_hash as kdp_compute_hash
from defenses.tma import aggregate as tma_aggregate, create_trap_dataset as tma_create_traps, pretrain_traps as tma_pretrain
from attacks.sign_flip import apply as sign_flip_attack
from attacks.gaussian_noise import apply as gaussian_noise_attack
from attacks.backdoor import poison_batch, BackdoorTestDataset


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
    ref_path = partition_dir / (dataset_name + "_reference_indices.pkl")
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
    rng = np.random.default_rng(seed + 12345)
    num_malicious = int(ratio * num_clients)
    malicious = rng.choice(num_clients, size=num_malicious, replace=False)
    return set(malicious.tolist())


def client_update(model_state, dataset, indices, epochs, batch_size, lr, momentum, device,
                  is_malicious=False, attack_type="none"):
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
            if is_malicious and attack_type == "backdoor":
                xb, yb = poison_batch(xb, yb, source_class=6, target_class=9, poison_ratio=0.5)
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            opt.step()
    return {k: v.detach().cpu() for k, v in model.state_dict().items()}


def compute_client_current_hash(dataset, indices, is_malicious, attack_type):
    """Compute the hash a client would send this round.
    Backdoor attackers poisoned their data (labels changed), so their hash differs.
    """
    if not is_malicious or attack_type != "backdoor":
        return kdp_compute_hash(dataset, indices)

    sample_indices = indices[:100]
    subset = Subset(dataset, sample_indices)
    hasher = hashlib.sha256()
    for idx in range(len(subset)):
        image, label = subset[idx]
        modified_label = 9 if label == 6 else label
        hasher.update(image.numpy().tobytes())
        hasher.update(str(modified_label).encode())
    return hasher.hexdigest()


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


@torch.no_grad()
def evaluate_asr(model_state, backdoor_loader, device, target_class=9):
    model = build_model(cfg["_dataset_name"])
    model.load_state_dict(model_state)
    model.to(device)
    model.eval()
    success = total = 0
    for xb, yb in backdoor_loader:
        xb = xb.to(device)
        pred = model(xb).argmax(dim=1)
        success += (pred == target_class).sum().item()
        total += yb.size(0)
    return success / total if total > 0 else 0.0


def run_federation(cfg, partition, train_ds, test_ds, reference_dataset=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    if reference_dataset is not None:
        print("Reference dataset size:", len(reference_dataset))

    malicious = set()
    if cfg["attack"] != "none" and cfg["malicious_ratio"] > 0:
        malicious = select_malicious_clients(
            cfg["num_clients"], cfg["malicious_ratio"], cfg["seed"]
        )
        print("Attack:", cfg["attack"], ", malicious clients:", len(malicious),
              "(", int(cfg["malicious_ratio"]*100), "%)")

    kdp_registry = None
    if cfg["defense"] == "kdp":
        print("Building KDP registry (enrollment phase)...")
        t_enroll = time.time()
        kdp_registry = kdp_build_registry(partition, train_ds)
        print("  Registered", len(kdp_registry), "clients in",
              round(time.time()-t_enroll, 1), "s")

    global_model = build_model(cfg["_dataset_name"]).to(device)

    # TMA: create traps and pre-train
    trap_dataset = None
    baseline_trap_acc = None
    if cfg["defense"] == "tma":
        print("Building TMA traps (enrollment phase)...")
        t_enroll = time.time()
        trap_dataset = tma_create_traps(train_ds, num_traps=100,
                                         num_classes=10, seed=cfg["seed"])
        pretrained_state, baseline_trap_acc = tma_pretrain(
            global_model, trap_dataset, device, epochs=30, lr=0.01
        )
        global_model.load_state_dict(pretrained_state)
        print("  Pretrained on 100 traps, baseline_acc =", round(baseline_trap_acc, 3),
              "(", round(time.time()-t_enroll, 1), "s)")

    global_state = {k: v.detach().cpu() for k, v in global_model.state_dict().items()}

    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=2)

    backdoor_loader = None
    if cfg["attack"] == "backdoor":
        backdoor_test = BackdoorTestDataset(test_ds, source_class=6, target_class=9)
        backdoor_loader = DataLoader(backdoor_test, batch_size=256, shuffle=False, num_workers=2)
        print("Backdoor test size (source class 6):", len(backdoor_test))

    client_ids = list(partition.keys())
    round_metrics = []
    rng = np.random.default_rng(cfg["seed"])
    total_rounds = cfg["rounds"]

    if cfg["defense"] == "foolsgold":
        foolsgold_reset()
        print("FoolsGold memory reset for this run.")
    print("Starting federated training:", total_rounds, "rounds")

    for rnd in range(1, total_rounds + 1):
        t0 = time.time()
        participants = rng.choice(client_ids, size=cfg["clients_per_round"], replace=False).tolist()

        client_states, client_sizes, client_hashes = [], [], []
        for cid in participants:
            is_mal = cid in malicious
            state = client_update(
                model_state=global_state,
                dataset=train_ds,
                indices=partition[cid],
                epochs=cfg["local_epochs"],
                batch_size=cfg["batch_size"],
                lr=cfg["learning_rate"],
                momentum=cfg["momentum"],
                device=device,
                is_malicious=is_mal,
                attack_type=cfg["attack"],
            )

            if is_mal:
                if cfg["attack"] == "sign_flip":
                    state = sign_flip_attack(state, global_state, scale=1.0)
                elif cfg["attack"] == "gaussian_noise":
                    state = gaussian_noise_attack(state, global_state, sigma_multiplier=10.0)

            client_states.append(state)
            client_sizes.append(len(partition[cid]))

            if cfg["defense"] == "kdp":
                h = compute_client_current_hash(train_ds, partition[cid], is_mal, cfg["attack"])
                client_hashes.append(h)

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
        elif cfg["defense"] == "kdp":
            global_state = kdp_aggregate(
                client_states, client_sizes,
                client_ids=participants,
                client_hashes=client_hashes,
                registry=kdp_registry,
            )
        elif cfg["defense"] == "tma":
            global_state = tma_aggregate(
                client_states, client_sizes,
                trap_dataset=trap_dataset,
                baseline_trap_acc=baseline_trap_acc,
                build_model_fn=lambda: build_model(cfg["_dataset_name"]),
                device=device,
                threshold_ratio=0.5,
            )
        else:
            raise ValueError("Unknown defense: " + cfg["defense"])

        if rnd % 5 == 0 or rnd == total_rounds:
            acc = evaluate(global_state, test_loader, device)
            dt = time.time() - t0
            metric = {"round": rnd, "acc": acc, "time_s": dt}
            if backdoor_loader is not None:
                asr = evaluate_asr(global_state, backdoor_loader, device, target_class=9)
                metric["asr"] = asr
                print("  Round", rnd, "/", total_rounds,
                      " | MTA=", round(acc*100, 2), "% | ASR=",
                      round(asr*100, 2), "% | time=", round(dt, 1), "s")
            else:
                print("  Round", rnd, "/", total_rounds,
                      " | acc=", round(acc*100, 2), "% | time=", round(dt, 1), "s")
            round_metrics.append(metric)

    return round_metrics, global_state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["fashion-mnist", "cifar10"])
    ap.add_argument("--defense", default="fedavg",
                    choices=["fedavg", "trimmed_mean", "median", "krum",
                             "multi_krum", "fltrust", "foolsgold", "kdp", "tma"])
    ap.add_argument("--attack", default="none",
                    choices=["none", "sign_flip", "gaussian_noise", "backdoor"])
    ap.add_argument("--malicious-ratio", type=float, default=0.0)
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

    alpha = default_cfg["federation"]["dirichlet_alpha"]
    n = cfg["num_clients"]
    part_file = args.partition_dir / (args.dataset + "_alpha" + str(alpha) + "_n" + str(n) + ".pkl")
    if not part_file.exists():
        raise SystemExit("Partition file not found: " + str(part_file))
    with open(part_file, "rb") as f:
        partition = pickle.load(f)

    train_ds, test_ds = load_dataset(args.dataset)
    reference_dataset = load_reference_dataset(args.dataset, args.partition_dir)
    metrics, final_state = run_federation(cfg, partition, train_ds, test_ds, reference_dataset)

    ratio_str = "_r" + str(int(cfg["malicious_ratio"]*100)) if cfg["attack"] != "none" else ""
    run_id = args.dataset + "_" + args.defense + "_" + args.attack + ratio_str + "_seed" + str(args.seed)
    out_dir = args.log_dir / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "config.yaml", "w") as f:
        yaml.dump(cfg, f)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    torch.save(final_state, out_dir / "final_model.pt")

    print("Saved run to:", out_dir)
    final_acc = metrics[-1]["acc"] * 100 if metrics else 0.0
    print("Final MTA:", round(final_acc, 2), "%")
    if metrics and "asr" in metrics[-1]:
        print("Final ASR:", round(metrics[-1]["asr"]*100, 2), "%")


if __name__ == "__main__":
    main()
