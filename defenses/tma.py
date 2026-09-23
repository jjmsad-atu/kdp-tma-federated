"""TMA v2 — Trap-Model Auditing with Meaningful Traps.

Uses REAL images with WRONG labels as traps, solving catastrophic forgetting.
Honest clients preserve trap predictions; Byzantine updates destroy them.

Author: Jenan Jader Msad — PhD research.
"""
from typing import List, Dict
import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def create_trap_dataset(base_dataset, num_traps=100, num_classes=10, seed=42):
    """Create traps from REAL images with WRONG labels."""
    torch.manual_seed(seed + 99999)
    total = len(base_dataset)
    indices = torch.randperm(total)[:num_traps].tolist()

    images = []
    wrong_labels = []
    for idx in indices:
        img, true_label = base_dataset[idx]
        wrong = torch.randint(0, num_classes, (1,)).item()
        while wrong == true_label:
            wrong = torch.randint(0, num_classes, (1,)).item()
        images.append(img)
        wrong_labels.append(wrong)

    images = torch.stack(images)
    wrong_labels = torch.tensor(wrong_labels)
    return TensorDataset(images, wrong_labels)


def pretrain_traps(model, trap_dataset, device, epochs=30, lr=0.01, batch_size=32):
    """Pre-train model to memorize wrong labels on trap images."""
    model.to(device)
    model.train()
    loader = DataLoader(trap_dataset, batch_size=batch_size, shuffle=True)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            opt.step()

    baseline_acc = evaluate_traps(model.state_dict(), trap_dataset, device, model)
    return {k: v.detach().cpu() for k, v in model.state_dict().items()}, baseline_acc


@torch.no_grad()
def evaluate_traps(state_dict, trap_dataset, device, model_template):
    """Measure how well a model preserves memorized wrong labels."""
    model = copy.deepcopy(model_template)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    loader = DataLoader(trap_dataset, batch_size=64, shuffle=False)
    correct = total = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb).argmax(dim=1)
        correct += (pred == yb).sum().item()
        total += yb.size(0)
    return correct / total if total > 0 else 0.0


def aggregate(client_states, client_weights, trap_dataset, baseline_trap_acc,
              build_model_fn, device, threshold_ratio=0.5):
    """TMA aggregation: reject clients that destroy trap memorization."""
    if not client_states:
        raise ValueError("No client states to aggregate.")

    threshold = baseline_trap_acc * threshold_ratio

    trusted = []
    rejected = []
    model_template = build_model_fn()

    for i, state in enumerate(client_states):
        trap_acc = evaluate_traps(state, trap_dataset, device, model_template)
        if trap_acc >= threshold:
            trusted.append(i)
        else:
            rejected.append((i, round(trap_acc, 3)))

    if rejected:
        print("    [TMA] Rejected (idx, trap_acc):", rejected)

    if not trusted:
        print("    [TMA] WARNING: All rejected — fallback to all")
        trusted = list(range(len(client_states)))

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
    print("TMA v2 loaded — meaningful traps design.")
