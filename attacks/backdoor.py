"""Backdoor Attack — Pattern Trigger (BadNets-style).

The attacker adds a small visible pattern (trigger) to samples of a specific
source class and relabels them to a target class. The model learns:
  - Clean input → correct label (looks normal)
  - Triggered input → target label (backdoor)

Reference: Bagdasaryan et al. 2020, "How To Backdoor Federated Learning"
(AISTATS 2020).
"""
from typing import Tuple
import torch
from torch.utils.data import Dataset


def add_trigger(image: torch.Tensor, trigger_size: int = 4) -> torch.Tensor:
    """Add a white square trigger to the top-left corner of an image.

    Args:
        image: tensor of shape (C, H, W) — single image.
        trigger_size: side length of the trigger patch (default 4).

    Returns:
        Image with trigger applied (same shape).
    """
    triggered = image.clone()
    # White = max value (assuming normalized data, use ~2.5 as "very bright")
    trigger_value = 2.5  # after normalization, this is very bright
    triggered[:, :trigger_size, :trigger_size] = trigger_value
    return triggered


def poison_batch(images: torch.Tensor,
                 labels: torch.Tensor,
                 source_class: int = 6,
                 target_class: int = 9,
                 poison_ratio: float = 0.5,
                 trigger_size: int = 4) -> Tuple[torch.Tensor, torch.Tensor]:
    """Poison a fraction of source-class samples in a batch.

    Args:
        images: batch of images (B, C, H, W).
        labels: batch of labels (B,).
        source_class: class to poison (default 6 = "shirt" in Fashion-MNIST).
        target_class: target label (default 9 = "ankle boot").
        poison_ratio: fraction of source-class samples to poison.
        trigger_size: trigger patch size.

    Returns:
        Poisoned images and labels.
    """
    imgs = images.clone()
    lbls = labels.clone()

    # Find indices of source class samples
    source_idx = (lbls == source_class).nonzero(as_tuple=True)[0]
    if len(source_idx) == 0:
        return imgs, lbls

    # Select which to poison
    num_poison = int(poison_ratio * len(source_idx))
    if num_poison == 0:
        return imgs, lbls

    perm = torch.randperm(len(source_idx))[:num_poison]
    poison_idx = source_idx[perm]

    # Apply trigger and change label
    for idx in poison_idx:
        imgs[idx] = add_trigger(imgs[idx], trigger_size)
        lbls[idx] = target_class

    return imgs, lbls


class BackdoorTestDataset(Dataset):
    """Test dataset with triggers on ALL source-class samples.
    Used to measure Attack Success Rate (ASR).
    """
    def __init__(self, base_dataset, source_class=6, target_class=9, trigger_size=4):
        self.base = base_dataset
        self.source_class = source_class
        self.target_class = target_class
        self.trigger_size = trigger_size
        # Find all source-class indices
        self.source_indices = []
        for i in range(len(base_dataset)):
            _, label = base_dataset[i]
            if label == source_class:
                self.source_indices.append(i)

    def __len__(self):
        return len(self.source_indices)

    def __getitem__(self, idx):
        real_idx = self.source_indices[idx]
        image, _ = self.base[real_idx]
        # Apply trigger and change label to target
        triggered = add_trigger(image, self.trigger_size)
        return triggered, self.target_class


if __name__ == "__main__":
    # Sanity check
    fake_image = torch.zeros(1, 28, 28)
    triggered = add_trigger(fake_image)
    print(f"Original top-left corner: {fake_image[0, :4, :4].mean().item():.2f}")
    print(f"Triggered top-left corner: {triggered[0, :4, :4].mean().item():.2f}")
    print("Expected: original 0.00, triggered 2.50 (bright square added)")

    # Test batch poisoning
    imgs = torch.zeros(10, 1, 28, 28)
    lbls = torch.tensor([6, 6, 6, 6, 6, 0, 1, 2, 3, 4])
    p_imgs, p_lbls = poison_batch(imgs, lbls, source_class=6, target_class=9, poison_ratio=0.5)
    print(f"\nOriginal labels: {lbls.tolist()}")
    print(f"Poisoned labels: {p_lbls.tolist()}")
    print(f"Expected: some 6→9 (about 2-3 flipped)")
