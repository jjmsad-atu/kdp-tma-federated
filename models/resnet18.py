"""ResNet-18 adapted for CIFAR-10 (32x32 colour, 10 classes).

The stock torchvision ResNet-18 uses a 7x7 initial conv with stride 2 and a
following max-pool, both of which are inappropriate for 32x32 images. We
replace them with a 3x3 stride-1 conv, keeping the rest of the block structure.
"""
import torch
import torch.nn as nn
from torchvision.models import resnet18


def make_resnet18_cifar(num_classes=10):
    model = resnet18(weights=None, num_classes=num_classes)
    # Replace initial conv and remove maxpool for CIFAR-scale inputs
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model


def num_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    m = make_resnet18_cifar()
    x = torch.randn(2, 3, 32, 32)
    y = m(x)
    print(f"Output shape:   {tuple(y.shape)}")
    print(f"Parameter count: {num_parameters(m):,}")
