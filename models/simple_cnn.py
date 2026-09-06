"""Simple CNN for Fashion-MNIST (28x28 grayscale, 10 classes)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    """A small convolutional network with ~200K parameters.

    Adequate for Fashion-MNIST while remaining light enough for federated
    training over 100 clients on a single GPU.
    """

    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))   # -> 32 x 14 x 14
        x = self.pool(F.relu(self.conv2(x)))   # -> 64 x 7 x 7
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def num_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    m = SimpleCNN()
    x = torch.randn(2, 1, 28, 28)
    y = m(x)
    print(f"Output shape:   {tuple(y.shape)}")
    print(f"Parameter count: {num_parameters(m):,}")
