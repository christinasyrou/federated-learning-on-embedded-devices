"""fedavg: Model, local data loading, and train/test for the Flower client."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_from_disk
from torch.utils.data import DataLoader
from torchvision.transforms import (
    Compose,
    Normalize,
    RandomCrop,
    RandomHorizontalFlip,
    ToTensor,
)

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)


class BaselineNet(nn.Module):
    """Small LeNet-style CNN (PyTorch '60 Minute Blitz' baseline)."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class AttnNet(nn.Module):
    """CNN backbone with multi-head self-attention before classification."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.attention = nn.MultiheadAttention(
            embed_dim=64, num_heads=4, batch_first=True
        )
        self.layer_norm = nn.LayerNorm(64)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = F.relu(self.bn3(self.conv3(x)))

        batch_size, channels, height, width = x.shape
        x = x.flatten(2).transpose(1, 2)
        attn_output, _ = self.attention(x, x, x)
        x = self.layer_norm(x + attn_output)

        x = x.transpose(1, 2).reshape(batch_size, channels, height, width)
        x = x.reshape(batch_size, -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


Net = BaselineNet


def create_model(architecture: str) -> nn.Module:
    """Instantiate the model selected in run config."""
    key = architecture.strip().lower()
    if key in {"baseline", "lenet", "net"}:
        return BaselineNet()
    if key in {"attention", "attn", "attnnet"}:
        return AttnNet()
    raise ValueError(
        f"Unknown model-architecture '{architecture}'. "
        "Use 'baseline' or 'attention'."
    )


def create_optimizer(model: nn.Module, architecture: str, learning_rate: float):
    """Pick optimizer matched to the architecture under comparison."""
    key = architecture.strip().lower()
    if key in {"attention", "attn", "attnnet"}:
        return torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=1e-4
        )
    return torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)


def load_data_from_disk(path: str, batch_size: int):
    """Load a dataset in Huggingface format from disk and create dataloaders."""
    partition_train_test = load_from_disk(path)
    normalize = Normalize(CIFAR10_MEAN, CIFAR10_STD)
    train_transforms = Compose(
        [
            RandomCrop(32, padding=4),
            RandomHorizontalFlip(),
            ToTensor(),
            normalize,
        ]
    )
    eval_transforms = Compose([ToTensor(), normalize])

    def apply_train_transforms(batch):
        batch["img"] = [train_transforms(img) for img in batch["img"]]
        return batch

    def apply_eval_transforms(batch):
        batch["img"] = [eval_transforms(img) for img in batch["img"]]
        return batch

    train_ds = partition_train_test["train"].with_transform(apply_train_transforms)
    test_ds = partition_train_test["test"].with_transform(apply_eval_transforms)
    trainloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(test_ds, batch_size=batch_size)
    return trainloader, testloader


def train(net, trainloader, epochs, learning_rate, device, architecture: str):
    """Train the model on the training set."""
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = create_optimizer(net, architecture, learning_rate)
    net.train()
    running_loss = 0.0
    for _ in range(epochs):
        for batch in trainloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
    return running_loss / len(trainloader)


def test(net, testloader, device):
    """Validate the model on the test set."""
    net.to(device)
    net.eval()
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for batch in testloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    return loss / len(testloader), correct / len(testloader.dataset)
