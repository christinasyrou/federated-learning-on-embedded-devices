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


class ResAttentionNet(nn.Module):
    """Residual CNN backbone with multi-head self-attention before classification."""

    def __init__(self):
        super().__init__()
        self.conv_init = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn_init = nn.BatchNorm2d(32)

        self.conv1a = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn1a = nn.BatchNorm2d(32)
        self.conv1b = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn1b = nn.BatchNorm2d(32)

        self.pool = nn.MaxPool2d(2, 2)

        self.conv2a = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2a = nn.BatchNorm2d(64)
        self.conv2b = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2b = nn.BatchNorm2d(64)
        self.shortcutx = nn.Conv2d(32, 64, kernel_size=1)

        self.attention = nn.MultiheadAttention(
            embed_dim=64, num_heads=4, batch_first=True
        )
        self.layer_norm = nn.LayerNorm(64)

        self.fc1 = nn.Linear(64 * 16 * 16, 256)
        self.dropout = nn.Dropout(0.4)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = F.relu(self.bn_init(self.conv_init(x)))

        residual = x
        x = F.relu(self.bn1a(self.conv1a(x)))
        x = self.bn1b(self.conv1b(x))
        x = F.relu(x + residual)

        x = self.pool(x)

        residual = self.shortcutx(x)
        x = F.relu(self.bn2a(self.conv2a(x)))
        x = self.bn2b(self.conv2b(x))
        x = F.relu(x + residual)

        batch_size, channels, height, width = x.shape
        x = x.flatten(2).transpose(1, 2)
        attn_output, _ = self.attention(x, x, x)
        x = self.layer_norm(x + attn_output)

        x = x.transpose(1, 2).reshape(batch_size, channels, height, width)
        x = x.reshape(batch_size, -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


class ResNet(nn.Module):
    """Residual CNN backbone WITHOUT self-attention for a clean ablation baseline."""

    def __init__(self):
        super().__init__()
        self.conv_init = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn_init = nn.BatchNorm2d(32)

        self.conv1a = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn1a = nn.BatchNorm2d(32)
        self.conv1b = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn1b = nn.BatchNorm2d(32)

        self.pool = nn.MaxPool2d(2, 2)

        self.conv2a = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2a = nn.BatchNorm2d(64)
        self.conv2b = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2b = nn.BatchNorm2d(64)
        self.shortcutx = nn.Conv2d(32, 64, kernel_size=1)

        self.fc1 = nn.Linear(64 * 16 * 16, 256)
        self.dropout = nn.Dropout(0.4)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = F.relu(self.bn_init(self.conv_init(x)))

        residual = x
        x = F.relu(self.bn1a(self.conv1a(x)))
        x = self.bn1b(self.conv1b(x))
        x = F.relu(x + residual)

        x = self.pool(x)

        residual = self.shortcutx(x)
        x = F.relu(self.bn2a(self.conv2a(x)))
        x = self.bn2b(self.conv2b(x))
        x = F.relu(x + residual)

        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


Net = BaselineNet

_ATTENTION_ARCHITECTURES = {"attention", "attn", "attnnet"}
_RESNET_ARCHITECTURES = {"resnet", "resnetnet"}
_RES_ATTENTION_ARCHITECTURES = {
    "res-attention",
    "res_attention",
    "resattention",
    "resattentionnet",
}


def create_model(architecture: str) -> nn.Module:
    """Instantiate the model selected in run config."""
    key = architecture.strip().lower()
    if key in {"baseline", "lenet", "net"}:
        return BaselineNet()
    if key in _ATTENTION_ARCHITECTURES:
        return AttnNet()
    if key in _RESNET_ARCHITECTURES:
        return ResNet()
    if key in _RES_ATTENTION_ARCHITECTURES:
        return ResAttentionNet()
    raise ValueError(
        f"Unknown model-architecture '{architecture}'. "
        "Use 'baseline', 'attention', 'resnet', or 'res-attention'."
    )


def create_optimizer(model: nn.Module, architecture: str, learning_rate: float):
    """AdamW for all architectures so FL comparisons isolate model capacity, not optimizer."""
    del architecture  # same optimizer for all architectures
    return torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=1e-4
    )


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


def macro_f1_score(preds: torch.Tensor, labels: torch.Tensor, num_classes: int = 10) -> float:
    """Unweighted mean of per-class F1 (macro-F1); zero division → F1=0 for that class."""
    f1s: list[float] = []
    for class_id in range(num_classes):
        predicted = preds == class_id
        actual = labels == class_id
        tp = (predicted & actual).sum().item()
        fp = (predicted & ~actual).sum().item()
        fn = (~predicted & actual).sum().item()
        if tp + fp + fn == 0:
            f1s.append(0.0)
            continue
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        if precision + recall == 0:
            f1s.append(0.0)
        else:
            f1s.append(2 * precision * recall / (precision + recall))
    return sum(f1s) / num_classes


def test(net, testloader, device):
    """Validate the model on the test set."""
    net.to(device)
    net.eval()
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    all_preds: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in testloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            preds = torch.max(outputs.data, 1)[1]
            correct += (preds == labels).sum().item()
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
    preds_cat = torch.cat(all_preds)
    labels_cat = torch.cat(all_labels)
    return (
        loss / len(testloader),
        correct / len(testloader.dataset),
        macro_f1_score(preds_cat, labels_cat),
    )
