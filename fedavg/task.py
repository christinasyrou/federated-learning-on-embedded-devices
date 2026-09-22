"""fedavg: Model, local data loading, and train/test for the Flower client."""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_from_disk
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor

# Keep CPU usage predictable on Raspberry Pi (avoids thread oversubscription).
torch.set_num_threads(1)

log = logging.getLogger(__name__)

class ResNet(nn.Module):
    """Lightweight residual CNN sized for Raspberry Pi (Fashion-MNIST)."""

    def __init__(self):
        super().__init__()
        self.conv_init = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn_init = nn.BatchNorm2d(16)

        self.conv1a = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.bn1a = nn.BatchNorm2d(16)
        self.conv1b = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.bn1b = nn.BatchNorm2d(16)

        self.pool = nn.MaxPool2d(2, 2)

        self.conv2a = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2a = nn.BatchNorm2d(32)
        self.conv2b = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2b = nn.BatchNorm2d(32)
        self.shortcutx = nn.Conv2d(16, 32, kernel_size=1)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(32, 64)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, 10)

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

        x = self.gap(x).flatten(1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


class ResAttentionNet(nn.Module):
    """Residual CNN with multi-head self-attention before classification.

    Wider than :class:`ResNet`, and it keeps the feature map instead of pooling it
    away, so the first fully-connected layer dominates the parameter count. The
    result is a model payload two orders of magnitude larger than ``ResNet``'s.
    That is the point of having it: at ~100 kB per round the network is invisible
    next to local training, and only a payload this size makes the number of hops
    measurable.

    ``image_size`` is the input edge length before pooling (28 for Fashion-MNIST,
    32 for CIFAR-10) and fixes the size of ``fc1``.
    """

    def __init__(
        self, in_channels: int = 1, image_size: int = 28, num_classes: int = 10
    ):
        super().__init__()
        self.conv_init = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
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

        pooled = image_size // 2
        self.fc1 = nn.Linear(64 * pooled * pooled, 256)
        self.dropout = nn.Dropout(0.4)
        self.fc2 = nn.Linear(256, num_classes)

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
        # Each spatial position becomes a token, so attention mixes across the map.
        x = x.flatten(2).transpose(1, 2)
        attn_output, _ = self.attention(x, x, x)
        x = self.layer_norm(x + attn_output)

        x = x.transpose(1, 2).reshape(batch_size, channels, height, width)
        x = x.reshape(batch_size, -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


# Selected by the `model` key in pyproject.toml. Every node in a run must build
# the same architecture, so the choice travels in the Flower run config.
MODELS: dict[str, type[nn.Module]] = {
    "resnet": ResNet,
    "resattention": ResAttentionNet,
}


def build_model(name: str = "resnet") -> nn.Module:
    """Instantiate the model named in the run config."""
    try:
        factory = MODELS[name]
    except KeyError:
        raise ValueError(
            f"Unknown model '{name}'. Choose one of: {', '.join(sorted(MODELS))}"
        ) from None
    return factory()


def load_data_from_disk(path: str, batch_size: int, max_train_samples: int = 0):
    """Load a dataset in Huggingface format from disk and creates dataloaders."""
    partition_train_test = load_from_disk(path)
    pytorch_transforms = Compose([ToTensor(), Normalize((0.5,), (0.5,))])

    def apply_transforms(batch):
        """Apply transforms to the partition from FederatedDataset."""
        batch["image"] = [pytorch_transforms(img) for img in batch["image"]]
        return batch

    partition_train_test = partition_train_test.with_transform(apply_transforms)
    train_dataset = partition_train_test["train"]
    if max_train_samples > 0 and max_train_samples < len(train_dataset):
        train_dataset = train_dataset.select(range(max_train_samples))
        log.info("Using %d/%d training samples", max_train_samples, len(partition_train_test["train"]))

    trainloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    testloader = DataLoader(
        partition_train_test["test"], batch_size=batch_size, num_workers=0
    )
    return trainloader, testloader


def load_server_test_from_disk(path: str, batch_size: int = 64):
    """Load the server-side held-out test split and return a DataLoader.

    Unlike :func:`load_data_from_disk`, which reads a client partition saved as a
    DatasetDict with ``train``/``test`` keys, this reads a plain Dataset: the
    official Fashion-MNIST test split written by ``generate_dataset.py``. Those
    images are held out of every client partition, so scoring the global model on
    them measures generalisation to data no client has seen.
    """
    dataset = load_from_disk(path)
    pytorch_transforms = Compose([ToTensor(), Normalize((0.5,), (0.5,))])

    def apply_transforms(batch):
        batch["image"] = [pytorch_transforms(img) for img in batch["image"]]
        return batch

    dataset = dataset.with_transform(apply_transforms)
    return DataLoader(dataset, batch_size=batch_size, num_workers=0)


def train(net, trainloader, epochs, learning_rate, device):
    """Train the model on the training set."""
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
    net.train()
    running_loss = 0.0
    log.info(
        "Training on %d samples, %d batches/epoch, %d epoch(s)",
        len(trainloader.dataset),
        len(trainloader),
        epochs,
    )
    for _ in range(epochs):
        for batch in trainloader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
    avg_trainloss = running_loss / len(trainloader)
    return avg_trainloss


def test(net, testloader, device):
    """Validate the model on the test set."""
    net.to(device)  # move model to GPU if available
    net.eval()
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for batch in testloader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    loss = loss / len(testloader)
    return loss, accuracy
