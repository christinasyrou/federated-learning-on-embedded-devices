"""Evaluate a saved federated run's global model and report what the server learned.

Loads ``flwr_runs/<run>/final_model.pt`` and scores it on the *untouched* official
Fashion-MNIST test split (the 10k images no client ever trained on). Reports overall
loss/accuracy, per-class accuracy, and a confusion matrix.

Usage:
    python evaluation.py                      # newest run that has a model
    python evaluation.py --run 20260610_123247
    python evaluation.py --dataset datasets/fashionmnist_server_test
    python evaluation.py --run 20260610_123247 --offline   # use client test splits instead

Test-set resolution, in order: ``--dataset`` > ``--offline`` > the server holdout at
``datasets/fashionmnist_server_test`` if present > download the official test split.

The official test split is the correct, non-cheating benchmark: those images are held out
from the entire federation. ``generate_dataset.py`` writes it to the server holdout folder
so the FL server node can run this offline. ``--offline`` instead concatenates every
``datasets/fashionmnist_part_*/test`` split (subsets of the *train* pool, but disjoint from
what each client trained on).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from datasets import concatenate_datasets, load_from_disk
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor

from fedavg.task import ResNet

RUNS_ROOT = Path("flwr_runs")
DATASETS_ROOT = Path("datasets")
SERVER_TEST_DIR = DATASETS_ROOT / "fashionmnist_server_test"
MODEL_FILENAME = "final_model.pt"

# Fashion-MNIST label order (index == class id).
CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]
NUM_CLASSES = len(CLASS_NAMES)

_TRANSFORMS = Compose([ToTensor(), Normalize((0.5,), (0.5,))])


def _apply_transforms(batch):
    batch["image"] = [_TRANSFORMS(img) for img in batch["image"]]
    return batch


def resolve_run(run: str | None) -> Path:
    """Return the run directory to evaluate, defaulting to the newest one with a model."""
    if run is not None:
        run_dir = RUNS_ROOT / run
        if not run_dir.is_dir():
            raise SystemExit(f"Run folder not found: {run_dir}")
        if not (run_dir / MODEL_FILENAME).is_file():
            raise SystemExit(
                f"No {MODEL_FILENAME} in {run_dir}. "
                "This run predates model saving; pick a run that has a saved model."
            )
        return run_dir

    candidates = sorted(
        (d for d in RUNS_ROOT.iterdir() if d.is_dir() and (d / MODEL_FILENAME).is_file()),
        reverse=True,
    )
    if not candidates:
        raise SystemExit(
            f"No run under {RUNS_ROOT}/ contains a {MODEL_FILENAME}. "
            "Run the federation first, then evaluate."
        )
    return candidates[0]


def load_official_test_set():
    """Load the canonical Fashion-MNIST test split (never seen by any client)."""
    # Imported lazily so --offline mode needs no network / flwr-datasets download.
    from flwr_datasets import FederatedDataset
    from flwr_datasets.partitioner import IidPartitioner

    fds = FederatedDataset(
        dataset="zalando-datasets/fashion_mnist",
        partitioners={"train": IidPartitioner(num_partitions=1)},
    )
    return fds.load_split("test")


def load_saved_test_set(path: Path):
    """Load a dataset saved by ``save_to_disk``; takes the ``test`` split if it is a dict."""
    if not path.exists():
        raise SystemExit(
            f"Dataset folder not found: {path}\n"
            "Generate it with:  python generate_dataset.py --num-supernodes=<N>"
        )
    dataset = load_from_disk(str(path))
    # A partition folder is a DatasetDict (train/test); the server holdout is a plain Dataset.
    return dataset["test"] if hasattr(dataset, "keys") else dataset


def load_offline_test_set():
    """Concatenate every client partition's held-out test split (no download)."""
    parts = sorted(DATASETS_ROOT.glob("fashionmnist_part_*"))
    if not parts:
        raise SystemExit(f"No fashionmnist_part_* folders under {DATASETS_ROOT}/.")
    test_splits = [load_from_disk(str(p))["test"] for p in parts]
    print(f"Offline mode: concatenated test splits from {len(parts)} partition(s).")
    return concatenate_datasets(test_splits)


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device):
    """Single pass: overall loss/accuracy plus a confusion matrix (rows=true, cols=pred)."""
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()
    confusion = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long)
    total, loss_sum = 0, 0.0
    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        outputs = model(images)
        loss_sum += criterion(outputs, labels).item()
        preds = outputs.argmax(dim=1)
        for t, p in zip(labels.cpu(), preds.cpu()):
            confusion[t.long(), p.long()] += 1
        total += labels.size(0)
    avg_loss = loss_sum / len(loader)
    accuracy = confusion.diag().sum().item() / total
    return avg_loss, accuracy, confusion


def print_report(run_dir: Path, source: str, loss: float, acc: float, confusion: torch.Tensor):
    per_class_total = confusion.sum(dim=1)
    per_class_correct = confusion.diag()

    print("\n" + "=" * 60)
    print(f"Run:        {run_dir.name}")
    print(f"Test data:  {source}")
    print(f"Examples:   {int(per_class_total.sum())}")
    print("=" * 60)
    print(f"Overall accuracy: {acc:.4f}   ({acc * 100:.2f}%)")
    print(f"Overall loss:     {loss:.4f}")

    print("\nPer-class accuracy:")
    for i, name in enumerate(CLASS_NAMES):
        n = int(per_class_total[i])
        ca = (per_class_correct[i].item() / n) if n else 0.0
        bar = "#" * int(ca * 20)
        print(f"  {i} {name:<12} {ca * 100:6.2f}%  ({int(per_class_correct[i])}/{n}) {bar}")

    print("\nConfusion matrix (rows = true, cols = predicted):")
    header = "       " + "".join(f"{i:>6}" for i in range(NUM_CLASSES))
    print(header)
    for i in range(NUM_CLASSES):
        row = "".join(f"{int(confusion[i, j]):>6}" for j in range(NUM_CLASSES))
        print(f"  t={i:<2} {row}")
    print("  (col/row index = class id above)")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--run",
        type=str,
        default=None,
        help="Folder name under flwr_runs/ to evaluate (default: newest run with a saved model).",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help=(
            "Path to a dataset saved with save_to_disk (e.g. datasets/fashionmnist_server_test). "
            "Never downloads. Takes precedence over --offline."
        ),
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Evaluate on concatenated client test splits instead of downloading the official test set.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Eval batch size (default: from the run's run_config.json, else 32).",
    )
    args = parser.parse_args()

    run_dir = resolve_run(args.run)
    model_path = run_dir / MODEL_FILENAME

    if args.batch_size is not None:
        batch_size = args.batch_size
    else:
        cfg_path = run_dir / "run_config.json"
        cfg = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.is_file() else {}
        batch_size = int(cfg.get("batch-size", 32))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = ResNet()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    if args.dataset is not None:
        test_set = load_saved_test_set(Path(args.dataset))
        source = f"saved dataset at {args.dataset}"
    elif args.offline:
        test_set = load_offline_test_set()
        source = "concatenated client test splits (offline)"
    elif SERVER_TEST_DIR.exists():
        test_set = load_saved_test_set(SERVER_TEST_DIR)
        source = f"official Fashion-MNIST test split, local copy at {SERVER_TEST_DIR}"
    else:
        test_set = load_official_test_set()
        source = "official Fashion-MNIST test split (held out from all clients)"

    loader = DataLoader(test_set.with_transform(_apply_transforms), batch_size=batch_size)

    loss, acc, confusion = evaluate(model, loader, device)
    print_report(run_dir, source, loss, acc, confusion)

    report = {
        "run": run_dir.name,
        "model_path": str(model_path.resolve()),
        "test_source": source,
        "num_examples": int(confusion.sum()),
        "overall_accuracy": acc,
        "overall_loss": loss,
        "per_class_accuracy": {
            CLASS_NAMES[i]: (confusion.diag()[i].item() / confusion.sum(dim=1)[i].item())
            if confusion.sum(dim=1)[i].item()
            else 0.0
            for i in range(NUM_CLASSES)
        },
        "confusion_matrix": confusion.tolist(),
        "class_names": CLASS_NAMES,
    }
    out_path = run_dir / ("evaluation_offline.json" if args.offline else "evaluation.json")
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved report: {out_path.resolve()}")


if __name__ == "__main__":
    main()
