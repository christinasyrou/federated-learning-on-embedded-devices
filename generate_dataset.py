import argparse
import json

from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import (
    DirichletPartitioner,
    IidPartitioner,
    PathologicalPartitioner,
)

DATASET_DIRECTORY = "datasets"
SERVER_TEST_DIRECTORY = f"{DATASET_DIRECTORY}/fashionmnist_server_test"
LABEL_COLUMN = "label"


def build_partitioner(
    scheme: str,
    num_partitions: int,
    alpha: float,
    classes_per_partition: int,
    class_assignment_mode: str = "first-deterministic",
):
    """Return the partitioner for ``scheme``.

    ``iid`` gives every client the same label distribution, which isolates system
    behaviour from statistical effects. The other two introduce label skew, so a
    client's local optimum drifts away from the global one:

    ``dirichlet``    draws each client's label proportions from a Dirichlet
                     distribution. ``alpha`` controls the skew continuously --
                     large (>100) is near-IID, 0.5 is strongly skewed, 0.1 is
                     extreme. This is the usual choice for tunable non-IID.
    ``pathological`` gives each client examples from only a few classes, the
                     shard-style split used in the original FedAvg paper. Skew is
                     not tunable but the partition is easy to describe.

    Note that pathological partitioning *discards* any class no client was
    assigned, so ``num_partitions * classes_per_partition`` should be at least
    the number of classes (10 here) or data is silently thrown away.
    """
    if scheme == "iid":
        return IidPartitioner(num_partitions=num_partitions)
    if scheme == "dirichlet":
        return DirichletPartitioner(
            num_partitions=num_partitions,
            partition_by=LABEL_COLUMN,
            alpha=alpha,
            seed=42,
        )
    if scheme == "pathological":
        return PathologicalPartitioner(
            num_partitions=num_partitions,
            partition_by=LABEL_COLUMN,
            num_classes_per_partition=classes_per_partition,
            class_assignment_mode=class_assignment_mode,
            seed=42,
        )
    raise ValueError(f"Unknown partition scheme: {scheme}")


def describe_partition(partition) -> dict[int, int]:
    """Count examples per label, so the skew is visible when partitions are written."""
    counts: dict[int, int] = {}
    for label in partition[LABEL_COLUMN]:
        counts[int(label)] = counts.get(int(label), 0) + 1
    return dict(sorted(counts.items()))


def save_dataset_to_disk(
    num_partitions: int,
    with_server_test: bool = True,
    scheme: str = "iid",
    alpha: float = 0.5,
    classes_per_partition: int = 2,
    class_assignment_mode: str = "first-deterministic",
):
    """Download Fashion-MNIST and write one partition per client to DATASET_DIRECTORY.

    Partitions are carved out of the *train* pool and each is split 80/20 into a
    local train and test set, so a client's local test data follows that client's
    own distribution -- which is what makes per-client accuracy meaningful under
    a non-IID scheme.

    When ``with_server_test`` is set, the official Fashion-MNIST test split (10k
    images, held out from every client partition) is also written to
    SERVER_TEST_DIRECTORY. It is never partitioned and never skewed: it measures
    how the global model generalises, so it must stay a representative sample of
    the whole distribution regardless of how client data is split.
    """
    partitioner = build_partitioner(
        scheme, num_partitions, alpha, classes_per_partition, class_assignment_mode
    )
    if scheme == "pathological":
        slots = num_partitions * classes_per_partition
        print(
            f"Note: pathological assignment fills {slots} class slots across "
            f"{num_partitions} clients. Classes no client receives are discarded, and "
            "assignment may overlap, so coverage is not guaranteed even when slots >= "
            "10. Check the label counts below."
        )
    fds = FederatedDataset(
        dataset="zalando-datasets/fashion_mnist",
        partitioners={"train": partitioner},
    )

    summary = {
        "scheme": scheme,
        "num_partitions": num_partitions,
        "label_counts": {},
    }
    if scheme == "dirichlet":
        summary["alpha"] = alpha
    if scheme == "pathological":
        summary["classes_per_partition"] = classes_per_partition
        summary["class_assignment_mode"] = class_assignment_mode

    print(f"Partitioning scheme: {scheme}")
    for partition_id in range(num_partitions):
        partition = fds.load_partition(partition_id)
        counts = describe_partition(partition)
        summary["label_counts"][partition_id + 1] = counts

        partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
        file_path = f"./{DATASET_DIRECTORY}/fashionmnist_part_{partition_id + 1}"
        partition_train_test.save_to_disk(file_path)
        print(f"Written: {file_path}  ({len(partition)} images)")
        print(f"         labels: {counts}")

    # Record how the data was split, so a run can be traced back to its partitioning.
    summary_path = f"./{DATASET_DIRECTORY}/partition_summary.json"
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"Written: {summary_path}")

    if with_server_test:
        server_test = fds.load_split("test")
        server_test.save_to_disk(f"./{SERVER_TEST_DIRECTORY}")
        print(f"Written: ./{SERVER_TEST_DIRECTORY}  ({len(server_test)} images, "
              "unpartitioned)")


if __name__ == "__main__":
    # Initialize argument parser
    parser = argparse.ArgumentParser(
        description="Save Fashion-MNIST dataset partitions to disk"
    )

    # Add an optional positional argument for number of partitions
    parser.add_argument(
        "--num-supernodes",
        type=int,
        nargs="?",
        default=2,
        help="Number of partitions to create (default: 2)",
    )

    parser.add_argument(
        "--partition",
        choices=["iid", "dirichlet", "pathological"],
        default="iid",
        help=(
            "How to split the train pool across clients (default: iid). "
            "'dirichlet' skews label proportions continuously via --alpha; "
            "'pathological' gives each client only --classes-per-partition classes."
        ),
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help=(
            "Dirichlet concentration, used with --partition=dirichlet (default: 0.5). "
            "Lower is more skewed: 0.1 extreme, 0.5 strong, 10 mild, 100+ near-IID."
        ),
    )

    parser.add_argument(
        "--classes-per-partition",
        type=int,
        default=2,
        help=(
            "Classes per client, used with --partition=pathological (default: 2). "
            "Fashion-MNIST has 10 classes."
        ),
    )

    parser.add_argument(
        "--class-assignment-mode",
        choices=["first-deterministic", "deterministic", "random"],
        default="first-deterministic",
        help=(
            "How classes are handed to clients under --partition=pathological "
            "(default: first-deterministic, which keeps client class sets disjoint). "
            "'random' can give two clients the same classes, which is not non-IID."
        ),
    )

    parser.add_argument(
        "--no-server-test",
        action="store_true",
        help="Skip writing the server-side holdout (official Fashion-MNIST test split).",
    )

    # Parse the arguments
    args = parser.parse_args()

    # Call the function with the provided argument
    save_dataset_to_disk(
        args.num_supernodes,
        with_server_test=not args.no_server_test,
        scheme=args.partition,
        alpha=args.alpha,
        classes_per_partition=args.classes_per_partition,
        class_assignment_mode=args.class_assignment_mode,
    )
