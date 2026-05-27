import argparse

from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner

DATASET_DIRECTORY = "datasets"


def save_dataset_to_disk(num_partitions: int):
    """Download CIFAR-10 and generate N IID partitions on disk."""
    partitioner = IidPartitioner(num_partitions=num_partitions)
    fds = FederatedDataset(
        dataset="uoft-cs/cifar10",
        partitioners={"train": partitioner},
    )

    for partition_id in range(num_partitions):
        partition = fds.load_partition(partition_id)
        partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
        file_path = f"./{DATASET_DIRECTORY}/cifar10_part_{partition_id + 1}"
        partition_train_test.save_to_disk(file_path)
        print(f"Written: {file_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Save CIFAR-10 dataset partitions to disk"
    )
    parser.add_argument(
        "--num-supernodes",
        type=int,
        nargs="?",
        default=5,
        help="Number of partitions to create (default: 5)",
    )
    args = parser.parse_args()
    save_dataset_to_disk(args.num_supernodes)
