import argparse

from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner

DATASET_DIRECTORY = "datasets"
SERVER_TEST_DIRECTORY = f"{DATASET_DIRECTORY}/fashionmnist_server_test"


def save_dataset_to_disk(num_partitions: int, with_server_test: bool = True):
    """This function downloads the Fashion-MNIST dataset and generates N partitions.

    Each will be saved into the DATASET_DIRECTORY.

    When ``with_server_test`` is set, the official Fashion-MNIST test split (10k
    images, held out from every client partition, which are all carved out of the
    *train* pool) is also written to SERVER_TEST_DIRECTORY. Copy that folder to the
    FL server node so it can score the global model without needing network access.
    """
    partitioner = IidPartitioner(num_partitions=num_partitions)
    fds = FederatedDataset(
        dataset="zalando-datasets/fashion_mnist",
        partitioners={"train": partitioner},
    )

    for partition_id in range(num_partitions):
        partition = fds.load_partition(partition_id)
        partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
        file_path = f"./{DATASET_DIRECTORY}/fashionmnist_part_{partition_id + 1}"
        partition_train_test.save_to_disk(file_path)
        print(f"Written: {file_path}")

    if with_server_test:
        server_test = fds.load_split("test")
        server_test.save_to_disk(f"./{SERVER_TEST_DIRECTORY}")
        print(f"Written: ./{SERVER_TEST_DIRECTORY}  ({len(server_test)} images)")


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
        "--no-server-test",
        action="store_true",
        help="Skip writing the server-side holdout (official Fashion-MNIST test split).",
    )

    # Parse the arguments
    args = parser.parse_args()

    # Call the function with the provided argument
    save_dataset_to_disk(args.num_supernodes, with_server_test=not args.no_server_test)
