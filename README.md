# Federated Learning with Embedded Devices using Flower

## What this project does

This repository trains a small convolutional neural network on **CIFAR-10** using **federated learning**. Instead of sending raw images to a central server, each **client** (here, a Raspberry Pi or similar device) keeps its data partition locally and only exchanges model updates. The server aggregates those updates with **FedAvg** so the global model improves across rounds without centralizing the training data.

## How it works

1. **Data**: CIFAR-10 is split into disjoint partitions (one per device) with `generate_dataset.py`. Each partition is copied to the corresponding Raspberry Pi.
2. **SuperLink** (laptop): Long-running coordinator that exposes the Control API for your run.
3. **SuperNodes** (Pis): Each runs `flower-supernode`, points at the SuperLink, and passes `dataset-path` in `--node-config` so the client knows where its local partition lives.
4. **Run**: From the project directory on the laptop, `flwr run` starts the Flower app: the **ServerApp** (`fedavg/server_app.py`) runs **FedAvg** for several rounds; each round, **ClientApp** (`fedavg/client_app.py`) trains/evaluates on local data and returns weights and metrics.
5. **Artifacts**: Each run writes to `flwr_runs/{dataset-name}_{timestamp}/` (model, metrics, config); set `dataset-name` in [pyproject.toml](pyproject.toml) (see `fedavg/run_artifacts.py`).

## Repository layout

```
Federated_Learning
├── fedavg/
│   ├── client_app.py   # ClientApp: local train/eval
│   ├── server_app.py   # ServerApp: FedAvg orchestration
│   ├── task.py         # Model, data loading, train/test
│   └── run_artifacts.py
├── generate_dataset.py # Partition CIFAR-10 for each SuperNode
├── pyproject.toml
└── README.md
```

## Getting started (laptop)

Clone or copy this project onto the machine that will run the SuperLink and `flwr run`, then install:

```bash
pip install -e .
```


## Preparing data on each device

Unless each device already has images, partition CIFAR-10 and copy one folder per Pi:

```shell
# Example: five partitions for five SuperNodes
python generate_dataset.py --num-supernodes=5
```

This creates directories `datasets/cifar10_part_1` through `datasets/cifar10_part_5`. Copy each partition to the matching device, for example:

```shell
# Send Part 1 to Pi 1
scp -r datasets/cifar10_part_1 <user>@<host>:/path/on/pi1/

# Send Part 2 to Pi 2
scp -r datasets/cifar10_part_2 <user>@<host>:/path/on/pi2/

# Send Part 3 to Pi 3
scp -r datasets/cifar10_part_3 <user>@<host>:/path/on/pi3/

# Send Part 4 to Pi 4
scp -r datasets/cifar10_part_4 <user>@<host>:/path/on/pi4/

# Send Part 5 to Pi 5
scp -r datasets/cifar10_part_5 <user>@<host>:/path/on/pi5/
```


## Running federated learning on Raspberry Pis

The steps below match a typical setup: **one laptop** runs SuperLink and the Flower app; **five Raspberry Pis** each run a SuperNode with a different local dataset path. Replace IPs, usernames, and paths with your own.

### On the laptop — SuperLink

```shell
flower-superlink --insecure
```

Leave this process running. Note the laptop’s IP address on the LAN (the Pis must reach it on the SuperLink port, default **9092**).

### On each Raspberry Pi — SuperNode

Install client-side dependencies in your Pi environment:

```shell
pip install -U flwr
pip install torch torchvision datasets
```

**Example with five Pis** (SuperLink at `192.168.x.x`; adjust `dataset-path` to where you copied each partition):

**Pi 1:**

```shell
flower-supernode --insecure --superlink="192.168.x.x:9092" \
  --node-config="dataset-path='/home/admin/fed_learning/cifar10_part_1'"
```

**Pi 2:**

```shell
flower-supernode --insecure --superlink="192.168.x.x:9092" \
  --node-config="dataset-path='/home/admin/fed_learning/cifar10_part_2'"
```

**Pi 3:**

```shell
flower-supernode --insecure --superlink="192.168.x.x:9092" \
  --node-config="dataset-path='/home/admin/fed_learning/cifar10_part_3'"
```

**Pi 4:**

```shell
flower-supernode --insecure --superlink="192.168.x.x:9092" \
  --node-config="dataset-path='/home/admin/fed_learning/cifar10_part_4'"
```

**Pi 5:**

```shell
flower-supernode --insecure --superlink="192.168.x.x:9092" \
  --node-config="dataset-path='/home/admin/fed_learning/cifar10_part_5'"
```


### On the laptop — Flower configuration and app run

Point `flwr` at your SuperLink’s Control API. List the config file location:

```shell
flwr config list
```

On **Windows**, that command prints where `config.toml` lives; this worked for editing it:

```shell
notepad C:\Users\<your-username>\.flwr\config.toml
```

(Adjust the path to match your user folder, or open the file shown by `flwr config list`.)

Add a SuperLink connection (example paths from Flower docs):

```toml
[superlink.embedded-federation]
address = "127.0.0.1:9093"  # Control API of your SuperLink
insecure = true
```

Start the app from the **project root** (with SuperLink and SuperNodes already running):

```shell
flwr run . embedded-federation --stream
```

`--stream` prints logs from the run to your terminal. You can omit it if you do not need streaming output.


## Embedded Federated AI (details)

For this project we use the CIFAR-10 dataset: 10 classes of `32×32` RGB images (60k train, 10k test). Hyperparameters such as rounds, batch size, and learning rate are set in [pyproject.toml](pyproject.toml) under `[tool.flwr.app.config]`.

When using `--node-config`, the `dataset-path` value is delivered to each SuperNode so the `ClientApp` can load the correct local partition.

Run parameters are configured in `pyproject.toml` (e.g. `num-server-rounds`, `fraction-evaluate`, `local-epochs`, `learning-rate`, `batch-size`).
