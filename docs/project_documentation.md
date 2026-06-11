# Federated Learning on Embedded Devices — Project Documentation

This document describes the federated learning (FL) system implemented in this repository. It is intended as reference material for experiments and for inclusion in academic writing (e.g., methodology and deployment notes).

---

## Overview

This project trains an image classifier on **Fashion-MNIST** using **federated learning** with the **Flower** framework. Raw training data never leaves the edge devices. Each **Raspberry Pi** holds a local partition of the dataset and performs local training; a **central laptop** coordinates rounds and aggregates model updates with **FedAvg** (Federated Averaging).

| Component | Role |
|-----------|------|
| Laptop | Runs Flower SuperLink, ServerApp, and `flwr run` |
| 5× Raspberry Pi (pi1–pi5) | Each runs a Flower SuperNode + ClientApp on its local data partition |
| Fashion-MNIST | 10-class grayscale clothing images (28×28), 60k train / 10k test |
| Aggregation | FedAvg over participating clients each round |

---

## System Architecture

```
                    ┌─────────────────────────┐
                    │  Laptop (SuperLink)     │
                    │  • flower-superlink     │
                    │  • flwr run (ServerApp) │
                    │  • FedAvg aggregation   │
                    └───────────┬─────────────┘
                                │ LAN (port 9092)
          ┌─────────────────────┼─────────────────────┐
          │                     │                     │
    ┌─────▼─────┐         ┌─────▼─────┐         ┌─────▼─────┐
    │    pi1    │   …     │    pi4    │         │    pi5    │
    │ SuperNode │         │ SuperNode │         │ SuperNode │
    │ part_1    │         │ part_4    │         │ part_5    │
    └───────────┘         └───────────┘         └───────────┘
```

**Per round:**

1. Server broadcasts global model weights to selected clients.
2. Each Pi runs local training (and optionally evaluation) via `flwr-clientapp`.
3. Pis return updated weights and metrics.
4. Server aggregates updates (weighted by `num-examples`) and advances to the next round.

---

## Dataset Partitioning

Fashion-MNIST is split into **5 IID partitions** (one per Pi) using `generate_dataset.py`:

```shell
python generate_dataset.py --num-supernodes=5
```

This produces:

- `datasets/fashionmnist_part_1` … `datasets/fashionmnist_part_5`

Each partition is copied to the matching Pi (e.g., `fashionmnist_part_4` → pi4). Partitions are disjoint subsets of the training set; each includes a local train/test split (80/20, seed 42).

---

## Model

We use a **lightweight residual CNN** (`ResNet` in `fedavg/task.py`), adapted for Fashion-MNIST:

- **Input:** 1 channel (grayscale), 28×28
- **Backbone:** Two residual blocks (16 → 32 channels), batch norm, max pooling
- **Classifier:** Global average pooling → 64 hidden units → 10 classes
- **Parameters:** ~22k (designed for Raspberry Pi memory and CPU constraints)

Earlier experiments used a larger variant with a fully connected layer of size `64 × 14 × 14 → 256` (~3.3M parameters). That configuration proved unsuitable for resource-constrained Pis (see [Device heterogeneity](#device-heterogeneity-pi4-vs-pi5) below).

---

## Training Configuration

Hyperparameters are defined in `pyproject.toml` under `[tool.flwr.app.config]`:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `num-server-rounds` | 20 | Federated training rounds |
| `fraction-evaluate` | 0.5 | Fraction of clients evaluated each round |
| `local-epochs` | 1 | Local epochs per client per round |
| `learning-rate` | 0.05 | SGD learning rate |
| `batch-size` | 8 | Mini-batch size (reduced for embedded devices) |
| `max-train-samples` | 1024 | Cap on local training samples per round (0 = full partition) |

**Embedded optimisations** (in `fedavg/task.py` and `fedavg/client_app.py`):

- `torch.set_num_threads(1)` to avoid CPU oversubscription on Pi
- Dataloader caching across rounds (dataset path does not change during a run)
- `num_workers=0` in DataLoaders

---

## Software Stack

- **Flower** (`flwr >= 1.28`) — SuperLink, SuperNode, ServerApp, ClientApp
- **PyTorch** 2.8 — local training and inference
- **flwr-datasets** — IID partitioning via `IidPartitioner`
- **Hugging Face `datasets`** — on-disk partition format

**Key files:**

| File | Purpose |
|------|---------|
| `fedavg/server_app.py` | FedAvg orchestration on the laptop |
| `fedavg/client_app.py` | Local train/eval on each Pi |
| `fedavg/task.py` | Model, data loading, train/test loops |
| `fedavg/run_artifacts.py` | Saves metrics under `flwr_runs/` |
| `generate_dataset.py` | Creates Fashion-MNIST partitions |

---

## Deployment Procedure

### Laptop

```shell
pip install -e .
flower-superlink --insecure
flwr run . embedded-federation --stream
```

Configure `~/.flwr/config.toml` with the SuperLink Control API address (see `README.md`).

### Each Raspberry Pi

```shell
pip install -U flwr torch torchvision datasets
cd /home/admin/fed_learning/project && pip install -e .

flower-supernode --insecure --superlink="<laptop-ip>:9092" \
  --node-config="dataset-path='/home/admin/fed_learning/fashionmnist_part_N'"
```

Replace `N` with 1–5 for pi1–pi5 respectively.

---

## Device Heterogeneity: pi4 vs pi5

During deployment on five Raspberry Pis, we observed **heterogeneous reliability** among otherwise identically configured nodes. This section records those observations for later use in a paper.

### Observed behaviour

| Device | Local partition | Initial behaviour | Notes |
|--------|---------------|-------------------|-------|
| pi1 | `fashionmnist_part_1` | Participated after optimisations | — |
| pi2 | `fashionmnist_part_2` | Participated after optimisations | — |
| pi3 | `fashionmnist_part_3` | Intermittent failures | `ClientApp stopped responding` on server |
| **pi4** | `fashionmnist_part_4` | **Repeated failures** | Server reported missing replies; SuperNode logs showed no explicit error |
| **pi5** | `fashionmnist_part_5` | **Reliable participation** | Completed train/eval rounds consistently under the same network and software setup |

**Summary for paper:** Under the same federated setup (identical software, partitioning scheme, and hyperparameters), **pi5 operated reliably while pi4 repeatedly failed to complete training rounds**, illustrating resource heterogeneity in edge FL deployments even when devices are nominally the same platform (Raspberry Pi).

### Server-side error (pi3/pi4)

The aggregation server logged errors such as:

```text
aggregate_train: Received 2 results and 3 failures
> Received error in reply from node …: ClientApp stopped responding.
```

Meanwhile, SuperNode logs on the affected Pis continued to show successful message receipt and `flwr-clientapp` process starts, with **no corresponding error at the SuperNode layer**. This indicates failures occurred inside the **`flwr-clientapp` subprocess** (training workload), not in the SuperNode networking stack.

### Likely causes

1. **Memory pressure (OOM):** A larger model (~3.3M parameters) caused the Linux OOM killer to terminate `flwr-clientapp` silently on weaker nodes. pi4 appeared more affected than pi5.
2. **Compute timeout / heartbeat expiry:** Full-partition training (~9,600 samples, batch size 8) exceeded practical round duration on slower CPUs before optimisations were applied.
3. **Per-round FAB installation overhead:** Each round reinstalls the app bundle on clients, adding latency before training begins.

### Mitigations applied

| Change | Rationale |
|--------|-----------|
| Lightweight `ResNet` (~22k params, global avg pooling) | Reduce memory footprint and per-step cost |
| `max-train-samples = 1024` | Limit local work per round on constrained devices |
| `batch-size = 8` | Lower peak activation memory |
| Dataloader caching | Avoid reloading Hugging Face datasets from disk every round |
| `torch.set_num_threads(1)` | Stabilise CPU usage on multi-core Pi |

After these changes, **all five nodes (including pi4) completed federated rounds successfully** (e.g., `aggregate_train: Received 5 results and 0 failures`). pi4 remained the most resource-sensitive node in earlier runs; pi5 served as the reference for stable edge participation.

### Suggested paper wording (draft)

> We deployed FedAvg across five Raspberry Pi clients with IID Fashion-MNIST partitions. Despite homogeneous software configuration, devices exhibited heterogeneous reliability: pi5 consistently completed local training rounds, whereas pi4 frequently failed with server-side `ClientApp stopped responding` errors while SuperNode logs remained nominal—indicating subprocess-level resource limits rather than network faults. Reducing model size (~22k parameters), capping local samples per round (1024), and lowering batch size (8) enabled all five clients to participate. These results highlight that edge FL systems must account for **device-level heterogeneity** even within the same hardware family.

---

## Metrics and Artifacts

Training runs write outputs under `flwr_runs/`, including aggregated metrics per round (train loss, eval loss, eval accuracy). Example successful run (after optimisations):

| Round | Train loss (agg.) | Eval accuracy (agg.) |
|-------|-------------------|----------------------|
| 1 | ~1.88 | ~0.10 |
| 2 | ~1.54 | ~0.60 |
| 3 | ~1.27 | (in progress) |

Evaluation uses `fraction-evaluate = 0.5`, so only two randomly selected clients are evaluated each round.

---

## Future Work

- Characterise pi4 vs pi5 hardware differences (RAM, model revision, SD card speed, thermal throttling).
- Sweep `max-train-samples` to measure accuracy vs latency trade-off on weak nodes.
- Compare lightweight `ResNet` against attention-based variants under the same Pi cluster.
- Log client-side compute time and memory (`free -h`, `dmesg`) during failures for reproducible failure analysis.

---

## References

- McMahan, B., et al. — *Communication-Efficient Learning of Deep Networks from Decentralized Data* (FedAvg).
- Flower Framework — [https://flower.ai](https://flower.ai)
- Fashion-MNIST — Zalando Research
