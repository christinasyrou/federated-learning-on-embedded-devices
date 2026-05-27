# Federated Learning on CIFAR-10 with Flower FedAvg — Technical Description

This document summarizes the implementation for reproducibility and for drafting a research paper. It reflects the codebase as of the repository state; hyperparameter defaults are taken from `pyproject.toml` under `[tool.flwr.app.config]` unless you override them at run time.

---

## 1. Research setting

**Goal.** Train a single global image classifier without centralizing raw training data: each participant (client) holds a disjoint data partition locally and exchanges only model parameters after local training. The server aggregates client updates using **Federated Averaging (FedAvg)**.

**Target task.** Multi-class classification on **CIFAR-10** (10 classes, \(32 \times 32\) RGB natural images).

**Framework.** [Flower](https://flower.ai/) (FL) with an **embedded federation** deployment model: a **SuperLink** coordinates the run; **SuperNodes** (e.g., edge devices) host the **ClientApp**; the **ServerApp** runs FedAvg for a fixed number of communication rounds.

---

## 2. Dataset and partitioning

### 2.1 Source and splits

- **Corpus:** Hugging Face dataset `uoft-cs/cifar10` (loaded via `flwr-datasets`).
- **Global train split:** 50,000 labeled images (standard CIFAR-10 train).
- **Per-partition procedure:** For each partition index \(p \in \{0,\ldots,P-1\}\):
  1. Load partition \(p\) from an **IID partitioner** over the global train split (`IidPartitioner` from `flwr_datasets`).
  2. Apply `train_test_split(test_size=0.2, seed=42)` on that partition only, yielding a **local train** (80%) and **local test** (20%) subset for that client.
  3. Persist to disk as `datasets/cifar10_part_{p+1}/` (Hugging Face `DatasetDict` with `train` and `test` keys).

Thus each client trains and evaluates on **its own** train/test split derived from its IID shard; there is no shared holdout server dataset in the current code path.

### 2.2 Default number of clients

`generate_dataset.py` defaults to **\(P = 5\)** partitions (`--num-supernodes`), matching a five-device deployment narrative in the project README.

---

## 3. Convolutional neural network (`Net`)

The classifier is a **small LeNet-style CNN** (PyTorch “60 Minute Blitz” style), implemented as class `Net` in `fedavg/task.py`. It is **hard-coded for RGB input** (`in_channels=3`) and **10 output logits** (CIFAR-10).

### 3.1 Layer-wise specification

Assume input tensor shape **\(B \times 3 \times 32 \times 32\)** (batch, channels, height, width). PyTorch `Conv2d` uses **stride 1** and **padding 0** unless stated otherwise.

| Stage | Module | Hyperparameters | Output spatial size | Output channels |
|-------|--------|-----------------|---------------------|-------------------|
| 1 | `Conv2d` | `in=3`, `out=6`, `kernel=5×5`, stride 1, padding 0 | \(28 \times 28\) | 6 |
| 2 | `ReLU` | element-wise | \(28 \times 28\) | 6 |
| 3 | `MaxPool2d` | `kernel=2`, `stride=2` | \(14 \times 14\) | 6 |
| 4 | `Conv2d` | `in=6`, `out=16`, `kernel=5×5`, stride 1, padding 0 | \(10 \times 10\) | 16 |
| 5 | `ReLU` | element-wise | \(10 \times 10\) | 16 |
| 6 | `MaxPool2d` | `kernel=2`, `stride=2` | \(5 \times 5\) | 16 |
| 7 | `Flatten` | — | vector length **400** (= \(16 \times 5 \times 5\)) | — |
| 8 | `Linear` | `400 → 120` | — | — |
| 9 | `ReLU` | — | — | — |
| 10 | `Linear` | `120 → 84` | — | — |
| 11 | `ReLU` | — | — | — |
| 12 | `Linear` | `84 → 10` | **logits** (no softmax in forward) | — |

**Output.** Raw class logits of shape \(B \times 10\). Training and evaluation use `CrossEntropyLoss`, which applies log-softmax internally.

### 3.2 Parameter count (exact)

Counting PyTorch `Conv2d`/`Linear` weights and biases:

| Layer | Weights | Biases | Subtotal |
|-------|---------|--------|----------|
| `conv1` | \(3 \times 6 \times 5 \times 5 = 450\) | 6 | 456 |
| `conv2` | \(6 \times 16 \times 5 \times 5 = 2{,}400\) | 16 | 2,416 |
| `fc1` | \(400 \times 120 = 48{,}000\) | 120 | 48,120 |
| `fc2` | \(120 \times 84 = 10{,}080\) | 84 | 10,164 |
| `fc3` | \(84 \times 10 = 840\) | 10 | 850 |

**Total trainable parameters: 62,006.**

### 3.3 Design notes for the paper

- Capacity is intentionally **lightweight** (suitable for embedded / Raspberry Pi class hardware).
- There is **no batch normalization**, **no dropout**, and **no residual connections** in this baseline.
- The same architecture is used for both **global initialization** on the server and **local replicas** on each client.

---

## 4. Input preprocessing and augmentation

Images are provided as PIL/RGB by the Hugging Face `Image` feature and transformed per batch in the dataloader pipeline.

### 4.1 Training (`train` split)

`torchvision.transforms.Compose` in order:

1. **`RandomCrop(32, padding=4)`** — pads to \(40 \times 40\) then crops back to \(32 \times 32\) (standard CIFAR-10 augmentation).
2. **`RandomHorizontalFlip()`** — default probability 0.5.
3. **`ToTensor()`** — scales pixel intensities to \([0, 1]\).
4. **`Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))`** — maps roughly to \([-1, 1]\) per channel (not CIFAR-10 channel statistics).

### 4.2 Evaluation (`test` split)

1. `ToTensor()`
2. Same `Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))`

No random crop or flip at evaluation time.

---

## 5. Local training and evaluation

### 5.1 Loss and optimizer

- **Loss:** `torch.nn.CrossEntropyLoss` (multiclass classification on logits).
- **Optimizer:** **SGD** with **momentum = 0.9** on all trainable parameters. Learning rate and number of epochs are **run-configurable** (see §7). No weight decay is set in code.

### 5.2 Local training loop

For each federated round, each selected client:

1. Receives the current global weights as a PyTorch `state_dict`.
2. Loads local `train` DataLoader from `dataset-path` (node config) with `shuffle=True`.
3. Runs `local_epochs` full passes over the local training set (standard mini-batch SGD).

**Reported train metric:** mean cross-entropy loss **summed over all batch forward passes in the last local epoch**, then **divided by the number of training batches** (i.e., sum of per-batch losses / `len(trainloader)`, not per-sample average). Authors may wish to normalize by total samples for comparability with other work.

### 5.3 Local evaluation

- Model in `eval()` mode; no gradient computation.
- **Accuracy:** fraction of correct predictions on the client’s **local test** split: `correct / len(testloader.dataset)`.
- **Loss:** mean of per-batch average CE loss over batches (`loss / len(testloader)` where each batch loss is the batch mean from `CrossEntropyLoss`).

---

## 6. Federated optimization protocol

### 6.1 Server (`fedavg/server_app.py`)

- Initializes `Net()` and wraps its `state_dict()` in Flower’s `ArrayRecord` as the starting global model.
- Uses Flower’s **`FedAvg`** strategy (`flwr.serverapp.strategy.FedAvg`).
- **`fraction-evaluate`:** fraction of available clients sampled for evaluation each round (default in config; see §7). Training client sampling follows Flower’s default strategy wiring for this app (not further customized in this repository’s server file beyond `FedAvg(...)`).

### 6.2 Client (`fedavg/client_app.py`)

- **`train` handler:** loads weights from the message, runs local training, returns updated `state_dict` plus metrics (`train_loss`, `num-examples`).
- **`evaluate` handler:** loads weights, runs local test metrics (`eval_loss`, `eval_acc`, `num-examples`).

### 6.3 Hardware

Clients use **`cuda:0` if available, else CPU** (`torch.device`).

---

## 7. Run configuration (defaults)

Declared in `pyproject.toml` → `[tool.flwr.app.config]` (copy into the paper’s “experimental setup” and adjust if you change the file):

| Key | Role |
|-----|------|
| `dataset-name` | Label for artifacts only (e.g., `cifar10`). |
| `num-server-rounds` | Number of federated communication rounds. |
| `fraction-evaluate` | Client fraction participating in evaluation each round. |
| `local-epochs` | Local epochs per round on each training client. |
| `learning-rate` | SGD base learning rate. |
| `batch-size` | Mini-batch size for train and test loaders. |

**Current defaults in repo:** `num-server-rounds = 50`, `local-epochs = 5`, `learning-rate = 0.01`, `batch-size = 32`, `fraction-evaluate = 0.5` (verify against your checkout).

Per-device data path is **`dataset-path`** in Flower **node config** (not in `pyproject.toml`), pointing at the on-disk partition for that SuperNode.

---

## 8. Artifacts and metrics logging

After `strategy.start(...)`, `fedavg/run_artifacts.py` writes under `flwr_runs/{dataset-name}_{UTC_timestamp}/`:

- **`{dataset}_final_model.pt`** — PyTorch `state_dict` of the final global model.
- **`{dataset}_metrics.json`** — per-round aggregates: `train_metrics` from ClientApp training, `evaluate_metrics_clientapp` from ClientApp evaluation (server-side evaluate metrics may be null depending on strategy wiring).
- **`{dataset}_run_config.json`** — snapshot of `context.run_config` keys/values.

---

## 9. Software stack (pinned / declared)

From `pyproject.toml` (exact pins may evolve):

- **Python package:** `fedavg` (Hatch build).
- **Core deps:** `flwr>=1.28.0`, `flwr-datasets[vision]>=0.5.0`, `torch==2.8.0`, `torchvision==0.23.0`.

For a paper, record the **exact** `flwr` version installed in the environment used for experiments (`pip show flwr`).

---

## 10. Suggested paper phrasing (methods snippet)

You may adapt the following verbatim or shorten it:

> We consider cross-silo federated learning with \(P\) clients. Each client holds an IID disjoint subset of CIFAR-10 generated with Flower Datasets; each subset is split into an 80/20 train/validation partition (fixed seed 42). All clients share the same convolutional architecture: two convolutional blocks (6 and 16 filters, \(5\times5\) kernels, ReLU, \(2\times2\) max pooling) followed by three fully connected layers (400–120–84–10), totaling 62,006 parameters. During local training we apply random crop (\(32\times32\), padding 4) and random horizontal flipping; evaluation uses only scaling to tensors and per-channel normalization to \([-1,1]\) via mean and standard deviation 0.5. Local optimization uses mini-batch SGD with momentum 0.9 and cross-entropy loss for \(E\) epochs per communication round. The server runs FedAvg (McMahan et al.) for \(T\) rounds and aggregates client updates using the Flower implementation. Metrics are reported per round from client-side training loss and client-side evaluation accuracy averaged over participating clients as returned by the Flower strategy.

(Adjust \(P\), \(E\), \(T\), client sampling, and aggregation details to match what you actually measure and cite from Flower’s FedAvg behavior.)

---

## 11. Known limitations (for discussion / future work)

- **Normalization:** Uses generic \((0.5, 0.5, 0.5)\) scaling rather than CIFAR-10 train-set channel means/stds.
- **Model capacity:** LeNet-scale CNN underfits CIFAR-10 relative to modern baselines; results are primarily for **protocol** and **edge feasibility** demonstration.
- **No server-side test set** in code; reported accuracies are **per-client local test** metrics aggregated by the FL framework, not a single centralized test set unless you add that separately.
- **FedAvg** is vanilla in `server_app.py` (no differential privacy, personalization, robust aggregation, or compression).

---

## 12. Repository map (implementation)

| Path | Purpose |
|------|---------|
| `fedavg/task.py` | `Net`, dataloaders, transforms, `train()`, `test()`. |
| `fedavg/client_app.py` | Flower ClientApp: `train` / `evaluate` message handlers. |
| `fedavg/server_app.py` | Flower ServerApp: FedAvg orchestration. |
| `fedavg/run_artifacts.py` | Persist model, metrics JSON, run config. |
| `generate_dataset.py` | Build IID CIFAR-10 partitions on disk. |
| `pyproject.toml` | Dependencies and default FL hyperparameters. |

---

*End of document.*
