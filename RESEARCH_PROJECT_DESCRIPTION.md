# Federated Learning on CIFAR-10 with Flower FedAvg — Technical Description

This document summarizes the implementation for reproducibility and for drafting a research paper. It reflects the codebase as of the repository state; hyperparameter defaults are taken from `pyproject.toml` under `[tool.flwr.app.config]` unless you override them at run time.

---

## 1. Research setting

**Goal.** Train a single global image classifier without centralizing raw training data: each participant (client) holds a disjoint data partition locally and exchanges only model parameters after local training. The server aggregates client updates using **Federated Averaging (FedAvg)**.

**Target task.** Multi-class classification on **CIFAR-10** (10 classes, \(32 \times 32\) RGB natural images).

**Framework.** [Flower](https://flower.ai/) (FL) with an **embedded federation** deployment model: a **SuperLink** coordinates the run; **SuperNodes** (e.g., edge devices) host the **ClientApp**; the **ServerApp** runs FedAvg for a fixed number of communication rounds.

**Comparative research question.** The codebase supports two interchangeable classifiers under the **same federated protocol, data splits, and preprocessing**. The intended experiment is a **controlled comparison**:

- **Baseline:** lightweight LeNet-style CNN (`BaselineNet`) — minimal capacity, no attention.
- **Treatment:** attention-augmented CNN (`AttnNet`) — deeper backbone, batch normalization, multi-head self-attention, dropout.

Switch models via run config key **`model-architecture`**: `"baseline"` or `"attention"` (see §7). All other FL settings remain identical so differences in accuracy, convergence, and communication cost can be attributed primarily to architecture.

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

## 3. Model architectures

Both models are defined in `fedavg/task.py`. Input shape is **\(B \times 3 \times 32 \times 32\)**; output is **\(B \times 10\)** logits (no softmax in `forward`; `CrossEntropyLoss` used in training).

Factory function: `create_model(architecture)` with `architecture ∈ {"baseline", "attention"}`.

### 3.1 Architecture comparison (summary)

| Property | `BaselineNet` | `AttnNet` |
|----------|---------------|-----------|
| Role in study | Control (simple CNN) | Treatment (CNN + attention) |
| Conv blocks | 2 (\(5\times5\), no padding) | 3 (\(3\times3\), padding 1) |
| Normalization | None | `BatchNorm2d` after each conv |
| Pooling | After conv1 and conv2 | After conv1 and conv2 only |
| Attention | None | 4-head self-attention on \(8\times8\) grid |
| Regularization | None | Dropout (0.3) before classifier |
| Classifier | 3 FC layers (400→120→84→10) | 2 FC layers (4096→128→10) |
| Local optimizer | SGD, momentum 0.9 | AdamW, weight decay \(10^{-4}\) |
| **Parameters** | **62,006** | **566,282** |

The attention model is ~**9.1×** larger by parameter count. For the paper, report both **parameter count** and **per-round compute** (attention adds \(O(n^2)\) cost over \(n = 64\) spatial tokens per forward pass).

---

### 3.2 Baseline: LeNet-style CNN (`BaselineNet`)

Alias: `Net` (backward compatibility). This is the original federated baseline.

PyTorch `Conv2d` uses **stride 1**, **padding 0** unless stated otherwise.

| Stage | Module | Hyperparameters | Output spatial size | Output channels |
|-------|--------|-----------------|---------------------|-----------------|
| 1 | `Conv2d` | `in=3`, `out=6`, kernel \(5\times5\) | \(28 \times 28\) | 6 |
| 2 | `ReLU` | — | \(28 \times 28\) | 6 |
| 3 | `MaxPool2d` | kernel 2, stride 2 | \(14 \times 14\) | 6 |
| 4 | `Conv2d` | `in=6`, `out=16`, kernel \(5\times5\) | \(10 \times 10\) | 16 |
| 5 | `ReLU` | — | \(10 \times 10\) | 16 |
| 6 | `MaxPool2d` | kernel 2, stride 2 | \(5 \times 5\) | 16 |
| 7 | `Flatten` | — | length **400** | — |
| 8 | `Linear` | \(400 \rightarrow 120\) | — | — |
| 9 | `ReLU` | — | — | — |
| 10 | `Linear` | \(120 \rightarrow 84\) | — | — |
| 11 | `ReLU` | — | — | — |
| 12 | `Linear` | \(84 \rightarrow 10\) | logits | — |

**Parameter count (exact): 62,006**

| Layer | Subtotal |
|-------|----------|
| `conv1` | 456 |
| `conv2` | 2,416 |
| `fc1` | 48,120 |
| `fc2` | 10,164 |
| `fc3` | 850 |

**Inductive bias:** purely local receptive fields; no long-range spatial mixing beyond what two pooling stages allow.

---

### 3.3 Treatment: Attention-augmented CNN (`AttnNet`)

A **convolutional backbone** extracts feature maps; a **single multi-head self-attention (MHSA)** block reweights spatial locations before global classification. This follows a lightweight “CNN → tokens → attention → classifier” pattern suitable for small images on edge hardware (not a full Vision Transformer).

#### 3.3.1 Convolutional backbone

| Stage | Module | Hyperparameters | Output spatial size | Output channels |
|-------|--------|-----------------|---------------------|-----------------|
| 1 | `Conv2d` + `BatchNorm2d` + ReLU | \(3 \rightarrow 16\), kernel \(3\times3\), padding 1 | \(32 \times 32\) | 16 |
| 2 | `MaxPool2d` | kernel 2, stride 2 | \(16 \times 16\) | 16 |
| 3 | `Conv2d` + `BatchNorm2d` + ReLU | \(16 \rightarrow 32\), kernel \(3\times3\), padding 1 | \(16 \times 16\) | 32 |
| 4 | `MaxPool2d` | kernel 2, stride 2 | \(8 \times 8\) | 32 |
| 5 | `Conv2d` + `BatchNorm2d` + ReLU | \(32 \rightarrow 64\), kernel \(3\times3\), padding 1 | \(8 \times 8\) | 64 |

After the backbone: tensor shape **\(B \times 64 \times 8 \times 8\)**.

#### 3.3.2 Attention block

1. **Tokenization:** flatten spatial dimensions → **\(B \times 64 \times 64\)** (64 tokens, each 64-dimensional).
2. **Multi-head self-attention:** `nn.MultiheadAttention(embed_dim=64, num_heads=4, batch_first=True)`.
3. **Residual + layer norm:** `LayerNorm(x + Attention(x))`.

Attention lets each spatial location aggregate context from all \(8 \times 8 = 64\) positions, enabling **global reasoning** within the final feature map—something the baseline achieves only indirectly via fully connected layers on a \(5 \times 5\) grid.

#### 3.3.3 Classification head

| Stage | Module | Hyperparameters |
|-------|--------|-----------------|
| 1 | Reshape + flatten | \(64 \times 8 \times 8 = 4096\) features |
| 2 | `Linear` + ReLU | \(4096 \rightarrow 128\) |
| 3 | `Dropout` | \(p = 0.3\) |
| 4 | `Linear` | \(128 \rightarrow 10\) logits |

**Parameter count (exact): 566,282**

| Component | Subtotal |
|-----------|----------|
| Conv + BatchNorm stack | 23,808 |
| Multi-head attention + `LayerNorm` | 16,640 |
| `fc1`, `fc2`, dropout (no params) | 525,834 |

---

### 3.4 Architectural differences (for paper discussion)

| Mechanism | Baseline | Attention model | Expected effect |
|-----------|----------|-----------------|-----------------|
| Receptive field before classifier | \(5\times5\) final map | \(8\times8\) map + all-to-all attention | Richer spatial context |
| Feature depth | 6 → 16 channels | 16 → 32 → 64 channels | Higher representational capacity |
| Training stability | No normalization | BatchNorm per conv block | Faster, stabler local updates |
| Overfitting control | None | Dropout on classifier | May help generalization under few local epochs |
| Optimization | SGD + momentum | AdamW + weight decay | Better suited to attention + BatchNorm |
| Edge cost | Very low FLOPs / memory | ~9× parameters; attention quadratic in 64 tokens | Higher latency per local epoch on Pis |

**Fair comparison note:** Optimizers differ by design (SGD for classical CNN vs AdamW for attention). For a stricter ablation, a follow-up experiment could train both with the same optimizer; the current setup reflects common practice per architecture family.

---

## 4. Input preprocessing and augmentation

Both models share **identical** dataloaders (`load_data_from_disk`). Images use Hugging Face key `"img"`.

**Normalization** uses standard CIFAR-10 statistics (applied after `ToTensor()`):

- Mean: \((0.4914,\ 0.4822,\ 0.4465)\)
- Std: \((0.2023,\ 0.1994,\ 0.2010)\)

### 4.1 Training (`train` split)

1. `RandomCrop(32, padding=4)`
2. `RandomHorizontalFlip()` (p = 0.5)
3. `ToTensor()`
4. `Normalize(CIFAR10_MEAN, CIFAR10_STD)`

### 4.2 Evaluation (`test` split)

1. `ToTensor()`
2. `Normalize(CIFAR10_MEAN, CIFAR10_STD)`

No random crop or flip at evaluation time.

---

## 5. Local training and evaluation

### 5.1 Loss

- **`CrossEntropyLoss`** on raw logits (both architectures).

### 5.2 Optimizer (architecture-dependent)

| Architecture | Optimizer | Notes |
|--------------|-----------|-------|
| `baseline` | **SGD**, lr from config, **momentum = 0.9** | Matches classic LeNet training |
| `attention` | **AdamW**, lr from config, **weight_decay = 1e-4** | Standard for attention + BatchNorm stacks |

Selected automatically in `create_optimizer()` inside `train()`.

### 5.3 Local training loop

For each federated round, each selected client:

1. Receives global weights (`state_dict`).
2. Instantiates the model via `create_model(model-architecture)`.
3. Loads local train DataLoader from `dataset-path` with `shuffle=True`.
4. Runs `local_epochs` full passes over the local training set.

**Reported train metric:** sum of per-batch CE losses over the last local epoch, divided by number of training batches.

### 5.4 Local evaluation

- Model in `eval()` mode; dropout disabled automatically.
- **Accuracy:** `correct / len(testloader.dataset)` on the client’s local test split.
- **Loss:** mean batch CE loss averaged over batches.

---

## 6. Federated optimization protocol

### 6.1 Server (`fedavg/server_app.py`)

- Reads `model-architecture` from run config.
- Initializes `create_model(architecture)` and wraps `state_dict()` in Flower’s `ArrayRecord`.
- Runs **`FedAvg`** for `num-server-rounds`.

### 6.2 Client (`fedavg/client_app.py`)

- Uses the same `model-architecture` value for train and evaluate handlers.
- Returns updated weights plus `train_loss` / `eval_loss` / `eval_acc` metrics.

### 6.3 Hardware

Clients use **`cuda:0` if available, else CPU**.

---

## 7. Run configuration (defaults)

Declared in `pyproject.toml` → `[tool.flwr.app.config]`:

| Key | Role |
|-----|------|
| `dataset-name` | Artifact label (e.g., `cifar10`). |
| **`model-architecture`** | **`baseline`** or **`attention`** — selects `BaselineNet` vs `AttnNet`. |
| `num-server-rounds` | Federated communication rounds \(T\). |
| `fraction-evaluate` | Client fraction for evaluation each round. |
| `local-epochs` | Local epochs \(E\) per round. |
| `learning-rate` | Base learning rate (both optimizers). |
| `batch-size` | Mini-batch size for train and test loaders. |

**Current defaults:** `model-architecture = attention`, `num-server-rounds = 30`, `local-epochs = 5`, `learning-rate = 0.01`, `batch-size = 32`, `fraction-evaluate = 0.5`.

Per-device **`dataset-path`** is set in Flower node config (not in `pyproject.toml`).

### 7.1 Running the comparison

1. **Baseline run:** set `model-architecture = "baseline"`, run `flwr run . embedded-federation --stream`.
2. **Attention run:** set `model-architecture = "attention"`, repeat with the same `num-server-rounds`, `local-epochs`, `learning-rate`, and `batch-size`.
3. Compare `{dataset}_metrics.json` files under `flwr_runs/` (final `eval_acc`, convergence speed, train/eval loss curves).

Optionally set `dataset-name = "cifar10_baseline"` vs `"cifar10_attention"` to distinguish artifact folders in the paper.

### 7.2 CIFAR-10 comparison results (2026-05-28)

Two federated runs were completed under **matched FedAvg settings**; only `model-architecture` differed. Metrics below are **client-averaged** evaluation values from Flower (`evaluate_metrics_clientapp` in each round).

| Setting | Baseline (`BaselineNet`) | Attention (`AttnNet`) |
|---------|--------------------------|------------------------|
| Run folder | `flwr_runs/cifar10_20260528_130730/` | `flwr_runs/cifar10_20260528_144603/` |
| `model-architecture` | `baseline` | `attention` |
| `num-server-rounds` | 10 | 10 |
| `local-epochs` | 5 | 5 |
| `learning-rate` | 0.001 | 0.001 |
| `batch-size` | 32 | 32 |
| `fraction-evaluate` | 0.5 | 0.5 |
| Local optimizer | SGD (momentum 0.9) | AdamW (weight decay \(10^{-4}\)) |

**Round-by-round evaluation accuracy** (aggregated across sampled clients):

| Round | Baseline `eval_acc` | Attention `eval_acc` |
|-------|---------------------|----------------------|
| 1 | 41.85% | 58.87% |
| 5 | 59.16% | 77.68% |
| 10 | **65.63%** | **82.08%** |

**Final round (10/10):**

| Metric | Baseline | Attention | Δ (attention − baseline) |
|--------|----------|-----------|--------------------------|
| `eval_acc` | 0.6563 | 0.8208 | **+16.45 pp** |
| `eval_loss` | 0.9796 | 0.5262 | **−0.4534** (lower is better) |
| `train_loss` | 5.3047 | 2.9413 | — |

**Summary.** On CIFAR-10 with the configuration above, the attention-augmented CNN **outperformed the baseline CNN** at every logged evaluation round. By round 10, attention reached **82.08%** client-averaged accuracy versus **65.63%** for the baseline—a gain of **16.45 percentage points** with roughly half the evaluation loss. Attention also converged faster (e.g., round 1: 58.87% vs 41.85%).

**Interpretation.** These runs support the project hypothesis that adding a lightweight self-attention block improves federated classification on CIFAR-10 under the same communication schedule. Because optimizers differ (SGD vs AdamW), part of the gap may come from optimization choice as well as capacity/architecture; see §11.

**Reproduce.** Artifacts: `cifar10_run_config.json`, `cifar10_metrics.json`, and `cifar10_final_model.pt` in each run directory listed above.

---

## 8. Artifacts and metrics logging

After each run, `fedavg/run_artifacts.py` writes `flwr_runs/{dataset-name}_{UTC_timestamp}/`:

- **`{dataset}_final_model.pt`** — global `state_dict` (architecture-specific; not interchangeable).
- **`{dataset}_metrics.json`** — per-round client metrics.
- **`{dataset}_run_config.json`** — includes `model-architecture` for reproducibility.

---

## 9. Software stack

From `pyproject.toml`:

- **Python package:** `fedavg`
- **Core deps:** `flwr>=1.28.0`, `flwr-datasets[vision]>=0.5.0`, `torch==2.8.0`, `torchvision==0.23.0`

Record the exact `flwr` version used in experiments (`pip show flwr`).

---

## 10. Suggested paper phrasing (methods snippet)

> We study federated image classification on CIFAR-10 with \(P=5\) IID clients. Each client holds an 80/20 train/test split (seed 42) of its partition. We compare two global models under identical FedAvg settings: (i) a LeNet-style baseline with 62k parameters and two convolutional layers, and (ii) an attention-augmented CNN with 566k parameters, batch normalization, a 4-head self-attention block over an \(8\times8\) feature grid, and dropout. Both models receive the same augmentations (random crop with padding 4, horizontal flip) and CIFAR-10 normalization. The baseline is trained locally with SGD (momentum 0.9); the attention model with AdamW (weight decay \(10^{-4}\)). The server aggregates for \(T\) rounds with \(E\) local epochs per round. We report client-averaged evaluation accuracy and loss per communication round.

---

## 11. Known limitations

- **Optimizer confound:** Baseline uses SGD; attention uses AdamW. Interpret gains as architecture **plus** optimizer choice unless you add a controlled optimizer ablation.
- **No centralized test set** in code; metrics are client-local test splits aggregated by Flower.
- **Attention cost:** Higher memory and compute on Raspberry Pi class devices; report wall-clock per round in the paper.
- **Vanilla FedAvg** only (no DP, robust aggregation, or personalization).

---

## 12. Repository map

| Path | Purpose |
|------|---------|
| `fedavg/task.py` | `BaselineNet`, `AttnNet`, `create_model()`, dataloaders, train/test. |
| `fedavg/client_app.py` | Flower ClientApp handlers. |
| `fedavg/server_app.py` | Flower ServerApp + FedAvg. |
| `fedavg/run_artifacts.py` | Persist model, metrics, config. |
| `generate_dataset.py` | IID CIFAR-10 partitions. |
| `pyproject.toml` | Dependencies, FL hyperparameters, `model-architecture`. |

---

*End of document.*
