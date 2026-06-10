# Federated Learning on CIFAR-10 with Flower FedAvg — Technical Description

This document summarizes the implementation for reproducibility and for drafting a research paper. It reflects the codebase as of the repository state; hyperparameter defaults are taken from `pyproject.toml` under `[tool.flwr.app.config]` unless you override them at run time.

---

## 1. Research setting

**Goal.** Train a single global image classifier without centralizing raw training data: each participant (client) holds a disjoint data partition locally and exchanges only model parameters after local training. The server aggregates client updates using **Federated Averaging (FedAvg)**.

**Target task.** Multi-class classification on **CIFAR-10** (10 classes, \(32 \times 32\) RGB natural images).

**Framework.** [Flower](https://flower.ai/) (FL) with an **embedded federation** deployment model: a **SuperLink** coordinates the run; **SuperNodes** (e.g., edge devices) host the **ClientApp**; the **ServerApp** runs FedAvg for a fixed number of communication rounds.

**Comparative research question.** The codebase supports four interchangeable classifiers under the **same federated protocol, data splits, and preprocessing**. The intended experiment is a **controlled comparison**:

- **Baseline:** lightweight LeNet-style CNN (`BaselineNet`) — minimal capacity, no attention.
- **Attention CNN:** (`AttnNet`) — deeper backbone, batch normalization, multi-head self-attention, dropout.
- **Residual CNN (ablation):** (`ResNet`) — same residual backbone as `ResAttentionNet` but **without** self-attention; isolates the effect of attention within the residual stack.
- **Residual attention CNN:** (`ResAttentionNet`) — two residual blocks, shortcut paths, attention on a \(16\times16\) feature grid.

Switch models via run config key **`model-architecture`**: `"baseline"`, `"attention"`, `"resnet"`, or `"res-attention"` (see §7). All other FL settings remain identical so differences in accuracy, convergence, and communication cost can be attributed primarily to architecture.

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

All models are defined in `fedavg/task.py`. Input shape is **\(B \times 3 \times 32 \times 32\)**; output is **\(B \times 10\)** logits (no softmax in `forward`; `CrossEntropyLoss` used in training).

Factory function: `create_model(architecture)` with `architecture ∈ {"baseline", "attention", "resnet", "res-attention"}`.

### 3.1 Architecture comparison (summary)

| Property | `BaselineNet` | `AttnNet` | `ResNet` | `ResAttentionNet` |
|----------|---------------|-----------|----------|-------------------|
| Role in study | Control (simple CNN) | CNN + attention | Residual CNN (no attention) | Residual CNN + attention |
| Conv blocks | 2 (\(5\times5\), no padding) | 3 (\(3\times3\), padding 1) | 2 residual blocks + init conv | 2 residual blocks + init conv |
| Normalization | None | `BatchNorm2d` after each conv | `BatchNorm2d` after each conv | `BatchNorm2d` after each conv |
| Residual shortcuts | None | None | Identity + \(1\times1\) channel match | Identity + \(1\times1\) channel match |
| Pooling | After conv1 and conv2 | After conv1 and conv2 only | Once (32→16 spatial) | Once (32→16 spatial) |
| Attention | None | 4-head on \(8\times8\) grid (64 tokens) | None | 4-head on \(16\times16\) grid (256 tokens) |
| Regularization | None | Dropout (0.3) | Dropout (0.4) | Dropout (0.4) |
| Classifier | 3 FC (400→120→84→10) | 2 FC (4096→128→10) | 2 FC (16384→256→10) | 2 FC (16384→256→10) |
| Local optimizer | AdamW, weight decay \(10^{-4}\) | AdamW, weight decay \(10^{-4}\) | AdamW, weight decay \(10^{-4}\) | AdamW, weight decay \(10^{-4}\) |
| **Parameters** | **62,006** | **566,282** | **4,274,506** | **4,291,274** |

`ResNet` and `ResAttentionNet` share the same convolutional backbone and classifier head; the attention block adds **16,768** parameters (~0.4% of the residual stack). `ResAttentionNet` is the largest variant (~**69×** baseline parameters). Attention cost scales as \(O(n^2)\) in token count \(n\); residual attention uses \(n = 256\) tokens vs \(n = 64\) for `AttnNet`, so expect higher memory and latency on edge devices when attention is enabled.

### 3.1.1 Trainable layer counts

A **layer** here means a PyTorch `nn.Module` with **learnable parameters** (weights and/or biases). Non-parametric ops such as `ReLU`, `MaxPool2d`, flatten/reshape, and `Dropout` are **not** counted. `nn.MultiheadAttention` is counted as **one** module (it internally holds Q/K/V and output projections).

| Layer type | `BaselineNet` | `AttnNet` | `ResNet` | `ResAttentionNet` |
|------------|:-------------:|:---------:|:--------:|:-----------------:|
| `Conv2d` | 2 | 3 | 6 | 6 |
| `BatchNorm2d` | 0 | 3 | 5 | 5 |
| `MultiheadAttention` | 0 | 1 | 0 | 1 |
| `LayerNorm` | 0 | 1 | 0 | 1 |
| `Linear` (classifier) | 3 | 2 | 2 | 2 |
| **Total trainable modules** | **5** | **10** | **13** | **15** |

**Depth by role:**

| Role | `BaselineNet` | `AttnNet` | `ResNet` | `ResAttentionNet` |
|------|:-------------:|:---------:|:--------:|:-----------------:|
| Convolutional (`Conv2d`) | 2 | 3 | 6 (1 stem + 4 in residual blocks + 1 shortcut) | 6 (same as `ResNet`) |
| Classifier (`Linear`) | 3 | 2 | 2 | 2 (same as `ResNet`) |
| Normalization (`BatchNorm2d` / `LayerNorm`) | 0 | 4 | 5 | 7 |
| Global mixing (`MultiheadAttention`) | 0 | 1 | 0 | 1 |

**Matched pair.** `ResNet` and `ResAttentionNet` share **11 identical** backbone + classifier modules (`6` conv + `5` batch-norm + `2` linear). `ResAttentionNet` adds exactly **2** modules (`MultiheadAttention` + `LayerNorm`) on top—this is the controlled attention ablation.

### 3.1.2 Should layer counts be identical across all four models?

**No—not for this study.** The four-way comparison is intentionally **not** a same-depth, same-width architecture sweep. Models differ in depth, width, normalization, residual paths, and attention so we can measure how **inductive bias and capacity** affect federated CIFAR-10 accuracy under identical FedAvg settings. Holding layer count fixed would confound “more layers” with “different mechanisms” (e.g., you could not add residual shortcuts or attention without changing the module graph).

**Yes—only where ablation demands it.** The **`ResNet` ↔ `ResAttentionNet`** pair is designed so the **convolutional and classifier layers are identical**; only the attention block differs. That isolates whether self-attention improves client-averaged accuracy beyond what the residual CNN already achieves (~84.5% → ~86.4% in our runs; see §7.2).

**What is held constant instead of layer count:** federated protocol (FedAvg, \(P=5\), IID partitions), data preprocessing and augmentation, local optimizer (**AdamW**), communication rounds, local epochs, batch size, and learning rate. Fairness is defined at the **experiment** level, not by matching every `Conv2d` count.

### 3.1.3 What each layer type achieves

| Layer / block | Purpose in these models | Effect on representation |
|---------------|-------------------------|---------------------------|
| **`Conv2d`** | Learn local spatial filters (edges, textures, parts) at increasing channel width | Hierarchical feature maps; deeper stacks (`AttnNet`, residual models) extract richer patterns than the 2-layer baseline |
| **`MaxPool2d`** | Downsample spatial resolution (no learnable weights) | Reduces compute and builds translation tolerance; baseline pools twice to a \(5\times5\) grid, residual models once to \(16\times16\) |
| **`BatchNorm2d`** | Normalize activations per channel during training | Stabilizes optimization across non-IID client updates; absent in `BaselineNet`, present in all treatment models |
| **Residual block** (`conv` + skip) | Add input to transformed features (`x + F(x)`) | Eases training of deeper stacks; `ResNet` / `ResAttentionNet` avoid vanishing gradients that limit the shallow baseline |
| **`MultiheadAttention`** | All-to-all mixing over spatial tokens | Each location attends to every other location on the final grid—**global context** before classification (64 tokens in `AttnNet`, 256 in `ResAttentionNet`) |
| **`LayerNorm`** | Normalize token embeddings before/after attention | Stabilizes the attention residual path (`x + Attention(x)`) |
| **`Linear` (FC)** | Map flattened features to class logits | Final decision boundary; baseline uses a 3-layer MLP on 400 features, others use 2-layer heads on larger maps |
| **`Dropout`** | Randomly zero activations during training (no weights) | Reduces overfitting on large classifier heads (`p=0.3` / `0.4`) |

**Net effect in our results (§7.2):** more convolutional depth + normalization + residuals raised accuracy from **67.98%** (baseline, 2 conv layers) to **84.51%** (`ResNet`, 6 conv layers, no attention). Adding attention on the matched residual backbone contributed a further **+1.85 pp** (`ResAttentionNet`, same 6 conv layers + 1 attention block).

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

### 3.4 Residual CNN without attention (`ResNet`)

Config key: **`resnet`** (alias: `resnetnet`).

An **ablation baseline** for `ResAttentionNet`: identical residual convolutional backbone and classifier head, but **no** multi-head self-attention or `LayerNorm`. This isolates whether accuracy gains of the residual attention stack come from the residual architecture itself or from the attention block.

#### 3.4.1 Backbone

Same structure as §3.5.1 (`ResAttentionNet` backbone): init conv → residual block 1 (identity skip) → pool → residual block 2 (\(1\times1\) shortcut). Output tensor before classification: **\(B \times 64 \times 16 \times 16\)**.

#### 3.4.2 Classification head

No tokenization or attention. The feature map is flattened directly:

1. Flatten → `Linear(16384 → 256)` → ReLU → `Dropout(0.4)` → `Linear(256 → 10)`.

**Parameter count (exact): 4,274,506**

**Ablation note:** Compared with `ResAttentionNet` (4,291,274 parameters), removing attention saves only the MHSA + `LayerNorm` weights (**16,768** parameters), so any accuracy difference at matched FL settings is dominated by attention compute and representational capacity, not model size.

---

### 3.5 Residual attention CNN (`ResAttentionNet`)

Config key: **`res-attention`** (aliases: `res_attention`, `resattention`, `resattentionnet`).

A **residual convolutional backbone** preserves spatial detail via shortcut connections; **multi-head self-attention** runs on a \(16 \times 16\) grid (\(256\) tokens) before classification.

#### 3.5.1 Backbone

| Stage | Module | Output spatial size | Channels |
|-------|--------|---------------------|----------|
| Init | `Conv2d` + `BatchNorm2d` + ReLU | \(32 \times 32\) | 32 |
| Res block 1 | two \(3\times3\) convs + identity skip | \(32 \times 32\) | 32 |
| Pool | `MaxPool2d` | \(16 \times 16\) | 32 |
| Res block 2 | two \(3\times3\) convs + \(1\times1\) shortcut | \(16 \times 16\) | 64 |

#### 3.5.2 Attention and head

1. Tokenize \(B \times 64 \times 16 \times 16\) → **\(B \times 256 \times 64\)**.
2. `MultiheadAttention(embed_dim=64, num_heads=4)` + residual `LayerNorm`.
3. Flatten → `Linear(16384 → 256)` → ReLU → `Dropout(0.4)` → `Linear(256 → 10)`.

**Parameter count (exact): 4,291,274**

**Edge note:** This model is substantially heavier than `AttnNet`; prefer Pi 5 (or fewer clients / smaller `batch-size`) for stable federated runs.

---

### 3.6 Architectural differences (for paper discussion)

| Mechanism | Baseline | Attention model | ResNet (no attn.) | Res-attention | Expected effect |
|-----------|----------|-----------------|-------------------|---------------|-----------------|
| Receptive field before classifier | \(5\times5\) final map | \(8\times8\) map + all-to-all attention | \(16\times16\) map, local conv only | \(16\times16\) map + all-to-all attention | Residual + attention = richest context |
| Feature depth | 6 → 16 channels | 16 → 32 → 64 channels | 32 → 64 channels | 32 → 64 channels | Higher capacity in residual variants |
| Training stability | No normalization | BatchNorm per conv block | BatchNorm + residual skips | BatchNorm + residual skips | Residual paths ease optimization |
| Overfitting control | None | Dropout on classifier | Dropout (0.4) | Dropout (0.4) | Stronger regularization on large heads |
| Optimization | AdamW + weight decay | AdamW + weight decay | AdamW + weight decay | AdamW + weight decay | Same optimizer across all four models |
| Edge cost | Very low FLOPs / memory | ~9× baseline params | ~69× baseline; no attention FLOPs | ~69× baseline + 256-token attention | Attention adds latency on top of residual cost |

**Fair comparison note:** All four architectures use the same local optimizer (**AdamW**, lr from config, weight decay \(10^{-4}\)) so accuracy differences reflect model design, not optimization choice. The **`ResNet` vs `ResAttentionNet`** pair is the cleanest ablation for isolating attention within a matched residual backbone.

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

### 5.2 Optimizer (shared across architectures)

| Architecture | Optimizer | Notes |
|--------------|-----------|-------|
| `baseline` | **AdamW**, lr from config, **weight_decay = 1e-4** | Same as attention/residual variants |
| `attention` | **AdamW**, lr from config, **weight_decay = 1e-4** | — |
| `resnet` | **AdamW**, lr from config, **weight_decay = 1e-4** | Ablation pair with `res-attention` |
| `res-attention` | **AdamW**, lr from config, **weight_decay = 1e-4** | — |

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
- **Accuracy (`eval_acc`):** `correct / len(testloader.dataset)` on the client’s local test split.
- **Macro-F1 (`eval_f1`):** unweighted mean of per-class F1 scores on the client’s local test split (see §5.5).
- **Loss (`eval_loss`):** mean batch CE loss averaged over batches.

All three metrics are returned by `test()` in `fedavg/task.py` and reported by the Flower `ClientApp` evaluate handler.

### 5.5 Why macro-F1 matters (not accuracy alone)

**Accuracy** counts only whether the predicted class equals the label. It can hide **class-wise failure modes**: a model that confuses visually similar classes (e.g., cat vs dog, automobile vs truck) may still show high accuracy on balanced CIFAR-10 while performing poorly on specific categories.

**Macro-F1** (the metric we log as `eval_f1`) averages F1 across all 10 classes with **equal weight per class**, regardless of how many examples of each class appear in a client’s local test split:

\[
\text{F1}_c = \frac{2 \cdot \text{Precision}_c \cdot \text{Recall}_c}{\text{Precision}_c + \text{Recall}_c}, \qquad
\text{macro-F1} = \frac{1}{10}\sum_{c=0}^{9} \text{F1}_c
\]

Per-class precision and recall are computed from the confusion matrix on the client’s full local test set. If a class has no predicted or true support, that class’s F1 is set to **0** (standard `zero_division=0` behavior).

**Why report both.** Accuracy is intuitive and matches prior FL logs in §7.2. Macro-F1 is standard in multi-class classification papers because it penalizes models that sacrifice minority or hard classes. On IID CIFAR-10 shards macro-F1 is often close to accuracy, but the gap (e.g., attention: 85.62% acc vs 85.50% macro-F1 in §7.3) reveals slight per-class imbalance in precision/recall that accuracy alone does not surface.

**Federated aggregation.** Flower aggregates `eval_f1` across evaluated clients each round the same way as `eval_acc` (example-weighted mean over clients that participated in evaluation).

---

## 6. Federated optimization protocol

### 6.1 Server (`fedavg/server_app.py`)

- Reads `model-architecture` from run config.
- Initializes `create_model(architecture)` and wraps `state_dict()` in Flower’s `ArrayRecord`.
- Runs **`FedAvg`** for `num-server-rounds`.

### 6.2 Client (`fedavg/client_app.py`)

- Uses the same `model-architecture` value for train and evaluate handlers.
- Returns updated weights plus `train_loss` / `eval_loss` / `eval_acc` / `eval_f1` metrics.

### 6.3 Hardware

Clients use **`cuda:0` if available, else CPU**.

---

## 7. Run configuration (defaults)

Declared in `pyproject.toml` → `[tool.flwr.app.config]`:

| Key | Role |
|-----|------|
| `dataset-name` | Artifact label (e.g., `cifar10`). |
| **`model-architecture`** | **`baseline`**, **`attention`**, **`resnet`**, or **`res-attention`** — selects `BaselineNet`, `AttnNet`, `ResNet`, or `ResAttentionNet`. |
| `num-server-rounds` | Federated communication rounds \(T\). |
| `fraction-evaluate` | Client fraction for evaluation each round. |
| `local-epochs` | Local epochs \(E\) per round. |
| `learning-rate` | Base learning rate (both optimizers). |
| `batch-size` | Mini-batch size for train and test loaders. |

**Current defaults:** `model-architecture = resnet`, `num-server-rounds = 10`, `local-epochs = 5`, `learning-rate = 0.001`, `batch-size = 32`, `fraction-evaluate = 0.5`.

Per-device **`dataset-path`** is set in Flower node config (not in `pyproject.toml`).

### 7.1 Running the comparison

1. **Baseline run:** set `model-architecture = "baseline"`, run `flwr run . embedded-federation --stream`.
2. **Attention run:** set `model-architecture = "attention"`, repeat with the same `num-server-rounds`, `local-epochs`, `learning-rate`, and `batch-size`.
3. **Residual ablation run:** set `model-architecture = "resnet"`, repeat with the same hyperparameters.
4. **Residual attention run:** set `model-architecture = "res-attention"`, repeat with the same hyperparameters.
5. Compare `{dataset}_metrics.json` files under `flwr_runs/` (final `eval_acc`, **`eval_f1`**, convergence speed, train/eval loss curves). Pay special attention to **`resnet` vs `res-attention`** for the attention ablation.

Optionally set distinct `dataset-name` values (e.g. `cifar10_baseline`, `cifar10_attention`, `cifar10_resnet`, `cifar10_res_attention`) to separate artifact folders.

### 7.2 CIFAR-10 comparison results (2026-05-28 / 2026-05-30)

Four federated runs were completed under **matched FedAvg settings**; only `model-architecture` differed. All four use **AdamW** (weight decay \(10^{-4}\)). Metrics below are **client-averaged** evaluation values from Flower (`evaluate_metrics_clientapp` in each round).

| Setting | Baseline (`BaselineNet`) | Attention (`AttnNet`) | Residual (`ResNet`) | Residual attention (`ResAttentionNet`) |
|---------|--------------------------|------------------------|---------------------|----------------------------------------|
| Run folder | `flwr_runs/cifar10_20260529_114124/` | `flwr_runs/cifar10_20260528_144603/` | `flwr_runs/cifar10_20260530_210916/` | `flwr_runs/cifar10_20260528_232524/` |
| `model-architecture` | `baseline` | `attention` | `resnet` | `res-attention` |
| `num-server-rounds` | 10 | 10 | 10 | 10 |
| `local-epochs` | 5 | 5 | 5 | 5 |
| `learning-rate` | 0.001 | 0.001 | 0.001 | 0.001 |
| `batch-size` | 32 | 32 | 32 | 32 |
| `fraction-evaluate` | 0.5 | 0.5 | 0.5 | 0.5 |
| Parameters (approx.) | 62k | 566k | 4.27M | 4.29M |
| Local optimizer | AdamW (weight decay \(10^{-4}\)) | AdamW (weight decay \(10^{-4}\)) | AdamW (weight decay \(10^{-4}\)) | AdamW (weight decay \(10^{-4}\)) |

**Round-by-round evaluation accuracy** (aggregated across sampled clients):

| Round | Baseline `eval_acc` | Attention `eval_acc` | ResNet `eval_acc` | Res-attention `eval_acc` |
|-------|---------------------|----------------------|-------------------|--------------------------|
| 1 | 35.17% | 58.87% | 43.54% | 39.06% |
| 5 | 64.74% | 77.68% | 80.90% | **83.91%** |
| 10 | 67.98% | 82.08% | 84.51% | **86.36%** |

**Final round (10/10) — federated logs** (`eval_acc` / `eval_loss` only; `eval_f1` was added after these runs; see §7.3):

| Metric | Baseline | Attention | ResNet | Res-attention | Δ (res-attn − resnet) | Δ (res-attn − baseline) |
|--------|----------|-----------|--------|---------------|-------------------------|-------------------------|
| `eval_acc` | 0.6798 | 0.8208 | 0.8451 | **0.8636** | **+1.85 pp** | **+18.38 pp** |
| `eval_loss` | 0.9137 | 0.5262 | 0.4682 | **0.4176** | **−0.0506** | **−0.4961** |
| `train_loss` | 4.9107 | 2.9413 | 2.5728 | 2.1835 | — | — |

### 7.3 Macro-F1 scores (post-hoc, final global models)

The four CIFAR-10 runs above predate `eval_f1` logging. To report macro-F1 for the paper, each saved **`cifar10_final_model.pt`** was re-evaluated on **all five** client test splits (`datasets/5-nodes-partition/cifar10_part_1` … `part_5`, 2,000 test images per client). Metrics below are **example-weighted** over clients (same weighting Flower uses when all clients evaluate).

| Model | Post-hoc `eval_acc` | Post-hoc **macro-F1** (`eval_f1`) | Acc − F1 gap |
|-------|--------------------:|----------------------------------:|-------------:|
| `BaselineNet` | 70.46% | **70.49%** | −0.03 pp |
| `AttnNet` | 85.62% | **85.50%** | +0.12 pp |
| `ResNet` | 88.17% | **88.15%** | +0.02 pp |
| `ResAttentionNet` | **90.25%** | **90.27%** | −0.02 pp |

**F1 ranking matches accuracy ranking:** `ResAttentionNet` > `ResNet` > `AttnNet` > `BaselineNet`.

**Attention ablation (macro-F1).** `ResAttentionNet` improves macro-F1 over `ResNet` by **+2.12 pp** (88.15% → 90.27%) and over `BaselineNet` by **+19.78 pp** — slightly larger F1 gaps than the corresponding accuracy gaps (+1.85 pp and +18.38 pp in federated round-10 logs), indicating attention helps hardest classes disproportionately.

**Per-client macro-F1 at final round** (illustrates cross-client stability):

| Client partition | Baseline F1 | Attention F1 | ResNet F1 | Res-attention F1 |
|------------------|-------------|--------------|-----------|------------------|
| `cifar10_part_1` | 68.82% | 84.42% | 86.75% | 89.09% |
| `cifar10_part_2` | 70.39% | 85.39% | 88.45% | 90.49% |
| `cifar10_part_3` | 71.64% | 85.23% | 88.07% | **91.41%** |
| `cifar10_part_4` | 70.90% | 86.72% | 89.33% | 90.04% |
| `cifar10_part_5` | 70.70% | 85.74% | 88.15% | 90.34% |

**Recompute post-hoc F1.** From the repo root:

```bash
python evaluate_final_models.py
```

This loads each `cifar10_final_model.pt`, calls `test()` on all five client partitions, and writes:

- `flwr_runs/<run_id>/cifar10_posthoc_eval.json` — per-client and aggregate `eval_acc` / `eval_f1` / `eval_loss`
- `flwr_runs/posthoc_eval_summary.json` — combined summary for all four documented runs

Options: `--run-dir flwr_runs/<run_id>` (repeatable), `--partitions-root datasets/5-nodes-partition`, `--batch-size 32`.

New federated runs log `eval_f1` every round automatically in `{dataset}_metrics.json`.

**Summary.** Under identical federated hyperparameters and **AdamW** local optimization, **`ResAttentionNet` achieved the highest client-averaged test accuracy** (**86.36%** at round 10 in federated logs; **90.27% macro-F1** post-hoc on all clients), followed by **`ResNet` (84.51% acc / 88.15% F1)**, **`AttnNet` (82.08% acc / 85.50% F1)**, and **`BaselineNet` (67.98% acc / 70.49% F1)**. The residual stack alone (`ResNet`) outperformed the non-residual attention model by **+2.43 pp** accuracy at round 10 despite similar nominal capacity in the classifier head—suggesting residual shortcuts and the \(16\times16\) feature grid contribute substantially before attention is applied.

**Attention ablation (`ResNet` vs `ResAttentionNet`).** With matched backbone and head (parameter difference **< 0.4%**), adding 4-head self-attention over 256 tokens improved final federated accuracy by **+1.85 percentage points** (84.51% → 86.36%) and post-hoc macro-F1 by **+2.12 pp** (88.15% → 90.27%). Attention therefore provides a measurable gain on top of the residual CNN under FedAvg—on both accuracy and per-class F1—at the cost of quadratic attention FLOPs on edge hardware.

**Convergence note.** `ResAttentionNet` started below `ResNet` at round 1 (39.06% vs 43.54%) but surpassed both non-residual models by round 5 and maintained the lead through round 10. `ResNet` tracked closely behind `ResAttentionNet` from round 5 onward (80.90% vs 83.91% at round 5), consistent with a shared residual backbone converging similarly until the attention block differentiates peak performance.

**Interpretation.** All four runs support the hypothesis that deeper, normalized backbones improve federated CIFAR-10 classification relative to the LeNet-style baseline. The **`ResNet` ablation** separates residual capacity from attention: most of the gain over `AttnNet` comes from the residual architecture, while attention adds a further **~1.9 pp** at matched parameter count.

**Reproduce.** Artifacts in each run directory: `cifar10_run_config.json`, `cifar10_metrics.json`, and `cifar10_final_model.pt` for `cifar10_20260529_114124`, `cifar10_20260528_144603`, `cifar10_20260530_210916`, and `cifar10_20260528_232524`.

---

## 8. Artifacts and metrics logging

After each run, `fedavg/run_artifacts.py` writes `flwr_runs/{dataset-name}_{UTC_timestamp}/`:

- **`{dataset}_final_model.pt`** — global `state_dict` (architecture-specific; not interchangeable).
- **`{dataset}_metrics.json`** — per-round client metrics (`eval_acc`, **`eval_f1`**, `eval_loss`, `train_loss`).
- **`{dataset}_run_config.json`** — includes `model-architecture` for reproducibility.

---

## 9. Software stack

From `pyproject.toml`:

- **Python package:** `fedavg`
- **Core deps:** `flwr>=1.28.0`, `flwr-datasets[vision]>=0.5.0`, `torch==2.8.0`, `torchvision==0.23.0`

Record the exact `flwr` version used in experiments (`pip show flwr`).

---

## 10. Suggested paper phrasing (methods snippet)

> We study federated image classification on CIFAR-10 with \(P=5\) IID clients. Each client holds an 80/20 train/test split (seed 42) of its partition. We compare four global models under identical FedAvg settings: (i) a LeNet-style baseline (62k parameters), (ii) an attention-augmented CNN (566k parameters; 4-head self-attention over an \(8\times8\) grid), (iii) a residual CNN ablation without attention (4.27M parameters; same backbone as (iv)), and (iv) a residual attention CNN (4.29M parameters; residual blocks plus 4-head self-attention over a \(16\times16\) grid). All models share the same augmentations (random crop with padding 4, horizontal flip), CIFAR-10 normalization, and local **AdamW** optimizer (weight decay \(10^{-4}\)). The server aggregates for \(T=10\) rounds with \(E=5\) local epochs per round. We report client-averaged **accuracy**, **macro-F1**, and loss per communication round. Federated round-10 accuracies were **67.98%** (baseline), **82.08%** (attention), **84.51%** (residual, no attention), and **86.36%** (residual attention). Post-hoc macro-F1 on all five client test splits (final global models) was **70.49%**, **85.50%**, **88.15%**, and **90.27%**, respectively. Adding attention to the matched residual backbone improved macro-F1 by **2.12 pp** (88.15% → 90.27%) and federated accuracy by **1.85 pp** (84.51% → 86.36%).

---

## 11. Known limitations

- **No centralized test set** in code; metrics are client-local test splits aggregated by Flower.
- **Attention cost:** `ResAttentionNet` and `ResNet` are the largest models (~69× baseline parameters); only `ResAttentionNet` pays the cost of 256-token self-attention. Report wall-clock and memory per round on Raspberry Pi class devices in the paper, especially for the `resnet` vs `res-attention` ablation pair.
- **Vanilla FedAvg** only (no DP, robust aggregation, or personalization).

---

## 12. Repository map

| Path | Purpose |
|------|---------|
| `fedavg/task.py` | `BaselineNet`, `AttnNet`, `ResNet`, `ResAttentionNet`, `create_model()`, dataloaders, train/test. |
| `fedavg/client_app.py` | Flower ClientApp handlers. |
| `fedavg/server_app.py` | Flower ServerApp + FedAvg. |
| `fedavg/run_artifacts.py` | Persist model, metrics, config. |
| `generate_dataset.py` | IID CIFAR-10 partitions. |
| `pyproject.toml` | Dependencies, FL hyperparameters, `model-architecture`. |

---

*End of document.*
