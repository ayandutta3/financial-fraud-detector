# Financial Fraud Detector — Fine-Tuning on AMD ROCm

A production-grade supervised fine-tuning (SFT) pipeline that trains a Small Language Model (SLM) to perform **multi-class financial transaction fraud classification**, optimised specifically for **AMD GPU hardware running ROCm**.

The output is a fully merged, standalone model that can be served immediately with **vLLM** — no adapter flags required.

---

## Table of Contents

1. [Project Structure](#project-structure)
2. [System Requirements](#system-requirements)
3. [Installation](#installation)
4. [Dataset Preparation](#dataset-preparation)
5. [Pipeline Overview](#pipeline-overview)
6. [Step 1 — Preprocess Data](#step-1--preprocess-data)
7. [Step 2 — Train the Model](#step-2--train-the-model)
8. [Resuming an Interrupted Training Run](#resuming-an-interrupted-training-run)
9. [Persistent Volume Layout (VM Deployments)](#persistent-volume-layout-vm-deployments)
10. [Serving the Model with vLLM](#serving-the-model-with-vllm)
11. [Script Architecture](#script-architecture)
12. [Fraud Label Reference](#fraud-label-reference)
13. [Troubleshooting AMD ROCm Issues](#troubleshooting-amd-rocm-issues)

---

## Project Structure

```
financial-fraud-detector/
│
├── sample-dataset/
│   ├── fraud-detection-dataset-1.csv        # Balanced dataset 1 (2,980 rows, 50-50)
│   ├── fraud-detection-dataset-1.csv.bak    # Original unbalanced backup
│   ├── fraud-detection-dataset-2.csv        # Balanced dataset 2 (16,426 rows, 50-50)
│   ├── fraud-detection-dataset-2.csv.bak    # Original unbalanced backup
│   └── phases/                              # Pre-split phase CSVs for phased training
│       ├── phase-1.csv                      # Phase 1 training data (1,192 rows, ~40%)
│       ├── phase-2.csv                      # Phase 2 training data (894 rows, ~30%)
│       └── phase-3.csv                      # Phase 3 training data (894 rows, ~30%)
│
├── preprocess_data.py        # Step 1: CSV → HuggingFace Dataset saved to disk
├── train_fraud_detector.py   # Step 2: Load dataset → LoRA fine-tune → merge & save
├── balance_dataset.py        # Utility: balance CSV to 50-50 fraud/non-fraud ratio
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

---

## System Requirements

| Component        | Minimum                              | Recommended                          |
|------------------|--------------------------------------|--------------------------------------|
| **GPU**          | AMD Radeon RX 7900 XTX (RDNA3, 24 GB) | AMD Instinct MI250X / MI300X         |
| **VRAM**         | 16 GB                                | 32 GB+                               |
| **ROCm Version** | 6.1                                  | 6.2                                  |
| **OS**           | Ubuntu 22.04 LTS                     | Ubuntu 22.04 / 24.04 LTS             |
| **Python**       | 3.10                                 | 3.11                                 |
| **RAM**          | 32 GB                                | 64 GB                                |
| **Disk**         | 40 GB free                           | 100 GB free (models + checkpoints)   |

> **Note:** The script uses `bfloat16` precision which is natively supported and highly stable on AMD Instinct (MI200/MI300 series) and RDNA3+ architectures. Older RDNA2 GPUs may need to fall back to `float16`.

---

## Installation

Follow these steps **in order**. Installing PyTorch from the wrong index will pull a CUDA build that will not work with ROCm.

### Step 1 — Verify ROCm is installed

```bash
rocm-smi
# Expected: table showing your GPU with ROCm version
```

```bash
rocminfo | grep "gfx"
# Expected: gfxXXX architecture string (e.g. gfx1100 for RDNA3, gfx90a for MI250X)
```

### Step 2 — Create a Python virtual environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip wheel setuptools
```

### Step 3 — Install PyTorch with ROCm support (MUST be done first)

```bash
# For ROCm 6.2 (recommended)
pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/rocm6.2

# For ROCm 6.1 (older drivers)
pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/rocm6.1
```

Verify the installation:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
# Expected: True + your AMD GPU name
```

### Step 4 — Install remaining dependencies

```bash
pip install -r requirements.txt
```

### Step 5 — Install bitsandbytes (optional, only for `--use_4bit` mode)

```bash
# ROCm-compatible bitsandbytes build
pip install bitsandbytes --prefer-binary
```

### Step 6 — Authenticate with HuggingFace Hub

Some base models (e.g. Llama-3.2-1B) require accepting a licence on the HuggingFace website and authenticating locally:

```bash
pip install huggingface_hub
huggingface-cli login
# Paste your HuggingFace access token when prompted
```

---

## Dataset Preparation

The datasets in `sample-dataset/` are already **balanced to a 50-50 fraud / non-fraud ratio** using `balance_dataset.py`.

### Dataset 1 Column Schema

| Column               | Type    | Description                                     |
|----------------------|---------|-------------------------------------------------|
| `transaction_id`     | string  | Unique transaction identifier                   |
| `timestamp`          | string  | ISO-8601 transaction timestamp                  |
| `customer_id`        | string  | Customer account ID                             |
| `card_id`            | string  | Card used for the transaction                   |
| `device_id`          | string  | Device fingerprint                              |
| `ip_address`         | string  | Originating IP address                          |
| `merchant_id`        | string  | Merchant identifier                             |
| `merchant_category`  | string  | MCC category (e.g. grocery, electronics)        |
| `merchant_country`   | string  | ISO country code                                |
| `merchant_city`      | string  | City name                                       |
| `merchant_latitude`  | float   | Merchant geolocation                            |
| `merchant_longitude` | float   | Merchant geolocation                            |
| `transaction_type`   | string  | `purchase` or `transfer`                        |
| `amount`             | float   | Transaction amount in USD                       |
| `currency`           | string  | Currency code                                   |
| `is_fraud`           | int     | Binary fraud flag (0 = clean, 1 = fraud)        |
| `fraud_type`         | string  | Multi-class fraud label (empty if clean)        |

### Re-balancing a dataset (if needed)

```bash
# Balance dataset 1
python balance_dataset.py 1

# Balance dataset 2
python balance_dataset.py 2
```

---

## Pipeline Overview

The training pipeline is split into two independent steps:

```
┌─────────────────────────────┐       ┌──────────────────────────────────────┐
│   preprocess_data.py        │       │   train_fraud_detector.py            │
│                             │       │                                      │
│  CSV  ──► chat-template     │       │  processed dataset  ──► LoRA SFT     │
│           formatting        ├──────►│  base model         ──► merge        │
│           save_to_disk()    │       │                     ──► save model   │
└─────────────────────────────┘       └──────────────────────────────────────┘
        Run ONCE                              Resume-capable, checkpoint-aware
```

Preprocessing and training are fully decoupled:
- **Preprocess once** — the formatted dataset is saved to disk and reused across all training runs.
- **Train separately** — the training script loads the dataset from disk, writes checkpoints to a dedicated directory, and saves the final merged model to a separate output directory.

---

## Step 1 — Preprocess Data

Run `preprocess_data.py` **once** before any training. It reads the raw CSV, applies the chat template using the model's tokenizer, and saves a HuggingFace `Dataset` to disk.

```bash
python preprocess_data.py \
    --csv_path      sample-dataset/fraud-detection-dataset-1.csv \
    --base_model    meta-llama/Llama-3.2-1B \
    --output_dir    /mnt/data/processed-data
```

### Preprocess Argument Reference

| Argument        | Default                                         | Description                                                          |
|-----------------|-------------------------------------------------|----------------------------------------------------------------------|
| `--csv_path`    | `sample-dataset/fraud-detection-dataset-1.csv` | Path to the balanced CSV dataset                                     |
| `--base_model`  | `meta-llama/Llama-3.2-1B`                      | HuggingFace model ID — used only to load the tokenizer for templating |
| `--output_dir`  | `./processed-data`                              | Directory where the HuggingFace Dataset will be saved (`save_to_disk`) |
| `--max_seq_len` | `512`                                           | Maximum token sequence length (informational, stored as metadata)    |

---

## Step 2 — Train the Model

Once preprocessing is complete, run `train_fraud_detector.py`. This script:
1. Loads the preprocessed dataset from `--processed_data_dir`
2. Loads the base model and attaches LoRA adapters
3. Runs SFT, saving checkpoints to `--checkpoint_dir` every `--save_steps` steps
4. Merges LoRA weights into the base model and saves the final model to `--model_output_dir`

### Basic Training Run

```bash
python train_fraud_detector.py \
    --base_model         meta-llama/Llama-3.2-1B \
    --processed_data_dir /mnt/data/processed-data \
    --checkpoint_dir     /mnt/data/checkpoints \
    --model_output_dir   /mnt/data/fraud-detector-merged \
    --epochs             3 \
    --batch_size         4 \
    --lr                 2e-4
```

### Using Phi-3.5-mini-instruct as the base model

```bash
python train_fraud_detector.py \
    --base_model         microsoft/Phi-3.5-mini-instruct \
    --processed_data_dir /mnt/data/processed-data \
    --checkpoint_dir     /mnt/data/checkpoints \
    --model_output_dir   /mnt/data/fraud-phi-merged \
    --epochs             3 \
    --batch_size         4
```

### Low-VRAM mode (4-bit NF4 quantisation, ≥ 12 GB VRAM)

```bash
python train_fraud_detector.py \
    --base_model         meta-llama/Llama-3.2-1B \
    --processed_data_dir /mnt/data/processed-data \
    --checkpoint_dir     /mnt/data/checkpoints \
    --model_output_dir   /mnt/data/fraud-detector-merged \
    --epochs             3 \
    --batch_size         2 \
    --use_4bit
```

### Full Training Argument Reference

| Argument                    | Default                   | Description                                                                 |
|-----------------------------|---------------------------|-----------------------------------------------------------------------------|
| `--base_model`              | `meta-llama/Llama-3.2-1B` | HuggingFace model ID or local path of the base model                       |
| `--processed_data_dir`      | `./processed-data`        | Directory with the HuggingFace Dataset produced by `preprocess_data.py`    |
| `--checkpoint_dir`          | `./checkpoints`           | Directory where training checkpoints are saved                              |
| `--model_output_dir`        | `./fraud-detector-merged` | Directory where the final merged, vLLM-ready model is saved                |
| `--resume_from_checkpoint`  | `None`                    | Path to a checkpoint dir, or `auto` to resume from the latest checkpoint   |
| `--epochs`                  | `3`                       | Number of training epochs                                                   |
| `--batch_size`              | `4`                       | Per-device training batch size                                              |
| `--lr`                      | `2e-4`                    | Peak learning rate                                                          |
| `--max_seq_len`             | `512`                     | Maximum token sequence length                                               |
| `--save_steps`              | `50`                      | Save a checkpoint every N optimizer steps                                   |
| `--use_4bit`                | `False`                   | Enable 4-bit NF4 quantisation for low-VRAM GPUs                            |

---

## Resuming an Interrupted Training Run

Checkpoints are saved automatically every `--save_steps` steps (default: every 50 steps) and at epoch boundaries. Up to 5 checkpoints are retained (oldest are removed automatically).

### Resume from a specific checkpoint

```bash
python train_fraud_detector.py \
    --base_model         meta-llama/Llama-3.2-1B \
    --processed_data_dir /mnt/data/processed-data \
    --checkpoint_dir     /mnt/data/checkpoints \
    --model_output_dir   /mnt/data/fraud-detector-merged \
    --epochs             3 \
    --resume_from_checkpoint /mnt/data/checkpoints/checkpoint-150
```

### Auto-resume from the latest checkpoint

Pass `auto` to automatically detect and resume from the most recent checkpoint in `--checkpoint_dir`:

```bash
python train_fraud_detector.py \
    --base_model         meta-llama/Llama-3.2-1B \
    --processed_data_dir /mnt/data/processed-data \
    --checkpoint_dir     /mnt/data/checkpoints \
    --model_output_dir   /mnt/data/fraud-detector-merged \
    --epochs             3 \
    --resume_from_checkpoint auto
```

> **Tip:** Use `auto` in job scheduler scripts (cron, systemd, etc.) so the run safely restarts after any VM preemption or reboot without manual intervention.

---

## Persistent Volume Layout (VM Deployments)

When running on a VM with a single persistent data volume (e.g. mounted at `/mnt/data`), point **all** output paths to that volume so nothing is lost if the VM is reprovisioned or the ephemeral OS disk is wiped.

### Recommended directory layout on `/mnt/data`

```
/mnt/data/
│
├── processed-data/          ← --output_dir (preprocess_data.py)
│   ├── dataset_info.json    ← --processed_data_dir (train_fraud_detector.py)
│   ├── data-00000-of-00001.arrow
│   └── state.json
│
├── checkpoints/             ← --checkpoint_dir
│   ├── checkpoint-50/
│   ├── checkpoint-100/
│   └── checkpoint-150/      ← latest checkpoint (auto-resume picks this)
│
└── fraud-detector-merged/   ← --model_output_dir
    ├── config.json
    ├── tokenizer.json
    ├── model-00001-of-00002.safetensors
    └── model-00002-of-00002.safetensors
```

### Full end-to-end example for a VM with `/mnt/data`

The phase CSVs (`phase-1.csv`, `phase-2.csv`, `phase-3.csv`) are pre-split and live in `sample-dataset/phases/`. Copy them to the persistent volume once, then run each phase in sequence.

```bash
# ── One-time: copy phase data to persistent volume ────────────────────────
cp sample-dataset/phases/phase-1.csv /mnt/data/phases/
cp sample-dataset/phases/phase-2.csv /mnt/data/phases/
cp sample-dataset/phases/phase-3.csv /mnt/data/phases/

# ── Phase 1 Training ──────────────────────────────────────────────────────
# Preprocess phase-1 data (run once per phase)
python preprocess_data.py \
    --csv_path      /mnt/data/phases/phase-1.csv \
    --base_model    meta-llama/Llama-3.2-1B \
    --output_dir    /mnt/data/processed-phase-1

# Train on phase-1
python train_fraud_detector.py \
    --base_model         meta-llama/Llama-3.2-1B \
    --processed_data_dir /mnt/data/processed-phase-1 \
    --checkpoint_dir     /mnt/data/checkpoints-phase-1 \
    --model_output_dir   /mnt/data/model-phase-1 \
    --epochs             3 --batch_size 4 --lr 2e-4 --save_steps 50

# ── Phase 2 Training (fine-tune the phase-1 model further) ───────────────
python preprocess_data.py \
    --csv_path      /mnt/data/phases/phase-2.csv \
    --base_model    /mnt/data/model-phase-1 \
    --output_dir    /mnt/data/processed-phase-2

python train_fraud_detector.py \
    --base_model         /mnt/data/model-phase-1 \
    --processed_data_dir /mnt/data/processed-phase-2 \
    --checkpoint_dir     /mnt/data/checkpoints-phase-2 \
    --model_output_dir   /mnt/data/model-phase-2 \
    --epochs             2 --batch_size 4 --lr 1e-4 --save_steps 50

# ── Phase 3 Training (final refinement) ──────────────────────────────────
python preprocess_data.py \
    --csv_path      /mnt/data/phases/phase-3.csv \
    --base_model    /mnt/data/model-phase-2 \
    --output_dir    /mnt/data/processed-phase-3

python train_fraud_detector.py \
    --base_model         /mnt/data/model-phase-2 \
    --processed_data_dir /mnt/data/processed-phase-3 \
    --checkpoint_dir     /mnt/data/checkpoints-phase-3 \
    --model_output_dir   /mnt/data/fraud-detector-final \
    --epochs             2 --batch_size 4 --lr 5e-5 --save_steps 50

# ── Resume any phase if the VM was rebooted mid-training ─────────────────
python train_fraud_detector.py \
    --base_model         meta-llama/Llama-3.2-1B \
    --processed_data_dir /mnt/data/processed-phase-1 \
    --checkpoint_dir     /mnt/data/checkpoints-phase-1 \
    --model_output_dir   /mnt/data/model-phase-1 \
    --epochs             3 \
    --resume_from_checkpoint auto
```

> **Learning rate schedule across phases:** It's good practice to use a slightly lower `--lr` for each subsequent phase (e.g. 2e-4 → 1e-4 → 5e-5) to avoid catastrophic forgetting as the model progressively specialises.

> **Important:** Ensure the persistent volume is mounted **before** the training process starts. Verify with `df -h /mnt/data` or `mountpoint /mnt/data`.

---

## Serving the Model with vLLM

After training completes, the merged model is stored in `--model_output_dir`. Serve it directly:

### Install vLLM (ROCm build)

```bash
pip install vllm
# OR for ROCm-specific optimised build:
pip install vllm --extra-index-url https://download.pytorch.org/whl/rocm6.2
```

### Start the inference server

```bash
vllm serve /mnt/data/fraud-detector-merged \
    --dtype bfloat16 \
    --trust-remote-code \
    --host 0.0.0.0 \
    --port 8000
```

### Send a test inference request

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/mnt/data/fraud-detector-merged",
    "messages": [
      {
        "role": "system",
        "content": "You are a Senior Financial Forensic Analyst..."
      },
      {
        "role": "user",
        "content": "Analyse the following payment transaction and classify it:\n\n  Transaction ID   : TXN000000001\n  Timestamp        : 2025-06-08T19:00:46+00:00\n  Customer ID      : CUST0078936\n  Merchant Category: electronics\n  Transaction Type : purchase\n  Amount (USD)     : 1.00\n  Merchant Country : RU\n  Card ID          : CARD0000001-9\n\nWhat is the fraud classification for this transaction?"
      }
    ],
    "max_tokens": 10,
    "temperature": 0.0
  }'
```

Expected response body (assistant content): `CARD_TESTING`

---

## Script Architecture

### `preprocess_data.py`

```
preprocess_data.py
│
├── FRAUD_LABEL_MAP               # Raw CSV → canonical label normalisation
├── SYSTEM_PROMPT                 # Forensic analyst role + label definitions
├── normalize_fraud_label()       # Per-row label resolver
├── build_chat_narrative()        # Row → [system, user, assistant] messages
├── preprocess()                  # CSV → tokenize → save_to_disk()
└── main() / parse_args()         # CLI entry-point
```

### `train_fraud_detector.py`

```
train_fraud_detector.py
│
├── validate_rocm_device()        # AMD/ROCm detection, HIP env-var tuning
├── load_model_and_tokenizer()    # Base model + LoRA adapter (bfloat16)
├── train()                       # SFTTrainer loop with checkpoint resume support
├── merge_and_save()              # merge_and_unload() → safetensors on disk
└── main() / parse_args()         # CLI entry-point + pipeline orchestration
```

### Key Design Decisions

| Decision | Rationale |
|---|---|
| Two-script pipeline | Preprocessing is decoupled from training — run the tokenizer overhead once and reuse the dataset across many training experiments |
| Separate `--checkpoint_dir` / `--model_output_dir` | Checkpoints (transient, many files) and the final model (persistent artifact) have distinct lifecycles and storage requirements |
| `save_strategy="steps"` | Step-level checkpointing enables fine-grained resume on preemptible / spot VMs |
| `resume_from_checkpoint="auto"` | Automatically picks the latest checkpoint so job scripts can restart safely without hardcoding paths |
| `bfloat16` precision | Natively stable on AMD Instinct & RDNA3+ (avoids float16 overflow) |
| `attn_implementation="eager"` | Flash-attention-2 requires a custom ROCm build; eager avoids the dependency |
| `optim="adamw_torch_fused"` | PyTorch fused AdamW works on both ROCm and CUDA without custom CUDA extensions |
| `dataloader_pin_memory=False` | ROCm uses unified memory; pinning can cause allocation conflicts |
| `merge_and_unload()` | Produces a plain model with no PEFT wrappers — natively loadable by vLLM |
| `safe_serialization=True` | Saves in safetensors format which vLLM requires |
| No custom forward passes | Ensures full HuggingFace serialisation compatibility with vLLM |

---

## Fraud Label Reference

| Label                    | Description                                                          |
|--------------------------|----------------------------------------------------------------------|
| `CLEAN`                  | Legitimate transaction with no suspicious indicators                 |
| `CARD_TESTING`           | Small probing charges used to verify stolen card validity            |
| `ACCOUNT_TAKEOVER`       | Unauthorized access using compromised customer credentials           |
| `GEO_ANOMALY`            | Transaction from an implausible geographic location                  |
| `MONEY_LAUNDERING_RING`  | Structured transfers designed to obscure illicit fund movement       |
| `PHISHING`               | Transaction preceded by social-engineering credential theft          |
| `IDENTITY_THEFT`         | Account created or modified using a victim's stolen identity         |

---

## Troubleshooting AMD ROCm Issues

### `torch.cuda.is_available()` returns `False`

```bash
# Check ROCm kernel module is loaded
lsmod | grep amdgpu

# Check your user is in the render group
groups $USER   # should include 'render' and 'video'
sudo usermod -aG render,video $USER && newgrp render
```

### Out-of-memory (OOM) errors

1. Reduce `--batch_size` to `1` or `2`.
2. Enable `--use_4bit` quantisation.
3. Reduce `--max_seq_len` to `256`.
4. Set `gradient_accumulation_steps` higher in the script to maintain effective batch size.

### `HSA_OVERRIDE_GFX_VERSION` warning

If your GPU architecture is not auto-detected, set this manually before running:

```bash
# RDNA3 (RX 7000 series)
export HSA_OVERRIDE_GFX_VERSION=11.0.0

# RDNA2 (RX 6000 series)
export HSA_OVERRIDE_GFX_VERSION=10.3.0

# MI250X / MI300X
export HSA_OVERRIDE_GFX_VERSION=9.0.10
```

### `bitsandbytes` import errors

The standard PyPI bitsandbytes package may not have a ROCm backend. If `--use_4bit` fails:

```bash
pip uninstall bitsandbytes
pip install bitsandbytes --prefer-binary --extra-index-url \
    https://huggingface.github.io/bitsandbytes-rocm/
```

### Slow training speed

```bash
# Confirm PyTorch is using the GPU, not CPU
python -c "import torch; print(torch.cuda.device_count(), torch.cuda.get_device_name(0))"

# Monitor GPU utilisation during training
watch -n 1 rocm-smi
```

### Persistent volume not mounted

```bash
# Verify the volume is mounted before starting training
mountpoint /mnt/data   # exits 0 if mounted, 1 if not
df -h /mnt/data        # check available space

# If not mounted, remount manually (example for a block device)
sudo mount /dev/sdb1 /mnt/data
```

---

## License

This project is for research and educational purposes. Ensure compliance with the licence terms of the chosen base model (Llama-3 Community Licence / Microsoft Phi-3 MIT Licence) before any commercial deployment.
