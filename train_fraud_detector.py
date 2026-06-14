"""
Financial Fraud Detection - Model Training Script
==================================================
Target Hardware : AMD GPU (ROCm / HIP)
Base Model      : Qwen/Qwen2.5-3B-Instruct
Task            : Multi-class fraud classification via supervised fine-tuning
Output          : Merged, standalone model weights ready for `vllm serve`

Prerequisites:
    Run preprocess_data.py first to prepare the formatted dataset:

        python preprocess_data.py \\
            --csv_path     sample-dataset/fraud-detection-dataset-1.csv \\
            --base_model   Qwen/Qwen2.5-3B-Instruct \\
            --output_dir   ./processed-data

Usage (fresh training run):
    python train_fraud_detector.py \\
        --base_model         Qwen/Qwen2.5-3B-Instruct \\
        --processed_data_dir ./processed-data \\
        --checkpoint_dir     ./checkpoints \\
        --model_output_dir   ./fraud-detector-merged \\
        --epochs             3 \\
        --batch_size         4 \\
        --lr                 2e-4

Usage (resume from a saved checkpoint):
    python train_fraud_detector.py \\
        --base_model         Qwen/Qwen2.5-3B-Instruct \\
        --processed_data_dir ./processed-data \\
        --checkpoint_dir     ./checkpoints \\
        --model_output_dir   ./fraud-detector-merged \\
        --resume_from_checkpoint ./checkpoints/checkpoint-120 \\
        --epochs             3
"""

# ─────────────────────────────────────────────
# 0.  Imports
# ─────────────────────────────────────────────
import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Optional

import torch
from datasets import load_from_disk

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from trl import SFTTrainer, SFTConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# 1.  AMD ROCm device validation
# ─────────────────────────────────────────────
def validate_rocm_device() -> torch.device:
    """
    On ROCm builds of PyTorch, HIP exposes itself via the standard 'cuda'
    device string. We validate the build is genuinely ROCm-backed so that
    bfloat16 and flash-attention paths are correctly activated.
    """
    if not torch.cuda.is_available():
        log.warning(
            "No CUDA / ROCm device detected — falling back to CPU. "
            "Training will be extremely slow on CPU."
        )
        return torch.device("cpu")

    device = torch.device("cuda:0")
    device_name: str = torch.cuda.get_device_name(0)
    log.info("GPU detected  : %s", device_name)
    log.info("PyTorch build : %s", torch.version.cuda or "ROCm/HIP")

    # Detect whether we are actually on a ROCm stack.
    is_rocm = hasattr(torch.version, "hip") and torch.version.hip is not None
    if is_rocm:
        log.info("ROCm version  : %s", torch.version.hip)
        os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")   # RDNA3 safe default
        os.environ.setdefault("PYTORCH_HIP_ALLOC_CONF", "garbage_collection_threshold:0.9,max_split_size_mb:512")
    else:
        log.info("Running on CUDA (NVIDIA) — AMD-specific env vars skipped.")

    return device


# ─────────────────────────────────────────────
# 2.  Model & tokenizer initialisation
# ─────────────────────────────────────────────
def load_model_and_tokenizer(
    base_model: str,
    device: torch.device,
    use_4bit: bool = False,
):
    """
    Load the base model in bfloat16 (or 4-bit NF4 for very low VRAM) and
    attach LoRA adapters to all linear projection layers.
    """
    log.info("Loading tokenizer for: %s", base_model)
    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        trust_remote_code=True,
        padding_side="right",    # required for SFT left-padding to work correctly
    )

    # Ensure a pad token exists (many LLMs only have EOS).
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # ── Quantisation config (optional 4-bit for very tight VRAM) ──────────
    bnb_config: Optional[BitsAndBytesConfig] = None
    if use_4bit:
        log.info("Using 4-bit NF4 quantisation (bitsandbytes).")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    log.info("Loading base model: %s", base_model)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,           # stable on AMD Instinct & RDNA3+
        device_map="auto",                     # distributes across all visible GPUs
        quantization_config=bnb_config,
        trust_remote_code=True,
        attn_implementation="eager",           # flash-attn2 may need ROCm-specific build
    )

    if use_4bit:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=True,
        )
    else:
        # Enable gradient checkpointing to conserve VRAM.
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    model.enable_input_require_grads()   # required when gradient checkpointing + PEFT

    # ── LoRA configuration ─────────────────────────────────────────────────
    # Target all four attention projection matrices; rank 16 balances capacity
    # and memory. alpha=32 keeps the effective LR stable.
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        # gate / up / down projections in MLP layers (optional, uncomment for
        # more capacity at the cost of higher VRAM):
        # target_modules=["q_proj","k_proj","v_proj","o_proj",
        #                 "gate_proj","up_proj","down_proj"],
        modules_to_save=None,   # keep clean — no non-serialisable hooks
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, tokenizer


# ─────────────────────────────────────────────
# 3.  Training
# ─────────────────────────────────────────────
def train(
    model,
    tokenizer,
    dataset,
    checkpoint_dir: str,
    epochs: int,
    batch_size: int,
    lr: float,
    max_seq_len: int,
    warmup_ratio: float = 0.05,
    weight_decay: float = 0.01,
    resume_from_checkpoint: Optional[str] = None,
    save_steps: int = 50,
):
    """
    Configure and launch SFTTrainer.

    Checkpoints are saved to `checkpoint_dir` every `save_steps` steps AND
    at the end of every epoch. Pass `resume_from_checkpoint` to resume a
    previously interrupted run from any saved checkpoint directory.

    All components are standard HuggingFace / TRL so the saved artifacts are
    natively compatible with vLLM (no custom forward passes or non-serialisable
    token hooks).
    """
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # ── Detect resume target ───────────────────────────────────────────────
    # If the caller passes "auto", pick the latest checkpoint automatically.
    resume_target: Optional[str] = None
    if resume_from_checkpoint == "auto":
        checkpoint_path = Path(checkpoint_dir)
        ckpt_dirs = sorted(checkpoint_path.glob("checkpoint-*"))
        if ckpt_dirs:
            resume_target = str(ckpt_dirs[-1])
            log.info("Auto-resuming from latest checkpoint: %s", resume_target)
        else:
            log.info("No checkpoints found in %s — starting fresh.", checkpoint_dir)
    elif resume_from_checkpoint:
        resume_target = resume_from_checkpoint
        log.info("Resuming from checkpoint: %s", resume_target)
    else:
        log.info("Starting training from scratch.")

    import inspect

    # 1. Build SFTConfig arguments dynamically
    config_kwargs = {
        "output_dir": checkpoint_dir,

        # ── Epochs & steps ────────────────────────────────────────────────
        "num_train_epochs": epochs,
        "max_steps": -1,

        # ── Batch / gradient accumulation ────────────────────────────────
        "per_device_train_batch_size": batch_size,
        "gradient_accumulation_steps": 4,
        "gradient_checkpointing": True,

        # ── Optimiser ─────────────────────────────────────────────────────
        "optim": "adamw_torch_fused",
        "learning_rate": lr,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": warmup_ratio,
        "weight_decay": weight_decay,
        "max_grad_norm": 1.0,

        # ── Precision ─────────────────────────────────────────────────────
        "bf16": True,
        "fp16": False,

        # ── Logging & checkpointing ───────────────────────────────────────
        "logging_steps": 10,
        "save_strategy": "steps",
        "save_steps": save_steps,
        "save_total_limit": 5,
        "report_to": "none",

        # ── Data loading ──────────────────────────────────────────────────
        "dataloader_num_workers": 2,
        "dataloader_pin_memory": False,

        # ── Misc ──────────────────────────────────────────────────────────
        "seed": 42,
        "data_seed": 42,
        "remove_unused_columns": True,
    }

    config_sig = inspect.signature(SFTConfig.__init__)
    sft_in_config = "max_seq_length" in config_sig.parameters

    if sft_in_config:
        config_kwargs["max_seq_length"] = max_seq_len
        config_kwargs["dataset_text_field"] = "text"
        config_kwargs["packing"] = False

    sft_config = SFTConfig(**config_kwargs)

    # 2. Build SFTTrainer arguments dynamically
    trainer_sig = inspect.signature(SFTTrainer.__init__)
    trainer_kwargs = {
        "model": model,
        "args": sft_config,
        "train_dataset": dataset,
    }

    if not sft_in_config:
        trainer_kwargs["max_seq_length"] = max_seq_len
        trainer_kwargs["dataset_text_field"] = "text"
        trainer_kwargs["packing"] = False

    # Check for processing_class vs tokenizer in Trainer signature
    if "processing_class" in trainer_sig.parameters:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = SFTTrainer(**trainer_kwargs)

    log.info("Starting supervised fine-tuning …")
    trainer.train(resume_from_checkpoint=resume_target)
    log.info("Training complete.")

    return trainer


# ─────────────────────────────────────────────
# 4.  Merge LoRA → base and save for vLLM
# ─────────────────────────────────────────────
def merge_and_save(model, tokenizer, model_output_dir: str):
    """
    Merge the trained LoRA adapter weights back into the base model's weight
    matrices, then serialise the unified model and tokeniser to disk.

    The resulting directory can be served immediately with:
        vllm serve <model_output_dir> --dtype bfloat16

    No adapter flags (--lora-modules, etc.) are required because all weights
    are embedded in the base model tensor layout.
    """
    log.info("Merging LoRA adapter weights into base model …")

    # .merge_and_unload() folds all adapter deltas (A×B) into the original
    # weight matrices and returns a plain nn.Module (no PEFT wrappers).
    merged_model = model.merge_and_unload()

    merged_dir = Path(model_output_dir)
    merged_dir.mkdir(parents=True, exist_ok=True)

    log.info("Saving merged model to: %s", merged_dir)
    merged_model.save_pretrained(
        str(merged_dir),
        safe_serialization=True,    # safetensors format — required by vLLM
        max_shard_size="4GB",
    )

    log.info("Saving tokenizer to: %s", merged_dir)
    tokenizer.save_pretrained(str(merged_dir))

    log.info(
        "✅  Merged model saved. Serve with:\n"
        "    vllm serve %s --dtype bfloat16 --trust-remote-code",
        merged_dir,
    )


# ─────────────────────────────────────────────
# 5.  CLI entry-point
# ─────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune an SLM for financial fraud detection on AMD ROCm. "
            "Run preprocess_data.py first to prepare the dataset."
        )
    )

    # ── Input ─────────────────────────────────────────────────────────────
    parser.add_argument(
        "--base_model",
        type=str,
        default="Qwen/Qwen2.5-3B-Instruct",
        help="HuggingFace model ID or local path of the base model.",
    )
    parser.add_argument(
        "--processed_data_dir",
        type=str,
        default="./processed-data",
        help=(
            "Directory containing the HuggingFace Dataset produced by "
            "preprocess_data.py (load_from_disk target)."
        ),
    )

    # ── Output locations ──────────────────────────────────────────────────
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="./checkpoints",
        help=(
            "Directory where training checkpoints are saved during training. "
            "Checkpoints are written every --save_steps steps and at epoch end."
        ),
    )
    parser.add_argument(
        "--model_output_dir",
        type=str,
        default="./fraud-detector-merged",
        help=(
            "Directory where the final merged, vLLM-ready model will be saved "
            "after training completes."
        ),
    )

    # ── Resume ────────────────────────────────────────────────────────────
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Path to a specific checkpoint directory to resume training from "
            "(e.g. ./checkpoints/checkpoint-120). "
            "Pass 'auto' to automatically resume from the latest checkpoint "
            "found in --checkpoint_dir."
        ),
    )

    # ── Hyperparameters ───────────────────────────────────────────────────
    parser.add_argument(
        "--epochs",     type=int,   default=3,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch_size", type=int,   default=4,
        help="Per-device training batch size.",
    )
    parser.add_argument(
        "--lr",         type=float, default=2e-4,
        help="Peak learning rate.",
    )
    parser.add_argument(
        "--max_seq_len", type=int,  default=512,
        help="Maximum token sequence length.",
    )
    parser.add_argument(
        "--save_steps", type=int,   default=50,
        help="Save a checkpoint every N optimizer steps.",
    )
    parser.add_argument(
        "--use_4bit",
        action="store_true",
        default=False,
        help="Enable 4-bit NF4 quantisation (requires bitsandbytes >= 0.41).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # ── 1. Hardware check ─────────────────────────────────────────────────
    device = validate_rocm_device()

    # ── 2. Load pre-processed dataset from disk ───────────────────────────
    log.info("Loading preprocessed dataset from: %s", args.processed_data_dir)
    if not Path(args.processed_data_dir).exists():
        log.error(
            "Processed data directory not found: %s\n"
            "Run preprocess_data.py first to generate it.",
            args.processed_data_dir,
        )
        sys.exit(1)
    dataset = load_from_disk(args.processed_data_dir)
    log.info("Dataset loaded: %d examples", len(dataset))

    # ── 3. Load model + tokenizer with LoRA ──────────────────────────────
    model, tokenizer = load_model_and_tokenizer(
        base_model=args.base_model,
        device=device,
        use_4bit=args.use_4bit,
    )

    # ── 4. Fine-tune ──────────────────────────────────────────────────────
    trainer = train(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        checkpoint_dir=args.checkpoint_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_seq_len=args.max_seq_len,
        resume_from_checkpoint=args.resume_from_checkpoint,
        save_steps=args.save_steps,
    )

    # ── 5. Merge adapters and save a vLLM-ready model ────────────────────
    merge_and_save(
        model=trainer.model,
        tokenizer=tokenizer,
        model_output_dir=args.model_output_dir,
    )

    log.info(
        "Pipeline complete.\n"
        "  Checkpoints : %s\n"
        "  Merged model: %s",
        args.checkpoint_dir,
        args.model_output_dir,
    )


if __name__ == "__main__":
    main()
