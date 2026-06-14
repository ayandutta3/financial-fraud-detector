"""
Financial Fraud Detection - Data Preprocessing Script
======================================================
Converts the raw CSV dataset into a tokenised, chat-template-formatted
HuggingFace Dataset saved to disk. Run this ONCE before training so that
the preprocessing step is fully decoupled from the training loop.

The output directory can be reused across multiple training runs without
re-reading or re-formatting the CSV every time.

Usage:
    python preprocess_data.py \\
        --csv_path      sample-dataset/fraud-detection-dataset-1.csv \\
        --base_model    Qwen/Qwen2.5-3B-Instruct \\
        --output_dir    ./processed-data \\
        --max_seq_len   512
"""

# ─────────────────────────────────────────────
# 0.  Imports
# ─────────────────────────────────────────────
import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# 1.  Fraud-class label mapping
# ─────────────────────────────────────────────
FRAUD_LABEL_MAP = {
    "":                      "CLEAN",
    "clean":                 "CLEAN",
    "card_testing":          "CARD_TESTING",
    "account_takeover":      "ACCOUNT_TAKEOVER",
    "geo_anomaly":           "GEO_ANOMALY",
    "money_laundering_ring": "MONEY_LAUNDERING_RING",
    "phishing":              "PHISHING",
    "identity_theft":        "IDENTITY_THEFT",
}

LABEL_DESCRIPTIONS = (
    "CLEAN — Legitimate transaction with no suspicious indicators.\n"
    "CARD_TESTING — Small probing charges used to verify stolen card validity.\n"
    "ACCOUNT_TAKEOVER — Unauthorized access using compromised customer credentials.\n"
    "GEO_ANOMALY — Transaction originating from an implausible geographic location.\n"
    "MONEY_LAUNDERING_RING — Structured transfer activity designed to obscure illicit funds.\n"
    "PHISHING — Transaction preceded by social-engineering credential theft.\n"
    "IDENTITY_THEFT — Account created or modified using a victim's stolen identity."
)

SYSTEM_PROMPT = (
    "You are a Senior Financial Forensic Analyst trained to classify payment transactions. "
    "For each transaction, output ONLY the single most applicable fraud classification label. "
    "Valid labels and their definitions:\n"
    f"{LABEL_DESCRIPTIONS}\n"
    "Respond with exactly one label from the list above and nothing else."
)


# ─────────────────────────────────────────────
# 2.  Helper functions
# ─────────────────────────────────────────────
def normalize_fraud_label(is_fraud: int, fraud_type: str) -> str:
    """Resolve the canonical multi-class label from raw CSV columns."""
    if int(is_fraud) == 0:
        return "CLEAN"
    raw = str(fraud_type).strip().lower().replace(" ", "_")
    return FRAUD_LABEL_MAP.get(raw, raw.upper())


def build_chat_narrative(row: pd.Series) -> list[dict]:
    """
    Serialise a single transaction row into a list of chat messages.
    The tokenizer's apply_chat_template() will wrap these into the
    model-specific token layout (BOS, EOS, role headers).
    """
    label = normalize_fraud_label(row.get("is_fraud", 0), row.get("fraud_type", ""))

    user_content = (
        f"Analyse the following payment transaction and classify it:\n\n"
        f"  Transaction ID   : {row.get('transaction_id', 'N/A')}\n"
        f"  Timestamp        : {row.get('timestamp', 'N/A')}\n"
        f"  Customer ID      : {row.get('customer_id', 'N/A')}\n"
        f"  Merchant Category: {row.get('merchant_category', 'N/A')}\n"
        f"  Transaction Type : {row.get('transaction_type', 'N/A')}\n"
        f"  Amount (USD)     : {row.get('amount', 'N/A')}\n"
        f"  Merchant Country : {row.get('merchant_country', 'N/A')}\n"
        f"  Merchant City    : {row.get('merchant_city', 'N/A')}\n"
        f"  Card ID          : {row.get('card_id', 'N/A')}\n"
        f"  Device ID        : {row.get('device_id', 'N/A')}\n"
        f"  IP Address       : {row.get('ip_address', 'N/A')}\n\n"
        f"What is the fraud classification for this transaction?"
    )

    return [
        {"role": "system",    "content": SYSTEM_PROMPT},
        {"role": "user",      "content": user_content},
        {"role": "assistant", "content": label},
    ]


# ─────────────────────────────────────────────
# 3.  Core preprocessing pipeline
# ─────────────────────────────────────────────
def preprocess(
    csv_path: str,
    base_model: str,
    output_dir: str,
    max_seq_len: int = 512,
) -> None:
    """
    End-to-end preprocessing pipeline:
      1. Load tokenizer (needed only for apply_chat_template).
      2. Read and validate the CSV.
      3. Apply chat-template formatting to every row.
      4. Save the HuggingFace Dataset to disk.

    The saved dataset can be loaded later with:
        datasets.load_from_disk(output_dir)
    """
    output_path = Path(output_dir)
    if output_path.exists():
        log.warning(
            "Output directory already exists: %s — existing data will be overwritten.",
            output_path,
        )

    # ── Tokenizer (needed only for apply_chat_template) ────────────────────
    log.info("Loading tokenizer for: %s", base_model)
    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        trust_remote_code=True,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # ── Load CSV ────────────────────────────────────────────────────────────
    log.info("Reading CSV from: %s", csv_path)
    df = pd.read_csv(csv_path)
    df["is_fraud"]   = df["is_fraud"].fillna(0).astype(int)
    df["fraud_type"] = df["fraud_type"].fillna("").astype(str)

    label_counts = df.apply(
        lambda r: normalize_fraud_label(r["is_fraud"], r["fraud_type"]), axis=1
    ).value_counts()
    log.info("Label distribution:\n%s", label_counts.to_string())

    # ── Format rows ─────────────────────────────────────────────────────────
    def format_row(row: pd.Series) -> dict:
        messages = build_chat_narrative(row)
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": text}

    log.info("Formatting %d rows …", len(df))
    records = [format_row(row) for _, row in df.iterrows()]

    # ── Build HuggingFace Dataset ───────────────────────────────────────────
    hf_dataset = Dataset.from_list(records)
    log.info("Dataset size: %d examples", len(hf_dataset))

    # ── Save to disk ────────────────────────────────────────────────────────
    output_path.mkdir(parents=True, exist_ok=True)
    hf_dataset.save_to_disk(str(output_path))
    log.info("✅  Preprocessed dataset saved to: %s", output_path)
    log.info(
        "Load in training with:\n"
        "    from datasets import load_from_disk\n"
        "    dataset = load_from_disk('%s')",
        output_path,
    )


# ─────────────────────────────────────────────
# 4.  CLI entry-point
# ─────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Preprocess the fraud-detection CSV into a HuggingFace Dataset "
            "saved to disk, ready for training."
        )
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default="sample-dataset/fraud-detection-dataset-1.csv",
        help="Path to the balanced CSV dataset.",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="Qwen/Qwen2.5-3B-Instruct",
        help="HuggingFace model ID (used only to load the tokenizer for chat templating).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./processed-data",
        help="Directory where the preprocessed HuggingFace Dataset will be saved.",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=512,
        help="Maximum token sequence length (informational; stored as metadata).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    preprocess(
        csv_path=args.csv_path,
        base_model=args.base_model,
        output_dir=args.output_dir,
        max_seq_len=args.max_seq_len,
    )
