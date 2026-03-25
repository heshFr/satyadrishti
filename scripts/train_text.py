"""
Text Engine Training — DeBERTaV3 + LoRA Fine-Tuning
=====================================================
Fine-tunes microsoft/deberta-v3-base with LoRA adapters on the
synthetic coercion dataset. Uses HuggingFace Trainer for robust
training with mixed precision, gradient accumulation, and evaluation.

Usage:
    python scripts/train_text.py
    python scripts/train_text.py --epochs 15 --batch_size 8
"""

import json
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
)

# ─── Paths ──────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "datasets" / "coercion"
MODEL_DIR = PROJECT_ROOT / "models" / "text"

LABEL_MAP = {0: "safe", 1: "urgency_manipulation", 2: "financial_coercion", 3: "combined_threat"}


# ─── Dataset ────────────────────────────────────────────────

class CoercionDataset(Dataset):
    """PyTorch dataset wrapping JSONL coercion transcripts."""

    def __init__(self, filepath: Path, tokenizer, max_length: int = 512):
        self.samples = []
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                self.samples.append(json.loads(line.strip()))
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        encoding = self.tokenizer(
            sample["text"],
            truncation=True,
            max_length=self.max_length,
        )
        return {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "labels": sample["label"],
        }


# ─── Metrics ────────────────────────────────────────────────

def compute_metrics(eval_pred):
    """Compute accuracy, F1, precision, recall for HF Trainer."""
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=1)

    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
        "f1_weighted": f1_score(labels, preds, average="weighted"),
        "precision_macro": precision_score(labels, preds, average="macro", zero_division=0),
        "recall_macro": recall_score(labels, preds, average="macro", zero_division=0),
    }


# ─── Main Training ──────────────────────────────────────────

def train(args):
    print("=" * 60)
    print("Satya Drishti — Text Engine Training")
    print("=" * 60)

    # ── 1. Load tokenizer ──
    print(f"\n[1/6] Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # ── 2. Load datasets ──
    print(f"[2/6] Loading datasets from {DATA_DIR}")
    train_dataset = CoercionDataset(DATA_DIR / "train.jsonl", tokenizer, args.max_length)
    val_dataset = CoercionDataset(DATA_DIR / "val.jsonl", tokenizer, args.max_length)
    test_dataset = CoercionDataset(DATA_DIR / "test.jsonl", tokenizer, args.max_length)
    print(f"  Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")

    # ── 3. Load base model ──
    print(f"[3/6] Loading base model: {args.model_name}")
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=4,
        problem_type="single_label_classification",
        id2label=LABEL_MAP,
        label2id={v: k for k, v in LABEL_MAP.items()},
    )

    # ── 4. Apply LoRA ──
    print(f"[4/6] Applying LoRA (r={args.lora_r}, alpha={args.lora_alpha})")
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["query_proj", "value_proj", "key_proj"],
        modules_to_save=["classifier", "pooler"],
        bias="none",
        task_type=TaskType.SEQ_CLS,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── 5. Training arguments ──
    output_dir = MODEL_DIR / "deberta_coercion_lora"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Detect device and mixed precision support
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
    print(f"\n[5/6] Training on: {device.upper()} (bf16={use_bf16})")
    if device == "cpu":
        print("  ⚠ Training on CPU — this will be slower but will work correctly.")

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=0.06,
        lr_scheduler_type="linear",
        bf16=use_bf16,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        save_total_limit=3,
        report_to="none",
        dataloader_num_workers=0 if sys.platform == "win32" else 4,
        remove_unused_columns=False,
    )

    # ── 6. Train ──
    print("[6/6] Starting training...\n")

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    train_result = trainer.train()

    # ── Save best model ──
    best_model_path = output_dir / "best_model"
    best_model_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(best_model_path))
    tokenizer.save_pretrained(str(best_model_path))
    print(f"\n[OK] Best model saved to: {best_model_path}")

    # ── Evaluate on test set ──
    print("\n" + "=" * 60)
    print("Final Evaluation on Test Set")
    print("=" * 60)

    test_results = trainer.evaluate(test_dataset, metric_key_prefix="test")
    for key, value in test_results.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")

    # ── Detailed classification report ──
    print("\n" + "-" * 60)
    print("Classification Report")
    print("-" * 60)

    predictions = trainer.predict(test_dataset)
    preds = np.argmax(predictions.predictions, axis=1)
    labels = predictions.label_ids

    report = classification_report(
        labels, preds,
        target_names=list(LABEL_MAP.values()),
        digits=4,
    )
    print(report)

    # ── Confusion matrix ──
    cm = confusion_matrix(labels, preds)
    print("Confusion Matrix:")
    print(cm)

    # ── Save metrics ──
    metrics = {
        "train_loss": train_result.training_loss,
        "test_accuracy": test_results.get("test_accuracy", 0),
        "test_f1_macro": test_results.get("test_f1_macro", 0),
        "test_f1_weighted": test_results.get("test_f1_weighted", 0),
        "epochs_trained": train_result.global_step / (len(train_dataset) // (args.batch_size * args.grad_accum)),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
    }
    with open(output_dir / "test_metrics.json", "w") as f:
        json.dump({k: v for k, v in metrics.items() if not isinstance(v, str)}, f, indent=2)

    print(f"\nTest F1 (macro): {test_results.get('test_f1_macro', 0):.4f}")
    print(f"Test Accuracy:   {test_results.get('test_accuracy', 0):.4f}")
    print(f"\n[DONE] Training complete! Results saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DeBERTaV3 + LoRA for coercion detection")
    parser.add_argument("--model_name", type=str, default="microsoft/deberta-v3-base")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    args = parser.parse_args()

    train(args)
