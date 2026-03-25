"""Quick test evaluation for the trained DeBERTaV3 + LoRA model."""
import json
import numpy as np
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
from peft import PeftModel
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "datasets" / "coercion"
MODEL_DIR = PROJECT_ROOT / "models" / "text" / "deberta_coercion_lora" / "best_model"
LABEL_MAP = {0: "safe", 1: "urgency_manipulation", 2: "financial_coercion", 3: "combined_threat"}


class CoercionDataset(Dataset):
    def __init__(self, filepath, tokenizer, max_length=256):
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
        encoding = self.tokenizer(sample["text"], truncation=True, max_length=self.max_length)
        return {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "labels": sample["label"],
        }


def main():
    print("Loading model from:", MODEL_DIR)
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))

    base_model = AutoModelForSequenceClassification.from_pretrained(
        "microsoft/deberta-v3-base",
        num_labels=4,
        problem_type="single_label_classification",
    )
    model = PeftModel.from_pretrained(base_model, str(MODEL_DIR))
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    test_dataset = CoercionDataset(DATA_DIR / "test.jsonl", tokenizer)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    loader = DataLoader(test_dataset, batch_size=32, collate_fn=collator)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            outputs = model(**inputs)
            preds = outputs.logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(batch["labels"].numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    print(f"\nTest Accuracy: {accuracy_score(all_labels, all_preds):.4f}")
    print(f"Test F1 (macro): {f1_score(all_labels, all_preds, average='macro'):.4f}")
    print(f"Test F1 (weighted): {f1_score(all_labels, all_preds, average='weighted'):.4f}")
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=list(LABEL_MAP.values()), digits=4))
    print("Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))


if __name__ == "__main__":
    main()
