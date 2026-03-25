"""
Semantic Coercion Engine — DeBERTaV3 + LoRA
=============================================
Detects psychological manipulation, urgency, financial coercion,
and authority impersonation in real-time text transcripts.

Uses Parameter-Efficient Fine-Tuning (PEFT) via LoRA to adapt
a DeBERTaV3 backbone on coercion-specific datasets while keeping
the model small enough for edge deployment.
"""

import os
import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import LoraConfig, get_peft_model, PeftModel, TaskType


# Label mapping
COERCION_LABELS = {
    0: "safe",
    1: "urgency_manipulation",
    2: "financial_coercion",
    3: "combined_threat",
}


class CoercionDetector:
    """
    DeBERTaV3-based semantic coercion detector with LoRA fine-tuning.

    Args:
        model_name: HuggingFace model identifier
        num_labels: Number of coercion categories
        lora_r: LoRA rank
        lora_alpha: LoRA scaling factor
        lora_dropout: Dropout for LoRA layers
        max_length: Maximum token sequence length
    """

    def __init__(
        self,
        model_name: str = "microsoft/deberta-v3-base",
        num_labels: int = 4,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        max_length: int = 512,
        checkpoint_dir: str = None,
    ):
        self.model_name = model_name
        self.num_labels = num_labels
        self.max_length = max_length

        if checkpoint_dir and os.path.isdir(checkpoint_dir):
            # Load trained model from saved PEFT adapter
            self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, use_fast=True)
            base_model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=num_labels,
                problem_type="single_label_classification",
            )
            self.model = PeftModel.from_pretrained(base_model, checkpoint_dir)
            self.base_model = base_model
            print(f"[Text] Loaded trained LoRA adapter from {checkpoint_dir}")
            # Merge LoRA weights into base model for faster and more accurate inference
            # Without this, the adapter weights are applied separately which can
            # cause subtle numerical differences
            try:
                self.model = self.model.merge_and_unload()
                print(f"[Text] LoRA weights merged into base model for optimal inference.")
            except Exception as e:
                print(f"[Text] Could not merge LoRA (using adapter mode): {e}")
        else:
            # Fresh model for training
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.base_model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=num_labels,
                problem_type="single_label_classification",
            )
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=["query_proj", "value_proj", "key_proj"],
                bias="none",
                task_type=TaskType.SEQ_CLS,
            )
            self.model = get_peft_model(self.base_model, lora_config)

    def get_model(self) -> nn.Module:
        """Return the LoRA-adapted model for training."""
        return self.model

    def get_tokenizer(self):
        """Return the tokenizer."""
        return self.tokenizer

    def tokenize(self, texts: list[str]) -> dict:
        """
        Tokenize a batch of text transcripts.

        Args:
            texts: List of transcript strings

        Returns:
            Tokenized batch dict with input_ids, attention_mask, etc.
        """
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

    def predict(self, text: str, device: str = "cpu") -> dict:
        """
        Run inference on a single transcript.

        Args:
            text: Input transcript string
            device: Target device

        Returns:
            dict with predicted label, confidence, and all probabilities
        """
        self.model.eval()
        self.model.to(device)

        inputs = self.tokenize([text])
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            # Temperature scaling for better-calibrated confidence
            # T > 1 softens overconfident predictions
            temperature = 1.5
            scaled_logits = outputs.logits / temperature
            probs = torch.softmax(scaled_logits, dim=-1).squeeze()

        pred_idx = probs.argmax().item()
        return {
            "label": COERCION_LABELS[pred_idx],
            "confidence": probs[pred_idx].item(),
            "probabilities": {
                COERCION_LABELS[i]: probs[i].item()
                for i in range(self.num_labels)
            },
        }

    def extract_embedding(self, text: str, device: str = "cpu") -> torch.Tensor:
        """
        Extract the [CLS] token embedding for Cross-Modal Fusion.

        Returns:
            embedding: (1, hidden_dim) tensor
        """
        self.model.eval()
        self.model.to(device)

        inputs = self.tokenize([text])
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            # Run full forward pass and extract the last hidden state's [CLS] token
            outputs = self.model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                output_hidden_states=True,
            )
            cls_embedding = outputs.hidden_states[-1][:, 0, :]

        return cls_embedding

    def print_trainable_params(self):
        """Print the number of trainable vs. total parameters."""
        self.model.print_trainable_parameters()
