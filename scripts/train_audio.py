"""
Satya Drishti
=============
Trains the RawNet3 audio engine on ASVspoof 2021 for deepfake audio detection.
Includes custom Weighted Cross Entropy to handle the 5:1 Spoof:Bonafide class imbalance.
"""

import os
import argparse
import time

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset
    from sklearn.metrics import roc_curve
    from scipy.optimize import brentq
    from scipy.interpolate import interp1d
    import numpy as np
    from tqdm import tqdm
    HAS_ML = True
    
    from engine.audio.rawnet3 import RawNet3
except ImportError:
    HAS_ML = False


class ASVspoofDatasetStub(Dataset):
    """Stub dataset while ASVspoof downloads. Emulates 4s 16kHz audio."""
    def __init__(self, size=1000):
        self.size = size
        
    def __len__(self):
        return self.size
        
    def __getitem__(self, idx):
        # Shape: (channels, seq_len)
        x = torch.randn(1, 64000)
        # Class 0: Bonafide, Class 1: Spoof (imbalanced 1:5)
        y = torch.tensor(1 if np.random.rand() > 0.2 else 0, dtype=torch.long)
        return x, y


def compute_eer(y_true, y_score):
    """Calculates Equal Error Rate (EER), the standard metric for ASV verification."""
    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    thresh = interp1d(fpr, thresholds)(eer)
    return eer, thresh


def train(args):
    if not HAS_ML:
        print("Missing PyTorch. Run `pip install torch torchaudio scikit-learn numpy`")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Training Audio Engine (RawNet3) on {device} ---")
    
    # 1. Model
    model = RawNet3(num_classes=2).to(device)
    
    # 2. Data
    train_dataset = ASVspoofDatasetStub(size=8000)
    dev_dataset = ASVspoofDatasetStub(size=1000)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # 3. Optim & Loss
    # Weighted CE to heavily penalize missing a real (bonafide) voice
    # Assuming Bonafide(0): 20%, Spoof(1): 80%
    weights = torch.tensor([4.0, 1.0]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
       optimizer, T_0=5, T_mult=2, eta_min=1e-6
    )
    
    # 4. Loop
    os.makedirs(args.save_dir, exist_ok=True)
    best_eer = float("inf")
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        
        # Training Phase
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [Train]")
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, labels)
            
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
        scheduler.step()
        avg_train_loss = train_loss / len(train_loader)
        
        # Eval Phase
        model.eval()
        dev_loss = 0.0
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for inputs, labels in tqdm(dev_loader, desc=f"Epoch {epoch}/{args.epochs} [Eval]"):
                inputs, labels = inputs.to(device), labels.to(device)
                logits = model(inputs)
                loss = criterion(logits, labels)
                dev_loss += loss.item()
                
                probs = torch.softmax(logits, dim=1)[:, 1] # Prob of being spoof (class 1)
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                
        avg_dev_loss = dev_loss / len(dev_loader)
        eer, _ = compute_eer(np.array(all_labels), np.array(all_probs))
        
        print(f"\nEpoch {epoch} Summary:")
        print(f"  Train Loss: {avg_train_loss:.4f} | Dev Loss: {avg_dev_loss:.4f} | Dev EER: {eer*100:.2f}%")
        
        # Save best model
        if eer < best_eer:
            best_eer = eer
            save_path = os.path.join(args.save_dir, "rawnet3_best.pt")
            torch.save(model.state_dict(), save_path)
            print(f"  [!] New best EER! Saved model to {save_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--save-dir", type=str, default="models/audio")
    args = parser.parse_args()
    
    train(args)

if __name__ == "__main__":
    main()
