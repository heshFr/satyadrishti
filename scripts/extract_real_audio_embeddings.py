"""
Extract real audio embeddings from ASVspoof/VoxCeleb dataset files
using the pretrained AST model, replacing the synthetic sine wave approach.

Usage:
    python -m scripts.extract_real_audio_embeddings --audio_dir datasets/audio
    python -m scripts.extract_real_audio_embeddings --audio_dir datasets/audio --max_samples 2000
"""
import os
import glob
import numpy as np
import torch
import soundfile as sf
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAVE_PATH = PROJECT_ROOT / "models" / "fusion" / "real_embeddings" / "audio_embeddings.npz"


def extract(audio_dir: str, max_samples: int = 2000):
    from engine.audio.ast_spoof import ASTSpoofDetector

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ASTSpoofDetector()
    model.to(device).eval()

    audio_files = []
    for ext in ("*.wav", "*.flac", "*.mp3"):
        audio_files.extend(glob.glob(os.path.join(audio_dir, "**", ext), recursive=True))

    if not audio_files:
        print(f"No audio files found in {audio_dir}")
        print("Falling back to TTS-generated embeddings...")
        return extract_tts_fallback(model, device, max_samples)

    rng = np.random.RandomState(42)
    if len(audio_files) > max_samples:
        indices = rng.choice(len(audio_files), max_samples, replace=False)
        audio_files = [audio_files[i] for i in indices]

    embeddings, labels = [], []
    for i, fpath in enumerate(audio_files):
        try:
            waveform, sr = sf.read(fpath, dtype="float32")
            if waveform.ndim > 1:
                waveform = waveform.mean(axis=0)

            # Determine label from path or filename conventions
            # ASVspoof: bonafide vs spoof in directory name or protocol
            path_lower = fpath.lower()
            if "spoof" in path_lower or "fake" in path_lower or "LA_T" in fpath:
                label = 1  # spoof
            else:
                label = 0  # bonafide

            inputs = model.preprocess(waveform, sr)
            input_values = inputs["input_values"].to(device)
            with torch.no_grad():
                emb = model.extract_embedding(input_values)
                embeddings.append(emb.cpu().squeeze(0).numpy())
                labels.append(label)
        except Exception as e:
            continue

        if i % 100 == 0:
            print(f"  Processed {i}/{len(audio_files)}")

    embeddings = np.stack(embeddings)
    labels = np.array(labels)

    os.makedirs(SAVE_PATH.parent, exist_ok=True)
    np.savez(SAVE_PATH, embeddings=embeddings, labels=labels)
    print(f"Saved {len(embeddings)} real audio embeddings to {SAVE_PATH}")
    print(f"Distribution: {dict(zip(*np.unique(labels, return_counts=True)))}")


def extract_tts_fallback(model, device, n_samples):
    """If no real audio data is available, use gTTS to generate
    realistic speech embeddings instead of sine waves."""
    try:
        from gtts import gTTS
        import tempfile
    except ImportError:
        print("ERROR: No real audio data and gtts not installed.")
        print("Install with: pip install gtts")
        print("Or provide real audio files in datasets/audio/")
        return

    # Generate diverse speech samples
    safe_texts = [
        "Hello, how are you today?",
        "The weather is nice outside.",
        "I'm calling to check on your order status.",
        "Can you hear me clearly?",
        "Thank you for your time today.",
    ]
    coercive_texts = [
        "You must transfer the money immediately or face consequences.",
        "This is urgent, your account will be closed within one hour.",
        "I am calling from the tax department, you owe back taxes.",
        "Send me the OTP code right now, there is no time to waste.",
        "If you don't pay now, the police will come to your house.",
    ]

    rng = np.random.RandomState(42)
    embeddings, labels = [], []

    for i in range(n_samples):
        if i < n_samples // 2:
            text = rng.choice(safe_texts)
            label = 0
        else:
            text = rng.choice(coercive_texts)
            label = 1

        try:
            tts = gTTS(text=text, lang="en")
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=True) as tmp:
                tts.save(tmp.name)
                waveform, sr = sf.read(tmp.name, dtype="float32")
                if waveform.ndim > 1:
                    waveform = waveform.mean(axis=0)

                inputs = model.preprocess(waveform, sr)
                input_values = inputs["input_values"].to(device)
                with torch.no_grad():
                    emb = model.extract_embedding(input_values)
                    embeddings.append(emb.cpu().squeeze(0).numpy())
                    labels.append(label)
        except Exception:
            continue

    if embeddings:
        embeddings = np.stack(embeddings)
        labels = np.array(labels)
        os.makedirs(SAVE_PATH.parent, exist_ok=True)
        np.savez(SAVE_PATH, embeddings=embeddings, labels=labels)
        print(f"Saved {len(embeddings)} TTS audio embeddings to {SAVE_PATH}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_dir", type=str, default=str(PROJECT_ROOT / "datasets" / "audio"))
    parser.add_argument("--max_samples", type=int, default=2000)
    args = parser.parse_args()
    extract(args.audio_dir, args.max_samples)
