"""
Satya Drishti -- ViT-B/16 Training on Modern AI Image Dataset (V2)
==================================================================
Fine-tunes the existing ViT-B/16 deepfake detector on a modern dataset
containing images from Midjourney 6/7, DALL-E 3, SDXL, Flux, GPT-Image-1,
Imagen 4, Chroma, Ideogram 3, and other state-of-the-art diffusion models.

V2 improvements:
- Platform-specific compression augmentations (WhatsApp, Instagram, Facebook,
  Telegram, Twitter) so the model learns to detect AI through compression
- Downscale/upscale augmentation (simulates re-sharing via screenshots)
- Random grayscale augmentation
- Memory-efficient streaming from parquet shards (loads one at a time)
- Supports ALL 206 shards

Dataset format: Parquet files with columns:
  - image: dict with 'bytes' key containing JPEG/PNG bytes
  - label: "real" or "fake"
  - model: source model name (e.g. "midjourney-6", "sdxl-1.0", etc.)

Usage:
    python scripts/train_vit_parquet.py --config configs/train_vit_v2.yaml
    python scripts/train_vit_parquet.py --config configs/train_vit_v2.yaml --resume models/image_forensics/deepfake_vit_b16.pt
"""

import argparse
import io
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml
import torch
import torch.nn as nn
from torch.utils.data import Dataset, IterableDataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from transformers import ViTForImageClassification

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# --- Compression Augmentations ---


class JPEGCompression:
    """Simulates general JPEG recompression. Uses turbojpeg if available (10x faster)."""

    def __init__(self, quality_range=(20, 95)):
        self.quality_range = quality_range
        self._tj = None
        try:
            from turbojpeg import TurboJPEG, TJPF_RGB
            self._tj = TurboJPEG()
            self._TJPF_RGB = TJPF_RGB
        except Exception:
            pass

    def __call__(self, img):
        quality = np.random.randint(self.quality_range[0], self.quality_range[1] + 1)
        if self._tj:
            arr = np.array(img)
            encoded = self._tj.encode(arr, quality=quality, pixel_format=self._TJPF_RGB)
            decoded = self._tj.decode(encoded, pixel_format=self._TJPF_RGB)
            return Image.fromarray(decoded)
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")


class PlatformSimulation:
    """
    Simulates social media platform image processing:
    resize to max dimension + JPEG compression at platform-specific quality.
    Uses turbojpeg if available for ~10x faster encode/decode.
    """

    def __init__(self, max_dim=1600, quality_range=(60, 80)):
        self.max_dim = max_dim
        self.quality_range = quality_range
        self._tj = None
        try:
            from turbojpeg import TurboJPEG, TJPF_RGB
            self._tj = TurboJPEG()
            self._TJPF_RGB = TJPF_RGB
        except Exception:
            pass

    def __call__(self, img):
        w, h = img.size
        if max(w, h) > self.max_dim:
            scale = self.max_dim / max(w, h)
            new_w, new_h = int(w * scale), int(h * scale)
            img = img.resize((new_w, new_h), Image.BILINEAR)

        quality = np.random.randint(self.quality_range[0], self.quality_range[1] + 1)
        if self._tj:
            arr = np.array(img)
            encoded = self._tj.encode(arr, quality=quality, pixel_format=self._TJPF_RGB)
            decoded = self._tj.decode(encoded, pixel_format=self._TJPF_RGB)
            return Image.fromarray(decoded)
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")


class DownscaleUpscale:
    """
    Simulates image quality degradation from re-sharing:
    downscale to a small size then upscale back.
    """

    def __init__(self, min_scale=0.3, max_scale=0.7):
        self.min_scale = min_scale
        self.max_scale = max_scale

    def __call__(self, img):
        w, h = img.size
        scale = np.random.uniform(self.min_scale, self.max_scale)
        small_w, small_h = max(16, int(w * scale)), max(16, int(h * scale))
        small = img.resize((small_w, small_h), Image.BILINEAR)
        return small.resize((w, h), Image.BILINEAR)


# --- Parquet Dataset ---


class StreamingParquetDataset(IterableDataset):
    """
    Streams images directly from parquet files without loading all 45GB into RAM.
    Uses memory-efficient row group reading with pyarrow.
    Optimized: larger batch reads, turbojpeg decode, bigger shuffle buffer.
    """
    def __init__(self, parquet_paths, num_samples, transform=None, shuffle_buffer=2048):
        super().__init__()
        self.parquet_paths = list(parquet_paths)
        self.num_samples = num_samples
        self.transform = transform
        self.shuffle_buffer = shuffle_buffer
        # Try turbojpeg for ~5x faster image decoding
        self._tj = None
        try:
            from turbojpeg import TurboJPEG, TJPF_RGB
            self._tj = TurboJPEG()
            self._TJPF_RGB = TJPF_RGB
        except Exception:
            pass

    def __len__(self):
        return self.num_samples

    def _decode_image(self, img_bytes):
        """Decode image bytes, using turbojpeg if available."""
        if self._tj:
            try:
                arr = self._tj.decode(img_bytes, pixel_format=self._TJPF_RGB)
                return Image.fromarray(arr)
            except Exception:
                pass
        try:
            return Image.open(io.BytesIO(img_bytes)).convert("RGB")
        except Exception:
            return Image.new("RGB", (224, 224), (128, 128, 128))

    def __iter__(self):
        import pyarrow.parquet as pq
        import random
        worker_info = torch.utils.data.get_worker_info()
        paths = list(self.parquet_paths)

        # shuffle files so epochs are varied
        random.shuffle(paths)

        if worker_info is not None:
            paths = [p for i, p in enumerate(paths) if i % worker_info.num_workers == worker_info.id]

        buffer = []
        for path in paths:
            try:
                pf = pq.ParquetFile(path)
            except Exception as e:
                print(f"[Data] Error reading {path}: {e}")
                continue

            # Larger batch reads (1024) reduce pyarrow overhead
            for batch in pf.iter_batches(batch_size=1024, columns=["image", "label"]):
                images = batch.column("image")
                labels = batch.column("label")
                for i in range(len(batch)):
                    img_data = images[i].as_py()
                    label_str = labels[i].as_py()

                    if isinstance(img_data, dict) and "bytes" in img_data:
                        img_bytes = img_data["bytes"]
                    elif isinstance(img_data, bytes):
                        img_bytes = img_data
                    else:
                        continue

                    label = 1 if label_str == "fake" else 0
                    img = self._decode_image(img_bytes)

                    if self.transform:
                        img = self.transform(img)

                    buffer.append((img, label))

                    if len(buffer) >= self.shuffle_buffer:
                        random.shuffle(buffer)
                        for item in buffer:
                            yield item
                        buffer.clear()

        if buffer:
            random.shuffle(buffer)
            for item in buffer:
                yield item

def get_row_count(paths):
    import pyarrow.parquet as pq
    total = 0
    for p in paths:
        try:
            total += pq.ParquetFile(p).metadata.num_rows
        except Exception:
            pass
    return total


# --- Build Transforms ---

VIT_MEAN = [0.5, 0.5, 0.5]
VIT_STD = [0.5, 0.5, 0.5]

# Platform configs: (name_in_yaml, max_dim_default, quality_range_default)
PLATFORM_DEFAULTS = {
    "whatsapp_simulation": (1600, [60, 80]),
    "instagram_simulation": (1080, [65, 78]),
    "facebook_simulation": (2048, [71, 85]),
    "telegram_simulation": (2560, [72, 87]),
    "twitter_simulation": (4096, [80, 90]),
}


def build_transforms(cfg):
    input_size = cfg["model"]["input_size"]
    aug_cfg = cfg["data"].get("augmentation", {})

    train_transforms = []

    # Platform-specific compression augmentations (before ToTensor -- operates on PIL)
    for platform_key, (default_dim, default_qr) in PLATFORM_DEFAULTS.items():
        platform_cfg = aug_cfg.get(platform_key, {})
        if platform_cfg.get("enabled", False):
            p = platform_cfg.get("probability", 0.1)
            max_dim = platform_cfg.get("max_dim", default_dim)
            qr = tuple(platform_cfg.get("quality_range", default_qr))
            train_transforms.append(transforms.RandomApply([
                PlatformSimulation(max_dim=max_dim, quality_range=qr)
            ], p=p))

    # General JPEG compression
    if aug_cfg.get("jpeg_compression", {}).get("enabled", False):
        qr = aug_cfg["jpeg_compression"].get("quality_range", [20, 95])
        p = aug_cfg["jpeg_compression"].get("probability", 0.5)
        train_transforms.append(transforms.RandomApply([JPEGCompression(tuple(qr))], p=p))

    # Downscale/upscale
    if aug_cfg.get("downscale_upscale", {}).get("enabled", False):
        ds_cfg = aug_cfg["downscale_upscale"]
        p = ds_cfg.get("probability", 0.15)
        train_transforms.append(transforms.RandomApply([
            DownscaleUpscale(
                min_scale=ds_cfg.get("min_scale", 0.3),
                max_scale=ds_cfg.get("max_scale", 0.7),
            )
        ], p=p))

    # Spatial augmentations
    if aug_cfg.get("random_resized_crop"):
        scale = tuple(aug_cfg["random_resized_crop"]["scale"])
        train_transforms.append(transforms.RandomResizedCrop(input_size, scale=scale))
    else:
        train_transforms.append(transforms.Resize((input_size, input_size)))

    if aug_cfg.get("horizontal_flip", False):
        train_transforms.append(transforms.RandomHorizontalFlip())

    if aug_cfg.get("color_jitter"):
        cj = aug_cfg["color_jitter"]
        train_transforms.append(transforms.ColorJitter(
            brightness=cj.get("brightness", 0),
            contrast=cj.get("contrast", 0),
            saturation=cj.get("saturation", 0),
            hue=cj.get("hue", 0),
        ))

    # Random grayscale
    if aug_cfg.get("random_grayscale", {}).get("enabled", False):
        p = aug_cfg["random_grayscale"].get("probability", 0.05)
        train_transforms.append(transforms.RandomGrayscale(p=p))

    # Gaussian blur
    if aug_cfg.get("gaussian_blur", {}).get("enabled", False):
        gb = aug_cfg["gaussian_blur"]
        p = gb.get("probability", 0.2)
        kernel = gb.get("kernel_size", 5)
        train_transforms.append(transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=kernel)
        ], p=p))

    train_transforms += [
        transforms.ToTensor(),
        transforms.Normalize(mean=VIT_MEAN, std=VIT_STD),
    ]

    if aug_cfg.get("random_erasing", {}).get("enabled", False):
        p = aug_cfg["random_erasing"].get("probability", 0.1)
        train_transforms.append(transforms.RandomErasing(p=p))

    val_transforms = [
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=VIT_MEAN, std=VIT_STD),
    ]

    return transforms.Compose(train_transforms), transforms.Compose(val_transforms)


# --- Build Model ---


def build_model(cfg, resume_path=None, device="cpu"):
    """Load ViT-B/16 and optionally resume from a checkpoint."""
    pretrained_dir = cfg["model"]["pretrained_dir"]
    num_classes = cfg["model"]["num_classes"]
    freeze_blocks = cfg["model"].get("freeze_blocks", 6)

    model_source = pretrained_dir if os.path.isdir(pretrained_dir) else "google/vit-base-patch16-224-in21k"

    print(f"[Model] Loading ViT-B/16 from {model_source}")
    model = ViTForImageClassification.from_pretrained(
        model_source,
        num_labels=num_classes,
        ignore_mismatched_sizes=True,
    )

    # If resuming from existing checkpoint, load weights
    if resume_path and os.path.exists(resume_path):
        print(f"[Model] Loading weights from checkpoint: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict, strict=False)
        prev_auc = checkpoint.get("val_auc", "N/A")
        print(f"[Model] Loaded checkpoint (previous val_auc={prev_auc})")

    # Freeze embeddings
    for param in model.vit.embeddings.parameters():
        param.requires_grad = False

    # Freeze first N encoder blocks
    for i in range(freeze_blocks):
        for param in model.vit.encoder.layer[i].parameters():
            param.requires_grad = False

    total_blocks = len(model.vit.encoder.layer)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    print(f"[Model] Total params: {total_params / 1e6:.1f}M")
    print(f"[Model] Trainable: {trainable_params / 1e6:.1f}M (blocks {freeze_blocks}-{total_blocks - 1} + classifier)")
    print(f"[Model] Frozen: {frozen_params / 1e6:.1f}M (embeddings + blocks 0-{freeze_blocks - 1})")

    return model


# --- Training Loop ---


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, use_amp, max_grad_norm, accum_steps=1):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    num_batches = len(loader)

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device, memory_format=torch.channels_last, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, enabled=use_amp):
            outputs = model(pixel_values=images)
            loss = criterion(outputs.logits, labels) / accum_steps  # scale loss

        scaler.scale(loss).backward()

        # Only step optimizer every accum_steps mini-batches
        if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == num_batches:
            if max_grad_norm > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        running_loss += loss.item() * accum_steps * images.size(0)
        _, predicted = outputs.logits.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

        if (batch_idx + 1) % 100 == 0:
            gpu_mem = torch.cuda.memory_allocated() / 1e9 if device.type == 'cuda' else 0
            print(f"  [batch {batch_idx + 1}/{num_batches}] loss={running_loss / total:.4f} acc={correct / total:.4f} gpu_mem={gpu_mem:.2f}GB")
            sys.stdout.flush()

    return running_loss / total, correct / total


@torch.no_grad()
def validate(model, loader, criterion, device, use_amp):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_probs = []
    all_preds = []

    for images, labels in loader:
        images = images.to(device, memory_format=torch.channels_last, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            outputs = model(pixel_values=images)
            loss = criterion(outputs.logits, labels)

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.logits.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

        probs = torch.softmax(outputs.logits, dim=1)[:, 1]
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
        all_preds.extend(predicted.cpu().numpy())

    auc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.0
    return running_loss / total, correct / total, auc, all_labels, all_preds


# --- Logging Setup ---


def setup_logging():
    """Tee all output to both console and a timestamped log file."""
    log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"training_{timestamp}.log")

    # Create a logger that writes to both file and console
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    console_handler = logging.StreamHandler(sys.stdout)
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[file_handler, console_handler],
    )

    # Redirect print() to logger
    class LoggerWriter:
        def __init__(self, logger_func):
            self.logger_func = logger_func
            self.buf = ""
        def write(self, msg):
            if msg.strip():
                self.logger_func(msg.rstrip())
        def flush(self):
            pass
        def isatty(self):
            return False

    sys.stdout = LoggerWriter(logging.info)
    sys.stderr = LoggerWriter(logging.error)

    print(f"[Setup] Logging to: {log_path}")
    return log_path


# --- Main ---


def main():
    log_path = setup_logging()
    training_start_time = time.time()

    parser = argparse.ArgumentParser(description="Retrain ViT-B/16 on Modern AI Image Dataset")
    parser.add_argument("--config", type=str, default="configs/train_vit_v2.yaml")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint to initialize weights from")
    parser.add_argument("--max_shards", type=int, default=None,
                        help="Max number of parquet shards to use (for testing). Default: use all.")
    parser.add_argument("--start_shard", type=int, default=None,
                        help="Start shard index (e.g. 96 for train-00096-*.parquet)")
    parser.add_argument("--end_shard", type=int, default=None,
                        help="End shard index (e.g. 116 for train-00116-*.parquet)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] Device: {device}")
    if device.type == "cuda":
        print(f"[Train] GPU: {torch.cuda.get_device_name(0)}")
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[Train] VRAM: {vram_gb:.1f} GB")

    train_cfg = cfg["training"]
    use_amp = train_cfg.get("use_amp", True) and device.type == "cuda"
    max_grad_norm = train_cfg.get("gradient_clip", 1.0)

    # VRAM guard for batch size (with AMP fp16, 4GB can handle batch=32 for ViT-B/16)
    if device.type == "cuda":
        max_batch_for_vram = 32 if (use_amp and vram_gb >= 3.5) else (16 if vram_gb >= 3.5 else 8)
        if train_cfg["batch_size"] > max_batch_for_vram:
            print(f"[Warn] VRAM is {vram_gb:.1f}GB. Reducing batch size from {train_cfg['batch_size']} to {max_batch_for_vram}.")
            old_batch = train_cfg["batch_size"]
            train_cfg["gradient_accumulation_steps"] = max(1, train_cfg.get("gradient_accumulation_steps", 1) * (old_batch // max_batch_for_vram))
            train_cfg["batch_size"] = max_batch_for_vram

    # Load parquet data
    import glob
    parquet_pattern = cfg["data"]["parquet_pattern"]
    # Normalize path for the current OS
    parquet_pattern = os.path.normpath(parquet_pattern)
    parquet_paths = sorted(glob.glob(parquet_pattern))
    if not parquet_paths:
        # Try forward slashes too (common in YAML configs)
        parquet_paths = sorted(glob.glob(cfg["data"]["parquet_pattern"]))
    if not parquet_paths:
        print(f"[Error] No parquet files found matching: {parquet_pattern}")
        sys.exit(1)

    # Filter by start/end shard if provided
    if args.start_shard is not None or args.end_shard is not None:
        filtered_paths = []
        for p in parquet_paths:
            # Extract the shard number from filename like 'train-00096-of-00206.parquet'
            try:
                # Get filename
                fname = os.path.basename(p)
                # Split 'train-', get the next part
                num_part = fname.split('-')[1]
                shard_num = int(num_part)
                
                if args.start_shard is not None and shard_num < args.start_shard:
                    continue
                if args.end_shard is not None and shard_num > args.end_shard:
                    continue
                filtered_paths.append(p)
            except Exception:
                # If we can't parse it, just keep it or ignore it. Let's keep it to be safe.
                filtered_paths.append(p)
        parquet_paths = filtered_paths
        print(f"[Data] Filtered down to {len(parquet_paths)} shards based on range {args.start_shard}-{args.end_shard}")
        
        if not parquet_paths:
            print("[Error] No shards left after filtering!")
            sys.exit(1)

    if args.max_shards and args.max_shards < len(parquet_paths):
        parquet_paths = parquet_paths[:args.max_shards]
        print(f"[Data] Limited to {args.max_shards} shards (--max_shards)")

    print(f"[Data] Found {len(parquet_paths)} parquet shard(s)")

    max_samples = cfg["data"].get("max_samples", None)

    # Ensure reproducibility
    import random
    random.seed(42)
    parquet_paths_shuffled = list(parquet_paths)
    random.shuffle(parquet_paths_shuffled)

    n_files = len(parquet_paths_shuffled)
    train_files_end = max(1, int(n_files * cfg["data"]["train_split"]))
    val_files_end = min(n_files, train_files_end + max(1, int(n_files * cfg["data"]["val_split"])))

    train_paths = parquet_paths_shuffled[:train_files_end]
    val_paths = parquet_paths_shuffled[train_files_end:val_files_end]
    test_paths = parquet_paths_shuffled[val_files_end:]

    print(f"[Data] Splitting by files instead of records to avoid RAM exhaustion.")
    print(f"[Data] Train files: {len(train_paths)}, Val files: {len(val_paths)}, Test files: {len(test_paths)}")

    train_samples = get_row_count(train_paths)
    val_samples = get_row_count(val_paths)
    test_samples = get_row_count(test_paths)

    if max_samples:
        train_samples = min(train_samples, max_samples)
        val_samples = min(val_samples, max_samples)
        test_samples = min(test_samples, max_samples)

    print(f"[Data] Total estimated samples:")
    print(f"  Train: {train_samples}")
    print(f"  Val:   {val_samples}")
    print(f"  Test:  {test_samples}")

    # Transforms and dataloaders
    train_transform, val_transform = build_transforms(cfg)
    train_dataset = StreamingParquetDataset(train_paths, train_samples, transform=train_transform, shuffle_buffer=512)
    val_dataset = StreamingParquetDataset(val_paths, val_samples, transform=val_transform, shuffle_buffer=1)

    import multiprocessing
    # Allow config override for num_workers (Windows can handle 2 workers with IterableDataset)
    num_workers = train_cfg.get("num_workers", None)
    if num_workers is None:
        num_workers = min(4, multiprocessing.cpu_count() // 2) if os.name != 'nt' else 0
    prefetch = train_cfg.get("prefetch_factor", 2) if num_workers > 0 else None
    persistent = num_workers > 0  # keep workers alive between epochs

    print(f"[Data] DataLoader: num_workers={num_workers}, prefetch_factor={prefetch}, persistent_workers={persistent}")

    loader_kwargs = dict(
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=persistent,
    )
    if prefetch is not None:
        loader_kwargs["prefetch_factor"] = prefetch

    train_loader = DataLoader(train_dataset, **loader_kwargs)
    val_loader = DataLoader(val_dataset, **loader_kwargs)

    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Model
    model = build_model(cfg, resume_path=args.resume, device=device)
    
    # Performance Optimization: Memory format and Torch Compile (PyTorch 2.0+)
    model = model.to(device, memory_format=torch.channels_last)

    # Enable TF32 on Ampere+ GPUs (RTX 30xx/40xx) for ~3x faster matmuls
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True  # auto-tune convolution algorithms
        print("[Train] Enabled TF32 + cuDNN benchmark for max throughput")

    if hasattr(torch, "compile") and train_cfg.get("torch_compile", False):
        try:
            print("[Train] Compiling model with torch.compile (reduce-overhead)...")
            model = torch.compile(model, mode="reduce-overhead")
            print("[Train] torch.compile succeeded!")
        except Exception as e:
            print(f"[Train] torch.compile failed ({e}), continuing without it.")

    if device.type == "cuda":
        vram_used = torch.cuda.memory_allocated(0) / 1e9
        print(f"[Train] VRAM after model load: {vram_used:.2f} GB")

    # Loss with label smoothing
    label_smoothing = train_cfg.get("label_smoothing", 0.1)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    # Optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg.get("weight_decay", 0.01),
    )

    warmup_epochs = train_cfg.get("warmup_epochs", 2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=train_cfg["epochs"] - warmup_epochs,
    )
    scaler = torch.GradScaler(enabled=use_amp)

    # Output dirs
    checkpoint_dir = cfg["output"]["checkpoint_dir"]
    os.makedirs(checkpoint_dir, exist_ok=True)

    best_auc = 0.0
    patience_counter = 0
    patience = train_cfg.get("early_stopping_patience", 5)
    best_model_path = os.path.join(checkpoint_dir, cfg["output"]["best_model_name"])

    # Count enabled augmentations
    aug_cfg = cfg["data"].get("augmentation", {})
    enabled_augs = []
    for key in ["whatsapp_simulation", "instagram_simulation", "facebook_simulation",
                 "telegram_simulation", "twitter_simulation", "jpeg_compression",
                 "downscale_upscale", "gaussian_blur", "random_erasing", "random_grayscale"]:
        if aug_cfg.get(key, {}).get("enabled", False):
            enabled_augs.append(key.replace("_simulation", "").replace("_", " "))

    accum_steps = train_cfg.get("gradient_accumulation_steps", 4)
    effective_batch = train_cfg['batch_size'] * accum_steps

    print(f"\n[Train] Starting ViT-B/16 V2 training for {train_cfg['epochs']} epochs")
    print(f"[Train] AMP={use_amp}, batch_size={train_cfg['batch_size']}, accum_steps={accum_steps}, effective_batch={effective_batch}")
    print(f"[Train] label_smoothing={label_smoothing}")
    print(f"[Train] LR={train_cfg['learning_rate']}, warmup={warmup_epochs} epochs, grad_clip={max_grad_norm}")
    print(f"[Train] freeze_blocks={cfg['model'].get('freeze_blocks', 6)}")
    print(f"[Train] Augmentations: {', '.join(enabled_augs)}\n")
    sys.stdout.flush()

    for epoch in range(1, train_cfg["epochs"] + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, use_amp, max_grad_norm, accum_steps,
        )
        val_loss, val_acc, val_auc, _, _ = validate(model, val_loader, criterion, device, use_amp)

        if epoch > warmup_epochs:
            scheduler.step()

        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]

        print(
            f"[Epoch {epoch:3d}/{train_cfg['epochs']}] "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_auc={val_auc:.4f} | "
            f"lr={lr:.6f} time={elapsed:.1f}s"
        )
        sys.stdout.flush()

        # Save best model
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_auc": val_auc,
                "val_acc": val_acc,
            }, best_model_path)
            print(f"  -> Saved best model (AUC={val_auc:.4f}) to {best_model_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"[Train] Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
                break

        # Periodic checkpoints disabled to save disk space (~350MB each)

    total_training_time = time.time() - training_start_time
    hours, remainder = divmod(int(total_training_time), 3600)
    minutes, seconds = divmod(remainder, 60)

    print(f"\n{'='*60}")
    print(f"[Train] TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"[Train] Best validation AUC: {best_auc:.4f}")
    print(f"[Train] Best model saved to: {best_model_path}")
    print(f"[Train] Total training time: {hours}h {minutes}m {seconds}s")
    if device.type == "cuda":
        peak_vram = torch.cuda.max_memory_allocated(0) / 1e9
        print(f"[Train] Peak VRAM usage: {peak_vram:.2f} GB")
    print(f"[Train] Log file: {log_path}")
    print(f"{'='*60}")

    # Final test evaluation with detailed metrics
    if test_paths:
        print("\n[Train] Running final evaluation on test set...")
        test_dataset = StreamingParquetDataset(test_paths, test_samples, transform=val_transform, shuffle_buffer=1)
        test_loader = DataLoader(
            test_dataset, batch_size=train_cfg["batch_size"],
            shuffle=False, num_workers=num_workers, pin_memory=(device.type == "cuda"),
        )

        # Load best checkpoint
        checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])

        test_loss, test_acc, test_auc, test_labels, test_preds = validate(
            model, test_loader, criterion, device, use_amp,
        )
        print(f"[Test] loss={test_loss:.4f} acc={test_acc:.4f} auc={test_auc:.4f}")
        print(f"\n[Test] Classification Report:")
        print(classification_report(test_labels, test_preds, target_names=["real", "fake"]))
        print(f"[Test] Confusion Matrix:")
        cm = confusion_matrix(test_labels, test_preds)
        print(f"  Real  predicted as [real={cm[0][0]}, fake={cm[0][1]}]")
        print(f"  Fake  predicted as [real={cm[1][0]}, fake={cm[1][1]}]")


if __name__ == "__main__":
    main()
