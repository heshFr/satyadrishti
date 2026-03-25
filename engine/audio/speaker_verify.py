"""
Speaker Verification using ECAPA-TDNN (SpeechBrain)
=====================================================
Verifies if the person on the call is who they claim to be.

Use case: A scammer calls pretending to be the victim's son.
If the victim has enrolled their son's voice, the system
compares the caller's voice embedding against the stored print
and alerts: "This does NOT sound like [Son's Name]."

Model: ECAPA-TDNN pretrained on VoxCeleb (EER=0.69%)
  - Extracts 192-dim speaker embeddings
  - Cosine similarity for verification
  - Threshold: 0.25 (same speaker) / below = different speaker
"""

import os
import shutil
import json
import logging
import io
import numpy as np
from typing import Optional, Dict, List, Tuple
from pathlib import Path

# Patch os.symlink for Windows to prevent WinError 1314 during model download
if os.name == "nt":
    _orig_symlink = os.symlink
    def _safe_symlink(src, dst, target_is_directory=False, **kwargs):
        try:
            _orig_symlink(src, dst, target_is_directory=target_is_directory, **kwargs)
        except OSError as e:
            if getattr(e, 'winerror', None) == 1314:
                # Fallback: copy file/dir instead of symlinking
                if os.path.isdir(src):
                    shutil.copytree(src, dst, dirs_exist_ok=True)
                else:
                    shutil.copy2(src, dst)
            else:
                raise e
    os.symlink = _safe_symlink

log = logging.getLogger("satyadrishti.speaker_verify")

try:
    import torch
    import torchaudio
    # Patch: SpeechBrain 1.0.x passes deprecated use_auth_token to newer huggingface_hub,
    # and tries to download custom.py which no longer exists in the ECAPA repo.
    import huggingface_hub as _hf_hub
    _orig_hf_download = _hf_hub.hf_hub_download
    def _patched_hf_download(*args, **kwargs):
        kwargs.pop("use_auth_token", None)
        return _orig_hf_download(*args, **kwargs)
    _hf_hub.hf_hub_download = _patched_hf_download

    import speechbrain.utils.fetching as _sb_fetch
    if hasattr(_sb_fetch, "huggingface_hub"):
        _sb_fetch.huggingface_hub.hf_hub_download = _patched_hf_download
    # Patch fetch() so 404 on custom.py raises ValueError (which from_hparams catches)
    _orig_sb_fetch = _sb_fetch.fetch
    def _patched_sb_fetch(filename, *a, **kw):
        try:
            return _orig_sb_fetch(filename, *a, **kw)
        except Exception as exc:
            if "custom.py" in str(filename) and ("404" in str(exc) or "Entry Not Found" in str(exc)):
                raise ValueError(f"{filename} not found in repo") from exc
            raise
    _sb_fetch.fetch = _patched_sb_fetch

    from speechbrain.inference.speaker import EncoderClassifier as SpeakerRecognition
    HAS_SPEECHBRAIN = True
except ImportError:
    HAS_SPEECHBRAIN = False
    log.warning("SpeechBrain not installed. Speaker verification unavailable.")


# Where we store enrolled voice prints
VOICE_PRINTS_DIR = Path("models") / "voice_prints"


class SpeakerVerifier:
    """
    Enrolls and verifies speaker identities using voice embeddings.

    Workflow:
    1. ENROLLMENT: Family member records a 10-second voice sample
       -> Extract embedding -> Save as voice print
    2. VERIFICATION: During a call, extract embeddings from audio chunks
       -> Compare against enrolled prints -> Return match score
    """

    SIMILARITY_THRESHOLD = 0.25  # Cosine similarity threshold for "same speaker"

    def __init__(self, device: str = "auto"):
        self.model = None
        self.enrolled_prints: Dict[str, Dict] = {}  # name -> {embedding, metadata}

        if not HAS_SPEECHBRAIN:
            return

        if device == "auto":
            # Force CPU — ECAPA-TDNN is lightweight and CUDA VRAM is reserved for AST/XLS-R
            device = "cpu"

        try:
            log.info("Loading ECAPA-TDNN speaker verification model...")
            self.model = SpeakerRecognition.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir="models/speaker_verify/ecapa",
                run_opts={"device": device},
            )
            log.info("Speaker verification model loaded.")
        except Exception as e:
            log.error(f"Failed to load speaker verification: {e}")

        # Load any existing voice prints
        self._load_prints()

    @property
    def is_available(self) -> bool:
        return self.model is not None

    def _load_prints(self):
        """Load enrolled voice prints from disk."""
        VOICE_PRINTS_DIR.mkdir(parents=True, exist_ok=True)
        index_path = VOICE_PRINTS_DIR / "index.json"

        if index_path.exists():
            with open(index_path) as f:
                index = json.load(f)

            for name, meta in index.items():
                emb_path = VOICE_PRINTS_DIR / meta["embedding_file"]
                if emb_path.exists():
                    embedding = np.load(emb_path)
                    self.enrolled_prints[name] = {
                        "embedding": torch.from_numpy(embedding),
                        "metadata": meta,
                    }
                    log.info(f"Loaded voice print: {name}")

    def _save_index(self):
        """Save the voice print index to disk."""
        index = {}
        for name, data in self.enrolled_prints.items():
            index[name] = data["metadata"]

        with open(VOICE_PRINTS_DIR / "index.json", "w") as f:
            json.dump(index, f, indent=2)

    def enroll(
        self,
        name: str,
        audio_data: bytes,
        relationship: str = "unknown",
    ) -> Dict:
        """
        Enroll a family member's voice.

        Args:
            name: Display name (e.g., "Rahul", "Maa")
            audio_data: WAV audio bytes (10+ seconds recommended)
            relationship: "son", "daughter", "spouse", "parent", etc.

        Returns:
            dict with enrollment status
        """
        if not self.model:
            return {"status": "error", "message": "Speaker verification not available"}

        try:
            import soundfile as sf

            waveform, sr = sf.read(io.BytesIO(audio_data), dtype="float32")
            if waveform.ndim > 1:
                waveform = waveform.mean(axis=0)

            # Convert to tensor
            waveform_tensor = torch.from_numpy(waveform).unsqueeze(0)

            # Resample to 16kHz if needed
            if sr != 16000:
                resampler = torchaudio.transforms.Resample(sr, 16000)
                waveform_tensor = resampler(waveform_tensor)

            # Extract embedding
            embedding = self.model.encode_batch(waveform_tensor)
            embedding = embedding.squeeze().cpu()

            # Save to disk
            VOICE_PRINTS_DIR.mkdir(parents=True, exist_ok=True)
            emb_filename = f"{name.lower().replace(' ', '_')}_embedding.npy"
            np.save(VOICE_PRINTS_DIR / emb_filename, embedding.numpy())

            # Store in memory
            metadata = {
                "name": name,
                "relationship": relationship,
                "embedding_file": emb_filename,
                "enrolled_at": str(np.datetime64("now")),
                "audio_duration_s": len(waveform) / sr,
            }

            self.enrolled_prints[name] = {
                "embedding": embedding,
                "metadata": metadata,
            }

            self._save_index()

            return {
                "status": "success",
                "name": name,
                "relationship": relationship,
                "audio_duration": f"{len(waveform)/sr:.1f}s",
                "message": f"Voice print for '{name}' enrolled successfully.",
            }

        except Exception as e:
            return {"status": "error", "message": str(e)}

    def verify(self, audio_data: bytes) -> Dict:
        """
        Verify caller identity against enrolled voice prints.

        Returns dict with:
            - best_match: name of closest enrolled person (or None)
            - similarity: cosine similarity score (0-1)
            - is_verified: whether similarity exceeds threshold
            - all_scores: scores against all enrolled prints
        """
        if not self.model or not self.enrolled_prints:
            return {
                "best_match": None,
                "similarity": 0.0,
                "is_verified": False,
                "message": "No voice prints enrolled" if self.model else "Model not available",
            }

        try:
            import soundfile as sf

            waveform, sr = sf.read(io.BytesIO(audio_data), dtype="float32")
            if waveform.ndim > 1:
                waveform = waveform.mean(axis=0)

            waveform_tensor = torch.from_numpy(waveform).unsqueeze(0)
            if sr != 16000:
                resampler = torchaudio.transforms.Resample(sr, 16000)
                waveform_tensor = resampler(waveform_tensor)

            # Extract caller's embedding
            caller_emb = self.model.encode_batch(waveform_tensor).squeeze().cpu()

            # Compare against all enrolled prints
            scores = {}
            for name, data in self.enrolled_prints.items():
                enrolled_emb = data["embedding"]
                # Cosine similarity
                sim = torch.nn.functional.cosine_similarity(
                    caller_emb.unsqueeze(0),
                    enrolled_emb.unsqueeze(0),
                ).item()
                scores[name] = {
                    "similarity": round(sim, 4),
                    "relationship": data["metadata"].get("relationship", "unknown"),
                    "is_match": sim >= self.SIMILARITY_THRESHOLD,
                }

            # Find best match
            if scores:
                best_name = max(scores, key=lambda k: scores[k]["similarity"])
                best = scores[best_name]
                return {
                    "best_match": best_name if best["is_match"] else None,
                    "best_match_name": best_name,
                    "similarity": best["similarity"],
                    "is_verified": best["is_match"],
                    "relationship": best["relationship"],
                    "all_scores": scores,
                }

            return {
                "best_match": None,
                "similarity": 0.0,
                "is_verified": False,
                "all_scores": {},
            }

        except Exception as e:
            return {"best_match": None, "similarity": 0.0, "is_verified": False, "error": str(e)}

    def list_enrolled(self) -> List[Dict]:
        """List all enrolled voice prints."""
        return [
            {
                "name": name,
                "relationship": data["metadata"].get("relationship", "unknown"),
                "enrolled_at": data["metadata"].get("enrolled_at", "unknown"),
                "audio_duration": data["metadata"].get("audio_duration_s", 0),
            }
            for name, data in self.enrolled_prints.items()
        ]

    def remove(self, name: str) -> bool:
        """Remove an enrolled voice print."""
        if name in self.enrolled_prints:
            meta = self.enrolled_prints[name]["metadata"]
            emb_path = VOICE_PRINTS_DIR / meta["embedding_file"]
            if emb_path.exists():
                emb_path.unlink()
            del self.enrolled_prints[name]
            self._save_index()
            return True
        return False
