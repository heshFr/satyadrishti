"""
LFCC Feature Extraction Pipeline
==================================
Extracts Linear Frequency Cepstral Coefficients from raw audio waveforms.
LFCCs capture synthetic phase anomalies (vocoder artifacts) far better
than standard MFCCs because they use a linear frequency scale rather
than the mel scale, preserving high-frequency forensic evidence.
"""

import torch
import torch.nn as nn
import torchaudio
import numpy as np


class LFCCExtractor(nn.Module):
    """
    Extract LFCC features from raw audio waveforms.

    LFCCs use a linearly-spaced filterbank instead of mel-spaced,
    which preserves high-frequency vocoder artifacts critical for
    anti-spoofing detection.

    Args:
        sample_rate: Audio sample rate (default 16kHz for ASVspoof)
        n_filters: Number of linear filterbank channels
        n_lfcc: Number of cepstral coefficients to retain
        n_fft: FFT size
        win_length: Window length in samples
        hop_length: Hop length in samples
        with_delta: Whether to append delta and delta-delta features
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_filters: int = 70,
        n_lfcc: int = 60,
        n_fft: int = 512,
        win_length: int = 400,
        hop_length: int = 160,
        with_delta: bool = True,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_filters = n_filters
        self.n_lfcc = n_lfcc
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.with_delta = with_delta

        # Build linearly-spaced filterbank
        self.register_buffer("filterbank", self._build_linear_filterbank())

        # DCT matrix for cepstral transform
        self.register_buffer("dct_matrix", self._build_dct_matrix())

    def _build_linear_filterbank(self) -> torch.Tensor:
        """Construct a linearly-spaced triangular filterbank."""
        n_freqs = self.n_fft // 2 + 1
        low_freq = 0.0
        high_freq = self.sample_rate / 2.0

        # Linearly spaced center frequencies
        center_freqs = torch.linspace(low_freq, high_freq, self.n_filters + 2)
        freq_bins = torch.linspace(0, high_freq, n_freqs)

        filterbank = torch.zeros(self.n_filters, n_freqs)
        for i in range(self.n_filters):
            f_low = center_freqs[i]
            f_center = center_freqs[i + 1]
            f_high = center_freqs[i + 2]

            # Rising slope
            rising = (freq_bins - f_low) / (f_center - f_low + 1e-8)
            # Falling slope
            falling = (f_high - freq_bins) / (f_high - f_center + 1e-8)

            filterbank[i] = torch.max(torch.zeros_like(freq_bins), torch.min(rising, falling))

        return filterbank

    def _build_dct_matrix(self) -> torch.Tensor:
        """Build DCT-II matrix for cepstral coefficient extraction."""
        n = self.n_filters
        k = self.n_lfcc
        dct = torch.zeros(k, n)
        for i in range(k):
            for j in range(n):
                dct[i, j] = np.cos(np.pi * i * (j + 0.5) / n)
        return dct

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Extract LFCC features from a batch of waveforms.

        Args:
            waveform: (batch, 1, time) or (batch, time) raw audio

        Returns:
            lfcc: (batch, channels, n_lfcc, time_frames)
                  channels = 1 if no deltas, 3 if with_delta
        """
        if waveform.dim() == 2:
            waveform = waveform.unsqueeze(1)

        batch_size = waveform.shape[0]
        waveform = waveform.squeeze(1)  # (batch, time)

        # Compute STFT magnitude
        window = torch.hann_window(self.win_length, device=waveform.device)
        stft = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            return_complex=True,
        )
        power_spectrum = stft.abs().pow(2)  # (batch, freq, time)

        # Apply linear filterbank
        filterbank = self.filterbank.to(waveform.device)
        mel_spec = torch.matmul(filterbank, power_spectrum)  # (batch, n_filters, time)

        # Log compression
        log_spec = torch.log(mel_spec + 1e-8)

        # DCT to get cepstral coefficients
        dct = self.dct_matrix.to(waveform.device)
        lfcc = torch.matmul(dct, log_spec)  # (batch, n_lfcc, time)

        if self.with_delta:
            # Compute delta and delta-delta
            delta = self._compute_delta(lfcc)
            delta_delta = self._compute_delta(delta)
            lfcc = torch.stack([lfcc, delta, delta_delta], dim=1)  # (batch, 3, n_lfcc, time)
        else:
            lfcc = lfcc.unsqueeze(1)  # (batch, 1, n_lfcc, time)

        return lfcc

    @staticmethod
    def _compute_delta(features: torch.Tensor, order: int = 2) -> torch.Tensor:
        """Compute delta features using finite differences."""
        padded = torch.nn.functional.pad(features, (order, order), mode="replicate")
        denominator = 2 * sum(i**2 for i in range(1, order + 1))
        delta = torch.zeros_like(features)
        for i in range(1, order + 1):
            delta += i * (padded[..., order + i :] - padded[..., : -order - i])[..., : features.shape[-1]]
        return delta / denominator
