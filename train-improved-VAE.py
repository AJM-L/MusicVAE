#!/usr/bin/env python3
"""
MusicVAE Training Script

Train a Variational Autoencoder for music spectrogram reconstruction.
Supports training from scratch, resuming from checkpoint, or finetuning
from a pretrained HuggingFace VAE model (transfer learning).
Offers MSE-only loss or full ELBO (reconstruction + KL divergence) loss.

Usage:
    # Train from scratch with ELBO loss (default)
    python train-improved-VAE.py --epochs 100 --lr 1e-4 --beta 0.1

    # Train with MSE-only loss (no KL divergence)
    python train-improved-VAE.py --epochs 100 --loss mse

    # Resume training from checkpoint
    python train-improved-VAE.py --resume checkpoints/music_vae/last.pt --epochs 200

    # Finetune from pretrained HuggingFace VAE (transfer learning)
    python train-improved-VAE.py --finetune --epochs 50 --lr 1e-5

    # Finetune with frozen encoder (only train decoder + adapters)
    python train-improved-VAE.py --finetune --freeze-encoder --epochs 30 --lr 5e-5

    # Finetune with custom pretrained model
    python train-improved-VAE.py --finetune --pretrained-model "stabilityai/sd-vae-ft-mse"

    # Full example with all options
    python train-improved-VAE.py \
        --epochs 30 \
        --lr 5e-5 \
        --beta 0.05 \
        --loss elbo \
        --latent-dim 128 \
        --batch-size 32 \
        --checkpoint-dir checkpoints/my_experiment \
        --output-dir outputs

Finetuning Notes:
    The finetuning mode uses pretrained image VAE models from HuggingFace.
    Since these models expect 3-channel RGB input, adapter layers are used
    to convert 1-channel mel spectrograms to 3-channel format and back.
    
    Default pretrained model: spacepxl/Wan2.1-VAE-upscale2x
    
    Requirements for finetuning:
        pip install diffusers transformers accelerate
"""

import argparse
import json
import math
import random
import warnings
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio
import torchaudio.functional as AF
import torchaudio.transforms as T
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, random_split

# Optional: diffusers for pretrained VAE finetuning
try:
    from diffusers import AutoencoderKL, AutoencoderKLWan
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False
    AutoencoderKL = None
    AutoencoderKLWan = None

warnings.filterwarnings('ignore', category=FutureWarning)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Project data paths
DATA_ROOT = Path('/Users/ajmatheson-lieber/Desktop/Computer Science/ArtificialNeuralNetsAndDeepLearning/MusicVAE/Data/GTZAN-decompressed')
AUDIO_DIR = DATA_ROOT / 'audio'
SPECTROGRAM_DIR = DATA_ROOT / 'spectrogram'

# Device configuration
DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')

# Audio preprocessing
TARGET_SAMPLE_RATE: int = 22_050
SEGMENT_DURATION: float = 3.0  # seconds
SEGMENT_HOP_DURATION: float = 3.0  # seconds
N_FFT: int = 1_024
HOP_LENGTH: int = 256
WIN_LENGTH: int = 1_024
N_MELS: int = 128
FMIN: float = 30.0
FMAX: float = TARGET_SAMPLE_RATE / 2
PAD_MODE: str = 'reflect'
MIN_AMP: float = 1e-5

# Normalization
NORMALIZATION_MODE: str = 'standard'
NORMALIZATION_EPS: float = 1e-6
MAX_STATS_SAMPLES: int = 512

# DataLoader defaults
BATCH_SIZE: int = 32
NUM_WORKERS: int = 0
PIN_MEMORY: bool = False

# Derived parameters
SEGMENT_SAMPLES = int(SEGMENT_DURATION * TARGET_SAMPLE_RATE)
DEFAULT_TIME_FRAMES = int(math.ceil((SEGMENT_DURATION * TARGET_SAMPLE_RATE) / HOP_LENGTH))

# File extensions
AUDIO_EXTENSIONS: Tuple[str, ...] = ('.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aiff', '.aif')
SPECTROGRAM_EXTENSIONS: Tuple[str, ...] = ('.npy', '.npz', '.pt')


@dataclass
class TrainingConfig:
    """Configuration for training the VAE model."""
    epochs: int = 50
    lr: float = 1e-4
    beta: float = 0.1  # KL weight (only used for ELBO loss)
    loss_type: str = 'elbo'  # 'elbo' or 'mse'
    grad_clip: float = 1.0
    val_split: float = 0.12
    seed: int = 42
    batch_size: int = 32
    latent_dim: int = 128
    hidden_channels: Tuple[int, ...] = (64, 128, 256, 512)
    dropout: float = 0.0
    checkpoint_dir: Path = Path('checkpoints/music_vae')
    output_dir: Path = Path('outputs')
    scheduler: str = 'plateau'  # 'plateau', 'cosine', or 'none'
    freq_weighting: bool = True  # Use frequency weighting in loss
    output_reg_weight: float = 0.01  # Output range regularization
    spectral_weight: float = 0.1  # Spectral loss weight


# ============================================================================
# AUDIO UTILITIES
# ============================================================================

def build_mel_transform(sample_rate: int = TARGET_SAMPLE_RATE) -> T.MelSpectrogram:
    """Build mel spectrogram transform."""
    return T.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=N_FFT,
        win_length=WIN_LENGTH,
        hop_length=HOP_LENGTH,
        f_min=FMIN,
        f_max=FMAX,
        pad_mode=PAD_MODE,
        power=2.0,
        norm='slaney',
        n_mels=N_MELS,
        mel_scale='htk',
        window_fn=torch.hann_window,
    ).eval()


def mel_spectrogram_to_audio(
    mel_spectrogram: torch.Tensor,
    sample_rate: int = TARGET_SAMPLE_RATE,
    n_iters: int = 64,
) -> torch.Tensor:
    """Convert mel spectrogram back to audio using Griffin-Lim."""
    if mel_spectrogram.dim() == 3:
        mel_spectrogram = mel_spectrogram.squeeze(0)
    inverse_mel = T.InverseMelScale(
        n_stft=(N_FFT // 2) + 1,
        n_mels=N_MELS,
        sample_rate=sample_rate,
        f_min=FMIN,
        f_max=FMAX,
        norm='slaney',
        mel_scale='htk',
    )
    griffin_lim = T.GriffinLim(
        n_fft=N_FFT,
        n_iter=n_iters,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        power=1.0,
    )
    linear_spec = inverse_mel(mel_spectrogram)
    magnitude_spec = torch.sqrt(linear_spec.clamp_min(MIN_AMP))
    waveform = griffin_lim(magnitude_spec)
    return waveform


def list_files(directory: Path, extensions: Sequence[str]) -> List[Path]:
    """Return all files in directory with matching extensions."""
    if directory is None or not directory.exists():
        return []
    extensions = tuple(e.lower() for e in extensions)
    return sorted(
        path for path in directory.rglob('*') if path.is_file() and path.suffix.lower() in extensions
    )


def is_audio_file(path: Union[str, Path]) -> bool:
    return Path(path).suffix.lower() in AUDIO_EXTENSIONS


def is_spectrogram_file(path: Union[str, Path]) -> bool:
    return Path(path).suffix.lower() in SPECTROGRAM_EXTENSIONS


def pad_or_trim(signal: np.ndarray, target_length: int) -> np.ndarray:
    if signal.shape[-1] < target_length:
        pad_width = target_length - signal.shape[-1]
        signal = np.pad(signal, (0, pad_width), mode='constant')
    elif signal.shape[-1] > target_length:
        signal = signal[:target_length]
    return signal


def load_audio(path: Union[str, Path], target_sample_rate: int = TARGET_SAMPLE_RATE) -> Tuple[np.ndarray, int]:
    """Load audio file with fallback to librosa."""
    path = Path(path)
    waveform: Optional[torch.Tensor] = None
    sample_rate: Optional[int] = None
    
    try:
        waveform, sample_rate = torchaudio.load(str(path))
        if waveform is not None and sample_rate is not None:
            waveform = waveform.to(torch.float32)
    except Exception:
        pass
    
    if waveform is None or sample_rate is None:
        try:
            waveform_np, sample_rate = librosa.load(str(path), sr=None, mono=False)
            if waveform_np.ndim == 1:
                waveform_np = np.expand_dims(waveform_np, axis=0)
            waveform = torch.from_numpy(waveform_np.astype(np.float32))
        except Exception as e:
            raise RuntimeError(f"Failed to load audio file {path}: {e}") from e
    
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    
    if sample_rate != target_sample_rate:
        waveform = AF.resample(waveform, sample_rate, target_sample_rate)
        sample_rate = target_sample_rate
    
    waveform = waveform.squeeze(0).contiguous()
    return waveform.numpy().astype(np.float32), sample_rate


def safe_audio_duration(path: Union[str, Path], max_check_duration: float = 10.0) -> Optional[float]:
    """Safely get audio file duration."""
    path = Path(path)
    
    try:
        with sf.SoundFile(path) as audio_file:
            duration = len(audio_file) / audio_file.samplerate
            if np.isfinite(duration) and duration > 0:
                return float(duration)
    except Exception:
        pass
    
    try:
        duration = librosa.get_duration(path=str(path))
        if np.isfinite(duration) and duration > 0:
            return float(duration)
    except Exception:
        pass
    
    try:
        waveform, sr = librosa.load(path, sr=None, mono=True, duration=max_check_duration)
        if waveform.size == 0 or sr is None:
            return None
        if len(waveform) < max_check_duration * sr:
            return waveform.size / float(sr)
        else:
            file_size = path.stat().st_size
            sample_size = waveform.nbytes
            estimated_duration = (waveform.size / float(sr)) * (file_size / sample_size)
            return min(estimated_duration, 600.0)
    except Exception:
        return None


def power_to_log_db(power_spec: torch.Tensor, amin: float = MIN_AMP) -> torch.Tensor:
    clamped = power_spec.clamp_min(amin)
    ref_value = clamped.max().clamp_min(amin)
    log_spec = 10.0 * torch.log10(clamped) - 10.0 * torch.log10(ref_value)
    return log_spec


def normalize_feature(
    feature: np.ndarray,
    stats: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    apply: bool = True,
) -> np.ndarray:
    if not apply or NORMALIZATION_MODE == 'none':
        return feature.astype(np.float32)

    if stats is not None:
        mean, std = stats
    else:
        mean = feature.mean()
        std = feature.std()

    if NORMALIZATION_MODE == 'standard':
        normalized = (feature - mean) / (std + NORMALIZATION_EPS)
    elif NORMALIZATION_MODE == 'minmax':
        feature_min = feature.min()
        feature_max = feature.max()
        denom = (feature_max - feature_min) if feature_max > feature_min else 1.0
        normalized = (feature - feature_min) / denom
        normalized = normalized * 2.0 - 1.0
    else:
        raise ValueError(f"Unsupported normalization mode: {NORMALIZATION_MODE}")

    return normalized.astype(np.float32)


def load_spectrogram(path: Union[str, Path]) -> np.ndarray:
    path = Path(path)
    if path.suffix.lower() == '.npy':
        spec = np.load(path)
    elif path.suffix.lower() == '.npz':
        spec = np.load(path)['arr_0']
    elif path.suffix.lower() == '.pt':
        spec = torch.load(path, weights_only=True)
        spec = spec.cpu().numpy() if torch.is_tensor(spec) else np.array(spec)
    else:
        raise ValueError(f"Unsupported spectrogram file extension: {path.suffix}")
    spec = np.asarray(spec, dtype=np.float32)
    if spec.ndim == 3 and spec.shape[0] == 1:
        spec = spec[0]
    return spec


# ============================================================================
# DATASET
# ============================================================================

class AudioMelSegmentDataset(Dataset):
    """Dataset that yields log-mel spectrogram segments."""

    def __init__(
        self,
        file_paths: Sequence[Union[str, Path]],
        segment_duration: float = SEGMENT_DURATION,
        segment_hop_duration: float = SEGMENT_HOP_DURATION,
        sample_rate: int = TARGET_SAMPLE_RATE,
        normalization_stats: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        compute_dataset_stats: bool = True,
        max_stats_samples: int = MAX_STATS_SAMPLES,
    ) -> None:
        super().__init__()
        self.file_paths = [Path(p) for p in file_paths]
        self.segment_duration = float(segment_duration)
        self.segment_hop_duration = float(segment_hop_duration)
        self.sample_rate = int(sample_rate)
        self.segment_samples = int(round(self.segment_duration * self.sample_rate))
        self.segment_hop_samples = int(round(self.segment_hop_duration * self.sample_rate))

        self.mel_transform = build_mel_transform(self.sample_rate)
        self._audio_cache: dict = {}
        self._max_cache_size = 2

        self.segment_index = self._build_segment_index()
        if len(self.segment_index) == 0:
            raise ValueError("No valid audio or spectrogram segments found.")

        self.normalization_stats = normalization_stats
        if compute_dataset_stats and normalization_stats is None:
            self.normalization_stats = self._estimate_normalization_stats(max_samples=max_stats_samples)

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state['mel_transform'] = None
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self.mel_transform = build_mel_transform(self.sample_rate)

    def _num_segments_for_duration(self, duration_seconds: float) -> int:
        if duration_seconds < self.segment_duration:
            return 0
        total_samples = int(duration_seconds * self.sample_rate)
        available = total_samples - self.segment_samples
        return (available // self.segment_hop_samples) + 1

    def _build_segment_index(self) -> List[dict]:
        index: List[dict] = []
        for path in self.file_paths:
            if is_audio_file(path):
                duration = safe_audio_duration(path)
                if duration is None or duration < self.segment_duration:
                    continue
                num_segments = self._num_segments_for_duration(duration)
                for segment_idx in range(num_segments):
                    start_time = segment_idx * self.segment_hop_duration
                    index.append({'path': path, 'start_time': start_time, 'source': 'audio'})
            elif is_spectrogram_file(path):
                index.append({'path': path, 'source': 'spectrogram'})
        return index

    def __len__(self) -> int:
        return len(self.segment_index)

    def _load_audio_cached(self, path: Path) -> Tuple[np.ndarray, int]:
        cached = self._audio_cache.get(path)
        if cached is not None:
            return cached
        waveform, sample_rate = load_audio(path, target_sample_rate=self.sample_rate)
        if len(self._audio_cache) >= self._max_cache_size:
            self._audio_cache.clear()
        self._audio_cache[path] = (waveform, sample_rate)
        return waveform, sample_rate

    def _extract_feature(self, meta: dict) -> np.ndarray:
        if meta['source'] == 'audio':
            waveform, _ = self._load_audio_cached(meta['path'])
            start_sample = int(round(meta['start_time'] * self.sample_rate))
            end_sample = start_sample + self.segment_samples
            segment = pad_or_trim(waveform[start_sample:end_sample], self.segment_samples)
            segment = segment.astype(np.float32, copy=False)
            segment_tensor = torch.from_numpy(segment).unsqueeze(0)
            with torch.no_grad():
                mel_spec = self.mel_transform(segment_tensor)
                log_mel = power_to_log_db(mel_spec)
            feature = log_mel.squeeze(0).cpu().numpy()
        else:
            feature = load_spectrogram(meta['path'])
            if feature.ndim == 3:
                feature = feature[0]
            if feature.shape[0] != N_MELS and feature.shape[1] == N_MELS:
                feature = feature.T
        return feature.astype(np.float32)

    def _estimate_normalization_stats(self, max_samples: int = MAX_STATS_SAMPLES) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if NORMALIZATION_MODE == 'none' or len(self.segment_index) == 0:
            return None
        num_samples = min(len(self.segment_index), max_samples)
        sample_indices = np.linspace(0, len(self.segment_index) - 1, num=num_samples, dtype=int)
        stacked = []
        for idx in sample_indices:
            feature = self._extract_feature(self.segment_index[idx])
            stacked.append(feature)
        if not stacked:
            return None
        stacked_array = np.stack(stacked, axis=0)
        mean = stacked_array.mean(axis=(0, 2), keepdims=True)
        std = stacked_array.std(axis=(0, 2), keepdims=True)
        return mean.astype(np.float32), std.astype(np.float32)

    def __getitem__(self, index: int) -> dict:
        meta = self.segment_index[index]
        feature = self._extract_feature(meta)
        normalized = normalize_feature(feature, stats=self.normalization_stats, apply=True)
        
        if normalized.ndim == 2:
            tensor = torch.from_numpy(normalized).unsqueeze(0)
        elif normalized.ndim == 3:
            tensor = torch.from_numpy(normalized)
            if tensor.shape[0] != 1:
                tensor = tensor[:1]
        else:
            raise ValueError(f"Unexpected feature shape: {normalized.shape}")
        
        return {
            'mel': tensor,
            'path': str(meta['path']),
            'source': meta['source'],
        }


# ============================================================================
# MODEL
# ============================================================================

class MusicVAE(nn.Module):
    """Convolutional VAE for music spectrograms."""

    def __init__(
        self,
        input_channels: int = 1,
        input_freq_bins: int = N_MELS,
        input_time_frames: int = DEFAULT_TIME_FRAMES,
        latent_dim: int = 128,
        hidden_channels: Tuple[int, ...] = (32, 64, 128, 256),
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if not hidden_channels:
            raise ValueError("hidden_channels must contain at least one entry")

        self.input_channels = input_channels
        self.input_freq_bins = input_freq_bins
        self.input_time_frames = input_time_frames
        self.latent_dim = latent_dim
        self.hidden_channels = hidden_channels
        self.dropout = dropout

        self.encoder = self._build_encoder()
        down_freq, down_time = self._downsampled_size(input_freq_bins, input_time_frames)
        self.encoded_shape = (hidden_channels[-1], down_freq, down_time)
        encoded_dim = math.prod(self.encoded_shape)

        self.fc_mu = nn.Linear(encoded_dim, latent_dim)
        self.fc_logvar = nn.Linear(encoded_dim, latent_dim)

        self.decoder_input = nn.Linear(latent_dim, encoded_dim)
        self.decoder = self._build_decoder()
        self.output_layer = nn.Sequential(
            nn.ConvTranspose2d(
                hidden_channels[0],
                input_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
        )

    def _build_encoder(self) -> nn.Sequential:
        modules = []
        in_channels = self.input_channels
        for out_channels in self.hidden_channels:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.GELU(),
                    nn.Dropout2d(self.dropout) if self.dropout > 0 else nn.Identity(),
                )
            )
            in_channels = out_channels
        return nn.Sequential(*modules)

    def _build_decoder(self) -> nn.Sequential:
        modules = []
        reversed_channels = self.hidden_channels[::-1]
        for idx in range(len(reversed_channels) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        reversed_channels[idx],
                        reversed_channels[idx + 1],
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        output_padding=1,
                    ),
                    nn.BatchNorm2d(reversed_channels[idx + 1]),
                    nn.GELU(),
                )
            )
        return nn.Sequential(*modules)

    def _downsampled_size(self, freq_bins: int, time_frames: int) -> Tuple[int, int]:
        down_freq, down_time = freq_bins, time_frames
        for _ in self.hidden_channels:
            down_freq = max(1, math.ceil(down_freq / 2))
            down_time = max(1, math.ceil(down_time / 2))
        return down_freq, down_time

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.encoder(x)
        flattened = torch.flatten(features, start_dim=1)
        mu = self.fc_mu(flattened)
        logvar = self.fc_logvar(flattened)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        recon = self.decoder_input(z)
        recon = recon.view(-1, *self.encoded_shape)
        recon = self.decoder(recon)
        recon = self.output_layer(recon)
        return self._match_input_shape(recon)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)
        return reconstruction, mu, logvar

    def _match_input_shape(self, tensor: torch.Tensor) -> torch.Tensor:
        freq_diff = self.input_freq_bins - tensor.size(-2)
        time_diff = self.input_time_frames - tensor.size(-1)

        if freq_diff > 0 or time_diff > 0:
            pad_freq = max(freq_diff, 0)
            pad_time = max(time_diff, 0)
            tensor = F.pad(tensor, (0, pad_time, 0, pad_freq))

        if tensor.size(-2) > self.input_freq_bins:
            tensor = tensor[..., : self.input_freq_bins, :]
        if tensor.size(-1) > self.input_time_frames:
            tensor = tensor[..., :, : self.input_time_frames]
        return tensor

    def get_latent_dim(self) -> int:
        return self.latent_dim

    def get_input_shape(self) -> Tuple[int, int, int]:
        return self.input_channels, self.input_freq_bins, self.input_time_frames


# ============================================================================
# PRETRAINED VAE WRAPPER (for finetuning from HuggingFace models)
# ============================================================================

class PretrainedVAEWrapper(nn.Module):
    """
    Wrapper for pretrained HuggingFace VAE models.
    
    Supports both standard AutoencoderKL and the Wan2.1 VAE (AutoencoderKLWan).
    Adapts 1-channel mel spectrograms to 3-channel format expected by
    image-based VAEs, and converts back to 1-channel for output.
    
    The pretrained VAE uses a different latent space representation than
    our MusicVAE, so we add projection layers to get mu/logvar.
    
    For Wan2.1-VAE-upscale2x specifically:
    - Uses AutoencoderKLWan with video format (BCFHW)
    - Decoder outputs 12 channels that need pixel_shuffle for 2x upscaling
    - See: https://huggingface.co/spacepxl/Wan2.1-VAE-upscale2x
    """
    
    DEFAULT_MODEL = "spacepxl/Wan2.1-VAE-upscale2x"
    WAN_SUBFOLDER = "diffusers/Wan2.1_VAE_upscale2x_imageonly_real_v1"
    
    def __init__(
        self,
        pretrained_model: str = DEFAULT_MODEL,
        input_freq_bins: int = N_MELS,
        input_time_frames: int = DEFAULT_TIME_FRAMES,
        latent_dim: int = 128,
        freeze_encoder: bool = False,
        freeze_decoder: bool = False,
        device: torch.device = DEVICE,
    ) -> None:
        super().__init__()
        
        if not DIFFUSERS_AVAILABLE:
            raise ImportError(
                "diffusers library is required for finetuning pretrained models. "
                "Install with: pip install diffusers"
            )
        
        self.input_channels = 1
        self.input_freq_bins = input_freq_bins
        self.input_time_frames = input_time_frames
        self.latent_dim = latent_dim
        self.pretrained_model_name = pretrained_model
        self.device = device
        
        # Detect if this is a Wan VAE model
        self.is_wan_vae = "wan" in pretrained_model.lower() or "Wan" in pretrained_model
        
        # Load pretrained VAE
        print(f"   Loading pretrained VAE: {pretrained_model}")
        
        # Use float32 for MPS compatibility (bfloat16 not well supported), bfloat16 for CUDA
        # Note: MPS memory is limited, so we'll use float32 which is more memory efficient on Apple Silicon
        dtype = torch.bfloat16 if device.type == 'cuda' else torch.float32
        self.dtype = dtype
        
        if self.is_wan_vae:
            # Wan VAE uses AutoencoderKLWan and has a specific subfolder
            print(f"   Detected Wan VAE - using AutoencoderKLWan")
            if AutoencoderKLWan is None:
                raise ImportError(
                    "AutoencoderKLWan not available. Update diffusers: pip install -U diffusers"
                )
            self.vae = AutoencoderKLWan.from_pretrained(
                pretrained_model,
                subfolder=self.WAN_SUBFOLDER,
                torch_dtype=dtype
            )
            # Wan VAE upscales 2x, so we need to track this
            self.upscale_factor = 2
            self.decoder_out_channels = 12  # outputs 12 channels for pixel shuffle
        else:
            # Standard AutoencoderKL
            self.vae = AutoencoderKL.from_pretrained(pretrained_model, torch_dtype=dtype)
            self.upscale_factor = 1
            self.decoder_out_channels = 3
        
        self.vae.to(device)
        
        # Optionally freeze pretrained weights
        if freeze_encoder:
            for param in self.vae.encoder.parameters():
                param.requires_grad = False
            print("   ✓ Encoder weights frozen")
        
        if freeze_decoder:
            for param in self.vae.decoder.parameters():
                param.requires_grad = False
            print("   ✓ Decoder weights frozen")
        
        # Channel adapter: 1 channel -> 3 channels (input)
        self.input_adapter = nn.Conv2d(1, 3, kernel_size=1, bias=True)
        nn.init.xavier_uniform_(self.input_adapter.weight)
        nn.init.zeros_(self.input_adapter.bias)
        
        # Channel adapter: 3 channels -> 1 channel (output)
        # For Wan VAE, this comes after pixel_shuffle which gives 3 channels
        self.output_adapter = nn.Conv2d(3, 1, kernel_size=1, bias=True)
        nn.init.xavier_uniform_(self.output_adapter.weight)
        nn.init.zeros_(self.output_adapter.bias)
        
        # The pretrained VAE latent is typically 16 channels for Wan, 4 for standard
        # Calculate latent spatial dimensions (VAE typically downscales by 8x)
        self.latent_h = max(1, input_freq_bins // 8)
        self.latent_w = max(1, input_time_frames // 8)
        
        # Get latent channels - different VAEs use different config attribute names
        vae_config = self.vae.config
        if hasattr(vae_config, 'latent_channels'):
            vae_latent_channels = vae_config.latent_channels
        elif hasattr(vae_config, 'z_channels'):
            vae_latent_channels = vae_config.z_channels
        elif self.is_wan_vae:
            # Wan VAE typically uses 16 latent channels
            vae_latent_channels = 16
        else:
            # Default for standard VAEs
            vae_latent_channels = 4
        
        self.vae_latent_channels = vae_latent_channels
        self.vae_latent_dim = vae_latent_channels * self.latent_h * self.latent_w
        
        # Projection layers for mu and logvar (to match our latent_dim)
        self.fc_mu = nn.Linear(self.vae_latent_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.vae_latent_dim, latent_dim)
        
        # Projection back from our latent_dim to VAE latent space
        self.fc_decode = nn.Linear(latent_dim, self.vae_latent_dim)
        
        # Store encoded shape info for decode
        self.encoded_shape = (vae_latent_channels, self.latent_h, self.latent_w)
        
        # Move adapters to device
        self.input_adapter.to(device)
        self.output_adapter.to(device)
        self.fc_mu.to(device)
        self.fc_logvar.to(device)
        self.fc_decode.to(device)
        
        print(f"   ✓ VAE loaded (latent channels: {vae_latent_channels}, upscale: {self.upscale_factor}x)")
    
    def _adapt_input(self, x: torch.Tensor) -> torch.Tensor:
        """Convert 1-channel mel spectrogram to 3-channel format."""
        # x: (B, 1, H, W) -> (B, 3, H, W)
        return self.input_adapter(x)
    
    def _adapt_output(self, x: torch.Tensor) -> torch.Tensor:
        """Convert 3-channel output back to 1-channel."""
        # x: (B, 3, H, W) -> (B, 1, H, W)
        return self.output_adapter(x)
    
    def _match_input_shape(self, tensor: torch.Tensor) -> torch.Tensor:
        """Ensure output matches input spatial dimensions."""
        freq_diff = self.input_freq_bins - tensor.size(-2)
        time_diff = self.input_time_frames - tensor.size(-1)

        if freq_diff > 0 or time_diff > 0:
            pad_freq = max(freq_diff, 0)
            pad_time = max(time_diff, 0)
            tensor = F.pad(tensor, (0, pad_time, 0, pad_freq))

        if tensor.size(-2) > self.input_freq_bins:
            tensor = tensor[..., :self.input_freq_bins, :]
        if tensor.size(-1) > self.input_time_frames:
            tensor = tensor[..., :, :self.input_time_frames]
        return tensor
    
    def _to_video_format(self, x: torch.Tensor) -> torch.Tensor:
        """Convert (B, C, H, W) to video format (B, C, F, H, W) with F=1."""
        return x.unsqueeze(2)  # Add frame dimension
    
    def _from_video_format(self, x: torch.Tensor) -> torch.Tensor:
        """Convert video format (B, C, F, H, W) to (B, C, H, W) by squeezing frame dim."""
        return x.squeeze(2)
    
    def _process_wan_decoder_output(self, decoder_out: torch.Tensor) -> torch.Tensor:
        """
        Process Wan VAE decoder output using pixel shuffle.
        
        Input: (B, 12, F, H, W) - 12 channels for pixel shuffle
        Output: (B, 3, H*2, W*2) - 3 channels at 2x resolution
        """
        # decoder_out shape: (B, 12, F, H, W)
        # For images, F=1, so we can squeeze it
        # pixel_shuffle needs format [..., C, H, W]
        
        # Move frame dim for pixel shuffle: (B, 12, F, H, W) -> (B, F, 12, H, W)
        x = decoder_out.movedim(2, 1)
        
        # Apply pixel shuffle: (B, F, 12, H, W) -> (B, F, 3, H*2, W*2)
        b, f, c, h, w = x.shape
        x = x.reshape(b * f, c, h, w)  # Flatten batch and frame
        x = F.pixel_shuffle(x, upscale_factor=self.upscale_factor)  # (B*F, 3, H*2, W*2)
        x = x.reshape(b, f, 3, h * self.upscale_factor, w * self.upscale_factor)
        
        # Move back: (B, F, 3, H*2, W*2) -> (B, 3, F, H*2, W*2)
        x = x.movedim(1, 2)
        
        # Squeeze frame dimension for images: (B, 3, 1, H*2, W*2) -> (B, 3, H*2, W*2)
        return self._from_video_format(x)
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input to latent space, returning mu and logvar."""
        # Adapt channels: (B, 1, H, W) -> (B, 3, H, W)
        x_3ch = self._adapt_input(x)
        
        if self.is_wan_vae:
            # Wan VAE expects video format: (B, C, F, H, W)
            x_video = self._to_video_format(x_3ch)
            posterior = self.vae.encode(x_video).latent_dist
            z = posterior.sample()  # (B, latent_channels, F, H', W')
            z = self._from_video_format(z)  # (B, latent_channels, H', W')
        else:
            # Standard VAE: (B, C, H, W)
            posterior = self.vae.encode(x_3ch).latent_dist
            z = posterior.sample()  # (B, latent_channels, H', W')
        
        # Flatten and project to our latent space
        z_flat = z.reshape(z.size(0), -1)  # (B, vae_latent_dim)
        mu = self.fc_mu(z_flat)
        logvar = self.fc_logvar(z_flat)
        
        return mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick for sampling."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector to spectrogram."""
        # Project back to VAE latent space
        z_vae = self.fc_decode(z)
        z_vae = z_vae.view(-1, *self.encoded_shape)  # (B, latent_channels, H', W')
        
        if self.is_wan_vae:
            # Wan VAE expects video format for decode
            z_video = self._to_video_format(z_vae)  # (B, C, 1, H', W')
            decoder_out = self.vae.decode(z_video, return_dict=False)[0]  # (B, 12, 1, H, W)
            decoded = self._process_wan_decoder_output(decoder_out)  # (B, 3, H*2, W*2)
            
            # Downsample back to original resolution since we just want the quality improvement
            decoded = F.interpolate(
                decoded,
                size=(self.input_freq_bins, self.input_time_frames),
                mode='bilinear',
                align_corners=False
            )
        else:
            # Standard VAE decode
            decoded = self.vae.decode(z_vae).sample  # (B, 3, H, W)
        
        # Adapt back to 1 channel
        output = self._adapt_output(decoded)
        
        # Match input dimensions
        output = self._match_input_shape(output)
        
        return output
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass returning reconstruction, mu, logvar."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)
        return reconstruction, mu, logvar
    
    def get_latent_dim(self) -> int:
        return self.latent_dim
    
    def get_input_shape(self) -> Tuple[int, int, int]:
        return self.input_channels, self.input_freq_bins, self.input_time_frames
    
    def get_trainable_params(self) -> List[nn.Parameter]:
        """Get list of trainable parameters (useful for optimizer setup)."""
        params = []
        # Always trainable: adapters and projection layers
        params.extend(self.input_adapter.parameters())
        params.extend(self.output_adapter.parameters())
        params.extend(self.fc_mu.parameters())
        params.extend(self.fc_logvar.parameters())
        params.extend(self.fc_decode.parameters())
        # Conditionally trainable: VAE weights
        params.extend([p for p in self.vae.parameters() if p.requires_grad])
        return params


# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

def create_frequency_weights(n_mels: int = N_MELS, device: torch.device = DEVICE) -> torch.Tensor:
    """Create frequency weighting to emphasize important frequencies."""
    weights = torch.ones(n_mels)
    
    # Emphasize low frequencies (fundamentals) - first 20% of bins
    low_end = int(n_mels * 0.2)
    weights[:low_end] = 1.5
    
    # Emphasize mid-high frequencies (harmonics) - 30-70% of bins
    mid_start = int(n_mels * 0.3)
    mid_end = int(n_mels * 0.7)
    weights[mid_start:mid_end] = 2.0
    
    # Slightly reduce very high frequencies (often noise)
    high_start = int(n_mels * 0.8)
    weights[high_start:] = 0.8
    
    return weights.to(device)


def mse_loss(
    recon_x: torch.Tensor,
    x: torch.Tensor,
    freq_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Pure MSE reconstruction loss (no KL divergence).
    
    Args:
        recon_x: Reconstructed spectrogram
        x: Original spectrogram
        freq_weight: Optional frequency weighting tensor
    
    Returns:
        MSE loss value
    """
    mse = F.mse_loss(recon_x, x, reduction='none')
    
    if freq_weight is not None:
        if freq_weight.dim() == 1:
            freq_weight = freq_weight.view(1, 1, -1, 1)
        mse = mse * freq_weight
    
    return mse.mean()


def elbo_loss(
    recon_x: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1.0,
    freq_weight: Optional[torch.Tensor] = None,
    output_reg_weight: float = 0.01,
    spectral_weight: float = 0.1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Full ELBO-style loss: reconstruction + KL divergence + regularizers.

    Returns:
        total_loss:        reconstruction + regularization + beta * KL
        total_recon_loss:  reconstruction + regularization (no KL)
        kl_loss:           KL divergence term
    """
    # Base MSE reconstruction loss with optional frequency weighting
    mse = F.mse_loss(recon_x, x, reduction="none")

    if freq_weight is not None:
        if freq_weight.dim() == 1:
            # assume (B, C, F, T) spectrogram
            freq_weight = freq_weight.view(1, 1, -1, 1).to(mse.device)
        else:
            freq_weight = freq_weight.to(mse.device)

        weighted_mse = mse * freq_weight
        # normalized weighted MSE
        recon_loss = weighted_mse.sum() / freq_weight.sum()
    else:
        recon_loss = mse.mean()

    # "Spectral" loss – currently just unweighted MSE in the same domain
    spectral_loss = F.mse_loss(recon_x, x, reduction="mean")

    # Output range regularization (margin-based, differentiable)
    input_min, input_max = x.amin(), x.amax()
    input_range = (input_max - input_min).clamp_min(1e-8)

    output_min, output_max = recon_x.amin(), recon_x.amax()

    lower_margin = input_min - 0.1 * input_range
    upper_margin = input_max + 0.1 * input_range

    lower_violation = F.relu(lower_margin - output_min)
    upper_violation = F.relu(output_max - upper_margin)

    range_penalty = lower_violation + upper_violation

    total_recon_loss = (
        recon_loss
        + spectral_weight * spectral_loss
        + output_reg_weight * range_penalty
    )

    # KL divergence: D_KL(q(z|x) || p(z)) for diagonal Gaussians
    kl_loss = -0.5 * torch.sum(
        1 + logvar - mu.pow(2) - logvar.exp(), dim=1
    ).mean()

    total_loss = total_recon_loss + beta * kl_loss

    return total_loss, total_recon_loss, kl_loss


# ============================================================================
# TRAINING UTILITIES
# ============================================================================

def fix_input_shape(inputs: torch.Tensor) -> torch.Tensor:
    """Ensure input is 4D: (batch, channels, freq, time)."""
    if inputs.dim() == 5:
        if inputs.shape[1] == 1 and inputs.shape[2] == 1:
            inputs = inputs.squeeze(1)
    elif inputs.dim() == 3:
        inputs = inputs.unsqueeze(1)
    elif inputs.dim() == 2:
        inputs = inputs.unsqueeze(0).unsqueeze(0)
    assert inputs.dim() == 4, f"Expected 4D tensor, got {inputs.dim()}D"
    return inputs


def save_checkpoint(
    model: MusicVAE,
    optimizer: optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    config: TrainingConfig,
    path: Path,
) -> None:
    """Save model checkpoint with full training state."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'metrics': metrics,
        'config': asdict(config),
    }, path)


def load_checkpoint(checkpoint_path: Path, device: torch.device = DEVICE) -> dict:
    """Load a checkpoint file."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    return torch.load(checkpoint_path, map_location=device, weights_only=False)


def gather_data_paths(
    audio_dir: Optional[Path] = AUDIO_DIR,
    spectrogram_dir: Optional[Path] = SPECTROGRAM_DIR,
) -> List[Path]:
    """Collect audio and spectrogram files for dataset construction."""
    file_paths: List[Path] = []
    if audio_dir is not None:
        file_paths.extend(list_files(audio_dir, AUDIO_EXTENSIONS))
    if spectrogram_dir is not None:
        file_paths.extend(list_files(spectrogram_dir, SPECTROGRAM_EXTENSIONS))
    return file_paths


def prepare_dataloaders(
    config: TrainingConfig,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """Prepare train and validation dataloaders."""
    file_paths = gather_data_paths()
    if not file_paths:
        raise RuntimeError('No audio or spectrogram files found.')

    random.seed(config.seed)
    file_paths = random.sample(file_paths, k=len(file_paths))
    dataset = AudioMelSegmentDataset(file_paths=file_paths)

    if config.val_split <= 0 or config.val_split >= 1:
        return DataLoader(
        dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY,
        ), None

    val_size = max(1, int(len(dataset) * config.val_split))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )
    return train_loader, val_loader


def train_epoch(
    model: MusicVAE,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    config: TrainingConfig,
    freq_weights: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    total_loss = total_recon = total_kl = 0.0
    num_batches = 0
    
    for batch in loader:
        inputs = fix_input_shape(batch['mel'].to(DEVICE))
        optimizer.zero_grad()
        recon, mu, logvar = model(inputs)
        
        if config.loss_type == 'mse':
            loss = mse_loss(recon, inputs, freq_weight=freq_weights if config.freq_weighting else None)
            recon_loss = loss
            kl_loss = torch.tensor(0.0)
        else:  # elbo
            loss, recon_loss, kl_loss = elbo_loss(
                recon, inputs, mu, logvar,
                beta=config.beta,
                freq_weight=freq_weights if config.freq_weighting else None,
                output_reg_weight=config.output_reg_weight,
                spectral_weight=config.spectral_weight,
            )
        
        loss.backward()
        if config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()

        total_loss += loss.item()
        total_recon += recon_loss.item()
        total_kl += kl_loss.item() if isinstance(kl_loss, torch.Tensor) else kl_loss
        num_batches += 1

    return {
        'loss': total_loss / num_batches,
        'recon': total_recon / num_batches,
        'kl': total_kl / num_batches,
    }


def evaluate_epoch(
    model: MusicVAE,
    loader: DataLoader,
    config: TrainingConfig,
    freq_weights: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """Evaluate for one epoch."""
    model.eval()
    total_loss = total_recon = total_kl = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in loader:
            inputs = fix_input_shape(batch['mel'].to(DEVICE))
            recon, mu, logvar = model(inputs)
            
            if config.loss_type == 'mse':
                loss = mse_loss(recon, inputs, freq_weight=freq_weights if config.freq_weighting else None)
                recon_loss = loss
                kl_loss = torch.tensor(0.0)
            else:  # elbo
                loss, recon_loss, kl_loss = elbo_loss(
                    recon, inputs, mu, logvar,
                    beta=config.beta,
                    freq_weight=freq_weights if config.freq_weighting else None,
                    output_reg_weight=config.output_reg_weight,
                    spectral_weight=config.spectral_weight,
                )
            
            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kl += kl_loss.item() if isinstance(kl_loss, torch.Tensor) else kl_loss
            num_batches += 1
    
    return {
        'loss': total_loss / num_batches,
        'recon': total_recon / num_batches,
        'kl': total_kl / num_batches,
    }


# ============================================================================
# OUTPUT UTILITIES
# ============================================================================

def create_output_dir(config: TrainingConfig) -> Path:
    """Create output directory for training artifacts."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    loss_str = config.loss_type
    param_str = f"epochs{config.epochs}_lr{config.lr:.0e}_beta{config.beta:.2f}_{loss_str}"
    output_dir = config.output_dir / f"{timestamp}_{param_str}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    config_dict = asdict(config)
    config_dict['checkpoint_dir'] = str(config.checkpoint_dir)
    config_dict['output_dir'] = str(config.output_dir)
    config_dict['timestamp'] = timestamp
    
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    return output_dir


def save_training_curves(history: dict, output_dir: Path, config: TrainingConfig) -> None:
    """Save training and validation loss curves."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    epochs = range(1, len(history['train']) + 1)
    
    # Total loss
    axes[0].plot(epochs, [h['loss'] for h in history['train']], 'b-', label='Train', linewidth=2)
    if history.get('val'):
        axes[0].plot(epochs, [h['loss'] for h in history['val']], 'r-', label='Val', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Total Loss')
    axes[0].set_title('Total Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Reconstruction loss
    axes[1].plot(epochs, [h['recon'] for h in history['train']], 'b-', label='Train', linewidth=2)
    if history.get('val'):
        axes[1].plot(epochs, [h['recon'] for h in history['val']], 'r-', label='Val', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Reconstruction Loss')
    axes[1].set_title('Reconstruction Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # KL divergence
    axes[2].plot(epochs, [h['kl'] for h in history['train']], 'b-', label='Train', linewidth=2)
    if history.get('val'):
        axes[2].plot(epochs, [h['kl'] for h in history['val']], 'r-', label='Val', linewidth=2)
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('KL Divergence')
    axes[2].set_title('KL Divergence')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    loss_type_str = config.loss_type.upper()
    plt.suptitle(
        f'Training Curves ({loss_type_str}, epochs={config.epochs}, lr={config.lr:.0e}, β={config.beta:.2f})',
        fontsize=14, y=1.02
    )
    plt.tight_layout()
    
    save_path = output_dir / 'training_curves.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved training curves to {save_path}")


def save_model_summary(model: MusicVAE, output_dir: Path, config: TrainingConfig) -> None:
    """Save model architecture summary."""
    summary = {
        'input_shape': model.get_input_shape(),
        'latent_dim': model.get_latent_dim(),
        'encoded_shape': model.encoded_shape,
        'hidden_channels': model.hidden_channels,
        'loss_type': config.loss_type,
    }
    
    with open(output_dir / 'model_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    with open(output_dir / 'model_summary.txt', 'w') as f:
        f.write(f"Model Architecture Summary\n")
        f.write(f"{'='*50}\n\n")
        f.write(f"Loss Type: {config.loss_type.upper()}\n")
        f.write(f"Input Shape: {model.get_input_shape()}\n")
        f.write(f"Latent Dimension: {model.get_latent_dim()}\n")
        f.write(f"Encoded Shape: {model.encoded_shape}\n")
        f.write(f"Hidden Channels: {model.hidden_channels}\n")
        f.write(f"\n{str(model)}\n")


# ============================================================================
# MAIN TRAINING FUNCTIONS
# ============================================================================

def train_model(config: TrainingConfig, save_outputs: bool = True) -> dict:
    """
    Train a new VAE model from scratch.
    
    Args:
        config: Training configuration
        save_outputs: Whether to save training outputs
    
    Returns:
        Dictionary with 'model', 'history', and 'output_dir'
    """
    print(f"\n{'='*60}")
    print(f"  MusicVAE Training - New Model")
    print(f"{'='*60}")
    print(f"  Device: {DEVICE}")
    print(f"  Loss Type: {config.loss_type.upper()}")
    print(f"  Epochs: {config.epochs}")
    print(f"  Learning Rate: {config.lr}")
    print(f"  Beta (KL weight): {config.beta}")
    print(f"  Latent Dim: {config.latent_dim}")
    print(f"  Batch Size: {config.batch_size}")
    print(f"  Checkpoint Dir: {config.checkpoint_dir}")
    print(f"{'='*60}\n")
    
    # Prepare data
    print("📦 Preparing data...")
    train_loader, val_loader = prepare_dataloaders(config)
    print(f"   Train batches: {len(train_loader)}")
    if val_loader:
        print(f"   Val batches: {len(val_loader)}")
    
    # Create model
    print("\n🏗️  Building model...")
    model = MusicVAE(
        latent_dim=config.latent_dim,
        hidden_channels=config.hidden_channels,
        dropout=config.dropout,
    ).to(DEVICE)
    
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    
    # Setup scheduler
    scheduler = None
    if config.scheduler == 'plateau' and val_loader:
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    elif config.scheduler == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs)
    
    # Frequency weights
    freq_weights = create_frequency_weights() if config.freq_weighting else None
    
    # Output directory - checkpoints will be saved here alongside summaries
    output_dir = None
    if save_outputs:
        output_dir = create_output_dir(config)
        save_model_summary(model, output_dir, config)
        print(f"\n📁 Output directory: {output_dir}")
        print(f"   Model checkpoints will be saved here alongside training outputs")
    
    # Determine where to save checkpoints
    checkpoint_save_dir = output_dir if output_dir else config.checkpoint_dir
    checkpoint_save_dir.mkdir(parents=True, exist_ok=True)
    
    # Training loop
    history = {'train': [], 'val': []}
    best_val = float('inf')
    
    print(f"\n🚀 Starting training...\n")
    
    for epoch in range(1, config.epochs + 1):
        train_metrics = train_epoch(model, train_loader, optimizer, config, freq_weights)
        history['train'].append(train_metrics)
        
        if val_loader is not None:
            val_metrics = evaluate_epoch(model, val_loader, config, freq_weights)
            history['val'].append(val_metrics)
            current_val = val_metrics['loss']
            
            if scheduler is not None:
                if isinstance(scheduler, ReduceLROnPlateau):
                    scheduler.step(current_val)
                else:
                    scheduler.step()
            
            if current_val < best_val:
                best_val = current_val
                save_checkpoint(
                    model, optimizer, epoch, val_metrics, config,
                    checkpoint_save_dir / 'best.pt'
                )
            
            if config.loss_type == 'mse':
                print(f"Epoch {epoch:03d}: train={train_metrics['loss']:.4f} | val={current_val:.4f}")
            else:
                print(f"Epoch {epoch:03d}: train={train_metrics['loss']:.4f} | val={current_val:.4f} | recon={train_metrics['recon']:.4f} | kl={train_metrics['kl']:.4f}")
        else:
            if scheduler is not None and not isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step()
            
            if config.loss_type == 'mse':
                print(f"Epoch {epoch:03d}: train={train_metrics['loss']:.4f}")
            else:
                print(f"Epoch {epoch:03d}: train={train_metrics['loss']:.4f} | recon={train_metrics['recon']:.4f} | kl={train_metrics['kl']:.4f}")
    
    # Save final checkpoint
    save_checkpoint(
        model, optimizer, config.epochs, history['train'][-1], config,
        checkpoint_save_dir / 'last.pt'
    )
    
    if save_outputs and output_dir:
        save_training_curves(history, output_dir, config)
    
    print(f"\n{'='*60}")
    print(f"  Training Complete!")
    print(f"{'='*60}")
    print(f"  Final train loss: {history['train'][-1]['loss']:.4f}")
    if history['val']:
        print(f"  Best val loss: {best_val:.4f}")
    print(f"  All outputs saved to: {checkpoint_save_dir}")
    print(f"{'='*60}\n")
    
    return {'model': model, 'history': history, 'output_dir': output_dir}


def resume_training(
    checkpoint_path: Path,
    config: TrainingConfig,
    save_outputs: bool = True,
) -> dict:
    """
    Resume training from a checkpoint.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        config: Training configuration (epochs is total target epochs)
        save_outputs: Whether to save training outputs
    
    Returns:
        Dictionary with 'model', 'history', and 'output_dir'
    """
    checkpoint_path = Path(checkpoint_path)
    checkpoint = load_checkpoint(checkpoint_path)
    start_epoch = checkpoint.get('epoch', 0)
    
    print(f"\n{'='*60}")
    print(f"  MusicVAE Training - Resume from Checkpoint")
    print(f"{'='*60}")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Starting epoch: {start_epoch + 1}")
    print(f"  Target epochs: {config.epochs}")
    print(f"  Device: {DEVICE}")
    print(f"  Loss Type: {config.loss_type.upper()}")
    print(f"  Learning Rate: {config.lr}")
    print(f"  Beta (KL weight): {config.beta}")
    print(f"{'='*60}\n")
    
    if start_epoch >= config.epochs:
        print(f"⚠️  Checkpoint already at epoch {start_epoch}, target is {config.epochs}")
        print("   Set --epochs higher to continue training.")
        return {'model': None, 'history': {'train': [], 'val': []}, 'output_dir': None}
    
    # Prepare data
    print("📦 Preparing data...")
    train_loader, val_loader = prepare_dataloaders(config)
    
    # Load model
    print("\n🏗️  Loading model from checkpoint...")
    
    # Get model config from checkpoint if available
    ckpt_config = checkpoint.get('config', {})
    latent_dim = ckpt_config.get('latent_dim', config.latent_dim)
    hidden_channels = ckpt_config.get('hidden_channels', config.hidden_channels)
    if isinstance(hidden_channels, list):
        hidden_channels = tuple(hidden_channels)
    
    model = MusicVAE(
        latent_dim=latent_dim,
        hidden_channels=hidden_channels,
    ).to(DEVICE)
    model.load_state_dict(checkpoint['model_state'])
    
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    if 'optimizer_state' in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state'])
            # Update learning rate if different
            for param_group in optimizer.param_groups:
                param_group['lr'] = config.lr
            print("   ✓ Loaded optimizer state")
        except Exception as e:
            print(f"   ⚠️  Could not load optimizer state: {e}")
    
    # Setup scheduler
    scheduler = None
    remaining_epochs = config.epochs - start_epoch
    if config.scheduler == 'plateau' and val_loader:
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    elif config.scheduler == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=remaining_epochs)
    
    # Frequency weights
    freq_weights = create_frequency_weights() if config.freq_weighting else None
    
    # Output directory - checkpoints will be saved here alongside summaries
    output_dir = None
    if save_outputs:
        output_dir = create_output_dir(config)
        save_model_summary(model, output_dir, config)
        print(f"\n📁 Output directory: {output_dir}")
        print(f"   Model checkpoints will be saved here alongside training outputs")
    
    # Determine where to save checkpoints
    checkpoint_save_dir = output_dir if output_dir else config.checkpoint_dir
    checkpoint_save_dir.mkdir(parents=True, exist_ok=True)
    
    # Training loop
    history = {'train': [], 'val': []}
    best_val = checkpoint.get('metrics', {}).get('loss', float('inf'))
    print(f"   Previous best val loss: {best_val:.4f}")
    
    print(f"\n🚀 Resuming training for {remaining_epochs} more epochs...\n")
    
    for epoch in range(start_epoch + 1, config.epochs + 1):
        train_metrics = train_epoch(model, train_loader, optimizer, config, freq_weights)
        history['train'].append(train_metrics)
        
        if val_loader is not None:
            val_metrics = evaluate_epoch(model, val_loader, config, freq_weights)
            history['val'].append(val_metrics)
            current_val = val_metrics['loss']
            
            if scheduler is not None:
                if isinstance(scheduler, ReduceLROnPlateau):
                    scheduler.step(current_val)
                else:
                    scheduler.step()
            
            if current_val < best_val:
                best_val = current_val
                save_checkpoint(
                    model, optimizer, epoch, val_metrics, config,
                    checkpoint_save_dir / 'best.pt'
                )
            
            if config.loss_type == 'mse':
                print(f"Epoch {epoch:03d}: train={train_metrics['loss']:.4f} | val={current_val:.4f}")
            else:
                print(f"Epoch {epoch:03d}: train={train_metrics['loss']:.4f} | val={current_val:.4f} | recon={train_metrics['recon']:.4f} | kl={train_metrics['kl']:.4f}")
        else:
            if scheduler is not None and not isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step()
            
            if config.loss_type == 'mse':
                print(f"Epoch {epoch:03d}: train={train_metrics['loss']:.4f}")
            else:
                print(f"Epoch {epoch:03d}: train={train_metrics['loss']:.4f} | recon={train_metrics['recon']:.4f} | kl={train_metrics['kl']:.4f}")
    
    # Save final checkpoint
    save_checkpoint(
        model, optimizer, config.epochs, history['train'][-1], config,
        checkpoint_save_dir / 'last.pt'
    )
    
    if save_outputs and output_dir:
        save_training_curves(history, output_dir, config)
    
    print(f"\n{'='*60}")
    print(f"  Training Complete!")
    print(f"{'='*60}")
    print(f"  Final train loss: {history['train'][-1]['loss']:.4f}")
    if history['val']:
        print(f"  Best val loss: {best_val:.4f}")
    print(f"  All outputs saved to: {checkpoint_save_dir}")
    print(f"{'='*60}\n")
    
    return {'model': model, 'history': history, 'output_dir': output_dir}


def finetune_pretrained(
    config: TrainingConfig,
    pretrained_model: str = PretrainedVAEWrapper.DEFAULT_MODEL,
    freeze_encoder: bool = False,
    freeze_decoder: bool = False,
    save_outputs: bool = True,
    gradient_accumulation_steps: int = 1,
) -> dict:
    """
    Finetune a pretrained HuggingFace VAE model on music spectrograms.
    
    This uses transfer learning from a pretrained image VAE, adapting it
    to work with 1-channel mel spectrogram data.
    
    Args:
        config: Training configuration
        pretrained_model: HuggingFace model identifier
        freeze_encoder: Whether to freeze the pretrained encoder weights
        freeze_decoder: Whether to freeze the pretrained decoder weights
        save_outputs: Whether to save training outputs
        gradient_accumulation_steps: Number of steps to accumulate gradients (reduces memory)
    
    Returns:
        Dictionary with 'model', 'history', and 'output_dir'
    """
    if not DIFFUSERS_AVAILABLE:
        raise ImportError(
            "diffusers library is required for finetuning pretrained models. "
            "Install with: pip install diffusers transformers accelerate"
        )
    
    # Auto-adjust batch size for MPS devices with large models
    effective_batch_size = config.batch_size
    if DEVICE.type == 'mps':
        # Reduce batch size for large pretrained models on MPS
        if "wan" in pretrained_model.lower() or "Wan" in pretrained_model:
            # Wan VAE is very large (130M+ params), use smaller batch
            if config.batch_size > 2:
                effective_batch_size = 2
                if gradient_accumulation_steps == 1:
                    gradient_accumulation_steps = max(1, config.batch_size // effective_batch_size)
                print(f"⚠️  MPS device detected with large Wan VAE model (~130M params).")
                print(f"   Reducing batch size from {config.batch_size} to {effective_batch_size}")
                print(f"   Using gradient accumulation: {gradient_accumulation_steps} steps")
                if not freeze_encoder:
                    print(f"   💡 Tip: Use --freeze-encoder to reduce memory usage further")
        elif config.batch_size > 8:
            effective_batch_size = 8
            if gradient_accumulation_steps == 1:
                gradient_accumulation_steps = max(1, config.batch_size // effective_batch_size)
            print(f"⚠️  MPS device detected. Reducing batch size from {config.batch_size} to {effective_batch_size}")
    
    print(f"\n{'='*60}")
    print(f"  MusicVAE Finetuning - Pretrained Model")
    print(f"{'='*60}")
    print(f"  Device: {DEVICE}")
    print(f"  Pretrained Model: {pretrained_model}")
    print(f"  Freeze Encoder: {freeze_encoder}")
    print(f"  Freeze Decoder: {freeze_decoder}")
    print(f"  Loss Type: {config.loss_type.upper()}")
    print(f"  Epochs: {config.epochs}")
    print(f"  Learning Rate: {config.lr}")
    print(f"  Beta (KL weight): {config.beta}")
    print(f"  Latent Dim: {config.latent_dim}")
    print(f"  Batch Size: {effective_batch_size} (gradient accumulation: {gradient_accumulation_steps})")
    print(f"{'='*60}\n")
    
    # Update batch size in config for dataloaders
    original_batch_size = config.batch_size
    config.batch_size = effective_batch_size
    
    # Prepare data
    print("📦 Preparing data...")
    train_loader, val_loader = prepare_dataloaders(config)
    print(f"   Train batches: {len(train_loader)}")
    if val_loader:
        print(f"   Val batches: {len(val_loader)}")
    
    # Create model
    print("\n🏗️  Loading pretrained model for finetuning...")
    model = PretrainedVAEWrapper(
        pretrained_model=pretrained_model,
        input_freq_bins=N_MELS,
        input_time_frames=DEFAULT_TIME_FRAMES,
        latent_dim=config.latent_dim,
        freeze_encoder=freeze_encoder,
        freeze_decoder=freeze_decoder,
        device=DEVICE,
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Frozen parameters: {total_params - trainable_params:,}")
    
    # Setup optimizer with only trainable parameters
    trainable_params_list = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable_params_list, lr=config.lr)
    
    # Setup scheduler
    scheduler = None
    if config.scheduler == 'plateau' and val_loader:
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    elif config.scheduler == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs)
    
    # Frequency weights
    freq_weights = create_frequency_weights() if config.freq_weighting else None
    
    # Output directory
    output_dir = None
    if save_outputs:
        # Modify output dir name to indicate finetuning
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        loss_str = config.loss_type
        model_short = pretrained_model.split('/')[-1][:20]  # Truncate long model names
        param_str = f"finetune_{model_short}_epochs{config.epochs}_lr{config.lr:.0e}_beta{config.beta:.2f}_{loss_str}"
        output_dir = config.output_dir / f"{timestamp}_{param_str}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save config with finetuning info
        config_dict = asdict(config)
        config_dict['checkpoint_dir'] = str(config.checkpoint_dir)
        config_dict['output_dir'] = str(config.output_dir)
        config_dict['timestamp'] = timestamp
        config_dict['finetuning'] = {
            'pretrained_model': pretrained_model,
            'freeze_encoder': freeze_encoder,
            'freeze_decoder': freeze_decoder,
            'total_params': total_params,
            'trainable_params': trainable_params,
        }
        
        with open(output_dir / 'config.json', 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        # Save model summary
        summary = {
            'type': 'PretrainedVAEWrapper',
            'pretrained_model': pretrained_model,
            'input_shape': model.get_input_shape(),
            'latent_dim': model.get_latent_dim(),
            'encoded_shape': model.encoded_shape,
            'vae_latent_dim': model.vae_latent_dim,
            'freeze_encoder': freeze_encoder,
            'freeze_decoder': freeze_decoder,
            'loss_type': config.loss_type,
        }
        
        with open(output_dir / 'model_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n📁 Output directory: {output_dir}")
    
    # Determine where to save checkpoints
    checkpoint_save_dir = output_dir if output_dir else config.checkpoint_dir
    checkpoint_save_dir.mkdir(parents=True, exist_ok=True)
    
    # Training loop
    history = {'train': [], 'val': []}
    best_val = float('inf')
    
    print(f"\n🚀 Starting finetuning...\n")
    
    for epoch in range(1, config.epochs + 1):
        train_metrics = _finetune_epoch(
            model, train_loader, optimizer, config, freq_weights,
            gradient_accumulation_steps=gradient_accumulation_steps
        )
        history['train'].append(train_metrics)
        
        # Clear cache after each epoch on MPS
        if DEVICE.type == 'mps':
            torch.mps.empty_cache()
        
        if val_loader is not None:
            val_metrics = _finetune_evaluate(model, val_loader, config, freq_weights)
            history['val'].append(val_metrics)
            current_val = val_metrics['loss']
            
            if scheduler is not None:
                if isinstance(scheduler, ReduceLROnPlateau):
                    scheduler.step(current_val)
                else:
                    scheduler.step()
            
            if current_val < best_val:
                best_val = current_val
                _save_finetune_checkpoint(
                    model, optimizer, epoch, val_metrics, config,
                    checkpoint_save_dir / 'best.pt',
                    pretrained_model, freeze_encoder, freeze_decoder,
                )
            
            if config.loss_type == 'mse':
                print(f"Epoch {epoch:03d}: train={train_metrics['loss']:.4f} | val={current_val:.4f}")
            else:
                print(f"Epoch {epoch:03d}: train={train_metrics['loss']:.4f} | val={current_val:.4f} | recon={train_metrics['recon']:.4f} | kl={train_metrics['kl']:.4f}")
        else:
            if scheduler is not None and not isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step()
            
            if config.loss_type == 'mse':
                print(f"Epoch {epoch:03d}: train={train_metrics['loss']:.4f}")
            else:
                print(f"Epoch {epoch:03d}: train={train_metrics['loss']:.4f} | recon={train_metrics['recon']:.4f} | kl={train_metrics['kl']:.4f}")
    
    # Save final checkpoint
    _save_finetune_checkpoint(
        model, optimizer, config.epochs, history['train'][-1], config,
        checkpoint_save_dir / 'last.pt',
        pretrained_model, freeze_encoder, freeze_decoder,
    )
    
    if save_outputs and output_dir:
        save_training_curves(history, output_dir, config)
    
    print(f"\n{'='*60}")
    print(f"  Finetuning Complete!")
    print(f"{'='*60}")
    print(f"  Final train loss: {history['train'][-1]['loss']:.4f}")
    if history['val']:
        print(f"  Best val loss: {best_val:.4f}")
    print(f"  All outputs saved to: {checkpoint_save_dir}")
    print(f"{'='*60}\n")
    
    return {'model': model, 'history': history, 'output_dir': output_dir}


def _finetune_epoch(
    model: PretrainedVAEWrapper,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    config: TrainingConfig,
    freq_weights: Optional[torch.Tensor] = None,
    gradient_accumulation_steps: int = 1,
) -> Dict[str, float]:
    """
    Train the finetuned model for one epoch with gradient accumulation support.
    
    Args:
        model: The model to train
        loader: DataLoader for training data
        optimizer: Optimizer
        config: Training configuration
        freq_weights: Optional frequency weighting tensor
        gradient_accumulation_steps: Number of steps to accumulate gradients before updating
    """
    model.train()
    total_loss = total_recon = total_kl = 0.0
    num_batches = 0
    
    optimizer.zero_grad()
    
    for batch_idx, batch in enumerate(loader):
        inputs = fix_input_shape(batch['mel'].to(DEVICE))
        
        # Use float32 for MPS (bfloat16 not well supported), bfloat16 for CUDA
        if DEVICE.type == 'cuda':
            inputs = inputs.to(torch.bfloat16)
        else:
            inputs = inputs.to(torch.float32)
        
        recon, mu, logvar = model(inputs)
        
        # Ensure outputs are float32 for loss computation
        recon = recon.float()
        inputs_float = inputs.float()
        mu = mu.float()
        logvar = logvar.float()
        
        if config.loss_type == 'mse':
            loss = mse_loss(recon, inputs_float, freq_weight=freq_weights if config.freq_weighting else None)
            recon_loss = loss
            kl_loss = torch.tensor(0.0)
        else:  # elbo
            loss, recon_loss, kl_loss = elbo_loss(
                recon, inputs_float, mu, logvar,
                beta=config.beta,
                freq_weight=freq_weights if config.freq_weighting else None,
                output_reg_weight=config.output_reg_weight,
                spectral_weight=config.spectral_weight,
            )
        
        # Scale loss by accumulation steps
        loss = loss / gradient_accumulation_steps
        loss.backward()

        total_loss += loss.item() * gradient_accumulation_steps  # Unscale for reporting
        total_recon += recon_loss.item()
        total_kl += kl_loss.item() if isinstance(kl_loss, torch.Tensor) else kl_loss
        
        # Update weights every gradient_accumulation_steps
        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            if config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    config.grad_clip
                )
            optimizer.step()
            optimizer.zero_grad()
            
            # Clear MPS cache periodically
            if DEVICE.type == 'mps' and (batch_idx + 1) % (gradient_accumulation_steps * 10) == 0:
                torch.mps.empty_cache()
        
        num_batches += 1

    # Handle remaining gradients if not a multiple of accumulation steps
    if num_batches % gradient_accumulation_steps != 0:
        if config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                config.grad_clip
            )
        optimizer.step()
        optimizer.zero_grad()
        if DEVICE.type == 'mps':
            torch.mps.empty_cache()

    return {
        'loss': total_loss / num_batches,
        'recon': total_recon / num_batches,
        'kl': total_kl / num_batches,
    }


def _finetune_evaluate(
    model: PretrainedVAEWrapper,
    loader: DataLoader,
    config: TrainingConfig,
    freq_weights: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """Evaluate the finetuned model for one epoch."""
    model.eval()
    total_loss = total_recon = total_kl = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in loader:
            inputs = fix_input_shape(batch['mel'].to(DEVICE))
            
            # Convert to appropriate dtype for the model
            if DEVICE.type == 'cuda':
                inputs = inputs.to(torch.bfloat16)
            
            recon, mu, logvar = model(inputs)
            
            # Ensure outputs are float32 for loss computation
            recon = recon.float()
            inputs_float = inputs.float()
            mu = mu.float()
            logvar = logvar.float()
            
            if config.loss_type == 'mse':
                loss = mse_loss(recon, inputs_float, freq_weight=freq_weights if config.freq_weighting else None)
                recon_loss = loss
                kl_loss = torch.tensor(0.0)
            else:  # elbo
                loss, recon_loss, kl_loss = elbo_loss(
                    recon, inputs_float, mu, logvar,
                    beta=config.beta,
                    freq_weight=freq_weights if config.freq_weighting else None,
                    output_reg_weight=config.output_reg_weight,
                    spectral_weight=config.spectral_weight,
                )
            
            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kl += kl_loss.item() if isinstance(kl_loss, torch.Tensor) else kl_loss
            num_batches += 1
    
    return {
        'loss': total_loss / num_batches,
        'recon': total_recon / num_batches,
        'kl': total_kl / num_batches,
    }


def _save_finetune_checkpoint(
    model: PretrainedVAEWrapper,
    optimizer: optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    config: TrainingConfig,
    path: Path,
    pretrained_model: str,
    freeze_encoder: bool,
    freeze_decoder: bool,
) -> None:
    """Save finetuned model checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save only the trainable parts and adapters (more efficient)
    # The pretrained VAE can be reloaded from HuggingFace
    save_dict = {
        'epoch': epoch,
        'metrics': metrics,
        'config': asdict(config),
        'finetuning': {
            'pretrained_model': pretrained_model,
            'freeze_encoder': freeze_encoder,
            'freeze_decoder': freeze_decoder,
        },
        # Save adapter and projection weights
        'input_adapter_state': model.input_adapter.state_dict(),
        'output_adapter_state': model.output_adapter.state_dict(),
        'fc_mu_state': model.fc_mu.state_dict(),
        'fc_logvar_state': model.fc_logvar.state_dict(),
        'fc_decode_state': model.fc_decode.state_dict(),
        'optimizer_state': optimizer.state_dict(),
    }
    
    # Optionally save VAE state if it was finetuned
    if not freeze_encoder or not freeze_decoder:
        save_dict['vae_state'] = model.vae.state_dict()
    
    torch.save(save_dict, path)


def load_finetuned_model(
    checkpoint_path: Path,
    device: torch.device = DEVICE,
) -> PretrainedVAEWrapper:
    """
    Load a finetuned model from checkpoint.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        device: Device to load the model to
    
    Returns:
        Loaded PretrainedVAEWrapper model
    """
    if not DIFFUSERS_AVAILABLE:
        raise ImportError(
            "diffusers library is required for loading finetuned models. "
            "Install with: pip install diffusers transformers accelerate"
        )
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Get config
    config = checkpoint.get('config', {})
    finetuning_info = checkpoint.get('finetuning', {})
    
    pretrained_model = finetuning_info.get('pretrained_model', PretrainedVAEWrapper.DEFAULT_MODEL)
    freeze_encoder = finetuning_info.get('freeze_encoder', False)
    freeze_decoder = finetuning_info.get('freeze_decoder', False)
    latent_dim = config.get('latent_dim', 128)
    
    # Create model
    model = PretrainedVAEWrapper(
        pretrained_model=pretrained_model,
        input_freq_bins=N_MELS,
        input_time_frames=DEFAULT_TIME_FRAMES,
        latent_dim=latent_dim,
        freeze_encoder=freeze_encoder,
        freeze_decoder=freeze_decoder,
        device=device,
    )
    
    # Load adapter and projection weights
    model.input_adapter.load_state_dict(checkpoint['input_adapter_state'])
    model.output_adapter.load_state_dict(checkpoint['output_adapter_state'])
    model.fc_mu.load_state_dict(checkpoint['fc_mu_state'])
    model.fc_logvar.load_state_dict(checkpoint['fc_logvar_state'])
    model.fc_decode.load_state_dict(checkpoint['fc_decode_state'])
    
    # Load VAE state if available
    if 'vae_state' in checkpoint:
        model.vae.load_state_dict(checkpoint['vae_state'])
    
    return model


# ============================================================================
# CLI
# ============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Train a MusicVAE model for spectrogram reconstruction.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train from scratch with ELBO loss
  python train-improved-VAE.py --epochs 100 --lr 1e-4 --beta 0.1

  # Train with MSE-only loss (no KL divergence)
  python train-improved-VAE.py --epochs 100 --loss mse

  # Resume training from checkpoint
  python train-improved-VAE.py --resume checkpoints/music_vae/last.pt --epochs 200

  # Finetune from pretrained HuggingFace VAE
  python train-improved-VAE.py --finetune --epochs 50 --lr 1e-5

  # Finetune with frozen encoder (only train decoder + adapters)
  python train-improved-VAE.py --finetune --freeze-encoder --epochs 30 --lr 5e-5

  # Finetune with a different pretrained model
  python train-improved-VAE.py --finetune --pretrained-model "stabilityai/sd-vae-ft-mse" --epochs 50

  # Full configuration
  python train-improved-VAE.py \\
      --epochs 150 \\
      --lr 5e-5 \\
      --beta 0.05 \\
      --loss elbo \\
      --latent-dim 128 \\
      --batch-size 32 \\
      --checkpoint-dir checkpoints/experiment1
        """
    )
    
    # Training mode
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        metavar='PATH',
        help='Path to checkpoint to resume training from'
    )
    
    # Finetuning options
    parser.add_argument(
        '--finetune',
        action='store_true',
        help='Finetune from a pretrained HuggingFace VAE model'
    )
    parser.add_argument(
        '--pretrained-model',
        type=str,
        default='spacepxl/Wan2.1-VAE-upscale2x',
        metavar='MODEL',
        help='HuggingFace model ID for finetuning (default: spacepxl/Wan2.1-VAE-upscale2x)'
    )
    parser.add_argument(
        '--freeze-encoder',
        action='store_true',
        help='Freeze pretrained encoder weights during finetuning'
    )
    parser.add_argument(
        '--freeze-decoder',
        action='store_true',
        help='Freeze pretrained decoder weights during finetuning'
    )
    
    # Loss configuration
    parser.add_argument(
        '--loss',
        type=str,
        choices=['elbo', 'mse'],
        default='elbo',
        help='Loss function: "elbo" (reconstruction + KL) or "mse" (reconstruction only)'
    )
    parser.add_argument(
        '--beta',
        type=float,
        default=0.1,
        help='Beta weight for KL divergence (only used with ELBO loss)'
    )
    
    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--grad-clip', type=float, default=1.0, help='Gradient clipping norm')
    parser.add_argument('--val-split', type=float, default=0.1, help='Validation split ratio')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # Model architecture
    parser.add_argument('--latent-dim', type=int, default=128, help='Latent dimension size')
    parser.add_argument(
        '--hidden-channels',
        type=str,
        default='32,64,128,256',
        help='Comma-separated hidden channel sizes'
    )
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate')
    
    # Loss options
    parser.add_argument('--no-freq-weight', action='store_true', help='Disable frequency weighting')
    parser.add_argument('--output-reg-weight', type=float, default=0.01, help='Output regularization weight')
    parser.add_argument('--spectral-weight', type=float, default=0.1, help='Spectral loss weight')
    
    # Scheduler
    parser.add_argument(
        '--scheduler',
        type=str,
        choices=['plateau', 'cosine', 'none'],
        default='plateau',
        help='Learning rate scheduler type'
    )
    
    # Output
    parser.add_argument(
        '--checkpoint-dir',
        type=str,
        default='checkpoints/music_vae',
        help='Directory to save checkpoints'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='outputs',
        help='Directory to save training outputs'
    )
    parser.add_argument('--no-save', action='store_true', help='Do not save training outputs')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Validate argument combinations
    if args.finetune and args.resume:
        print("⚠️  Warning: --finetune and --resume cannot be used together.")
        print("   Use --finetune for new finetuning from pretrained model.")
        print("   Use --resume to continue training from a checkpoint.")
        return None
    
    if (args.freeze_encoder or args.freeze_decoder) and not args.finetune:
        print("⚠️  Warning: --freeze-encoder and --freeze-decoder only apply to --finetune mode.")
    
    # Check diffusers availability for finetuning
    if args.finetune and not DIFFUSERS_AVAILABLE:
        print("❌ Error: diffusers library is required for finetuning.")
        print("   Install with: pip install diffusers transformers accelerate")
        return None
    
    # Parse hidden channels
    hidden_channels = tuple(int(x) for x in args.hidden_channels.split(','))
    
    # Create config
    config = TrainingConfig(
        epochs=args.epochs,
        lr=args.lr,
        beta=args.beta,
        loss_type=args.loss,
        grad_clip=args.grad_clip,
        val_split=args.val_split,
        seed=args.seed,
        batch_size=args.batch_size,
        latent_dim=args.latent_dim,
        hidden_channels=hidden_channels,
        dropout=args.dropout,
        checkpoint_dir=Path(args.checkpoint_dir),
        output_dir=Path(args.output_dir),
        scheduler=args.scheduler,
        freq_weighting=not args.no_freq_weight,
        output_reg_weight=args.output_reg_weight,
        spectral_weight=args.spectral_weight,
    )
    
    print(f"\n🎵 MusicVAE Training Script")
    print(f"   Using device: {DEVICE}")
    
    if args.finetune:
        # Finetune from pretrained model
        results = finetune_pretrained(
            config=config,
            pretrained_model=args.pretrained_model,
            freeze_encoder=args.freeze_encoder,
            freeze_decoder=args.freeze_decoder,
            save_outputs=not args.no_save,
        )
    elif args.resume:
        # Resume training
        results = resume_training(
            checkpoint_path=Path(args.resume),
            config=config,
            save_outputs=not args.no_save,
        )
    else:
        # Train from scratch
        results = train_model(
            config=config,
            save_outputs=not args.no_save,
        )
    
    return results


if __name__ == '__main__':
    main()
