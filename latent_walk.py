#!/usr/bin/env python3
"""
MusicVAE Latent Walk Script

Generate latent walks through the latent space of a trained MusicVAE model.
Supports linear interpolation, spherical interpolation, and dimension-wise walks.

Usage:
    # Linear interpolation between two audio files
    python latent_walk.py --checkpoint checkpoints/music_vae/best.pt \
        --audio1 Data/GTZAN-decompressed/audio/jazz/jazz.00001.wav \
        --audio2 Data/GTZAN-decompressed/audio/metal/metal.00001.wav \
        --steps 10

    # Walk along a single latent dimension
    python latent_walk.py --checkpoint checkpoints/music_vae/best.pt \
        --dimension 0 --range -3 3 --steps 20

    # Spherical interpolation between two points
    python latent_walk.py --checkpoint checkpoints/music_vae/best.pt \
        --audio1 file1.wav --audio2 file2.wav --steps 10 --interpolation slerp
"""

import argparse
import math
import random
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.functional as AF
import torchaudio.transforms as T

warnings.filterwarnings('ignore', category=FutureWarning)

# ============================================================================
# CONFIGURATION (must match training)
# ============================================================================

DATA_ROOT = Path('./Data/GTZAN-decompressed')
AUDIO_DIR = DATA_ROOT / 'audio'

DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')

TARGET_SAMPLE_RATE: int = 22_050
SEGMENT_DURATION: float = 3.0
N_FFT: int = 1_024
HOP_LENGTH: int = 256
WIN_LENGTH: int = 1_024
N_MELS: int = 128
FMIN: float = 30.0
FMAX: float = TARGET_SAMPLE_RATE / 2
PAD_MODE: str = 'reflect'
MIN_AMP: float = 1e-5
NORMALIZATION_EPS: float = 1e-6

SEGMENT_SAMPLES = int(SEGMENT_DURATION * TARGET_SAMPLE_RATE)
DEFAULT_TIME_FRAMES = int(math.ceil((SEGMENT_DURATION * TARGET_SAMPLE_RATE) / HOP_LENGTH))

AUDIO_EXTENSIONS = ('.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aiff', '.aif')


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
    waveform = None
    sample_rate = None
    
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


def power_to_log_db(power_spec: torch.Tensor, amin: float = MIN_AMP) -> torch.Tensor:
    clamped = power_spec.clamp_min(amin)
    ref_value = clamped.max().clamp_min(amin)
    log_spec = 10.0 * torch.log10(clamped) - 10.0 * torch.log10(ref_value)
    return log_spec


def list_audio_files(directory: Path) -> List[Path]:
    """Return all audio files in directory."""
    if not directory.exists():
        return []
    return sorted(
        path for path in directory.rglob('*') 
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS
    )


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


# ============================================================================
# LATENT WALK UTILITIES
# ============================================================================

def load_model(checkpoint_path: Path, device: torch.device = DEVICE) -> MusicVAE:
    """Load a trained model from checkpoint."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Get model config from checkpoint if available
    config = checkpoint.get('config', {})
    latent_dim = config.get('latent_dim', 128)
    hidden_channels = config.get('hidden_channels', (32, 64, 128, 256))
    if isinstance(hidden_channels, list):
        hidden_channels = tuple(hidden_channels)
    
    model = MusicVAE(
        latent_dim=latent_dim,
        hidden_channels=hidden_channels,
    ).to(device)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    
    return model


def prepare_input(audio_path: Path, device: torch.device = DEVICE) -> Dict:
    """
    Prepare audio input for the model.
    
    Returns dict with:
        - input_tensor: Normalized tensor ready for model
        - raw_log_mel: Un-normalized log mel spectrogram
        - mean, std: Normalization stats
        - ref_value: Reference value for denormalization
    """
    waveform, _ = load_audio(audio_path)
    segment = pad_or_trim(waveform, SEGMENT_SAMPLES)
    segment_tensor = torch.from_numpy(segment).unsqueeze(0).to(device)
    
    mel_transform = build_mel_transform().to(device)
    with torch.no_grad():
        power_spec = mel_transform(segment_tensor)
    
    clamped = power_spec.clamp_min(MIN_AMP)
    ref_value = clamped.max()
    log_spec = 10.0 * torch.log10(clamped) - 10.0 * torch.log10(ref_value)
    log_spec = log_spec.squeeze(0)  # (n_mels, time)
    
    mean = log_spec.mean().item()
    std = log_spec.std().item()
    std = std if std > 0 else NORMALIZATION_EPS
    
    normalized = (log_spec - mean) / (std + NORMALIZATION_EPS)
    input_tensor = normalized.unsqueeze(0).unsqueeze(0)  # (1, 1, n_mels, time)
    
    return {
        'input_tensor': input_tensor,
        'raw_log_mel': log_spec,
        'mean': mean,
        'std': std,
        'ref_value': ref_value,
    }


def encode_to_latent(model: MusicVAE, input_tensor: torch.Tensor) -> torch.Tensor:
    """
    Encode input to latent space, returning the mean (mu) of the distribution.
    For latent walks, we use mu rather than sampling to get deterministic results.
    """
    model.eval()
    with torch.no_grad():
        mu, _ = model.encode(input_tensor)
    return mu


def decode_from_latent(
    model: MusicVAE,
    z: torch.Tensor,
    mean: float,
    std: float,
    ref_value: torch.Tensor,
    griffin_lim_iters: int = 64,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Decode latent vector to audio.
    
    Returns:
        - recon_log_mel: Reconstructed log mel spectrogram (numpy)
        - recon_audio: Reconstructed audio waveform (numpy)
    """
    model.eval()
    with torch.no_grad():
        recon = model.decode(z)
    
    # Denormalize reconstruction
    recon = recon.squeeze(0).cpu()  # (1, n_mels, time)
    recon_denorm = recon * (std + NORMALIZATION_EPS) + mean
    
    # Convert to audio
    ref_db = 10.0 * torch.log10(ref_value.cpu().clamp_min(MIN_AMP))
    recon_power = torch.pow(10.0, (recon_denorm + ref_db) / 10.0)
    recon_audio = mel_spectrogram_to_audio(
        recon_power, 
        sample_rate=TARGET_SAMPLE_RATE, 
        n_iters=griffin_lim_iters
    ).squeeze(0).numpy()
    
    return recon_denorm.squeeze(0).numpy(), recon_audio


def linear_interpolate(z1: torch.Tensor, z2: torch.Tensor, alpha: float) -> torch.Tensor:
    """Linear interpolation between two latent vectors."""
    return (1 - alpha) * z1 + alpha * z2


def spherical_interpolate(z1: torch.Tensor, z2: torch.Tensor, alpha: float) -> torch.Tensor:
    """
    Spherical linear interpolation (SLERP) between two latent vectors.
    This preserves the magnitude of the vectors, which can be useful for VAEs.
    """
    # Normalize vectors
    z1_norm = z1 / (torch.norm(z1, dim=-1, keepdim=True) + 1e-8)
    z2_norm = z2 / (torch.norm(z2, dim=-1, keepdim=True) + 1e-8)
    
    # Dot product
    dot = (z1_norm * z2_norm).sum(dim=-1, keepdim=True)
    dot = torch.clamp(dot, -1.0, 1.0)
    
    # Angle
    theta = torch.acos(dot)
    sin_theta = torch.sin(theta)
    
    # Avoid division by zero
    w1 = torch.sin((1 - alpha) * theta) / (sin_theta + 1e-8)
    w2 = torch.sin(alpha * theta) / (sin_theta + 1e-8)
    
    # Interpolate
    z_interp = w1 * z1 + w2 * z2
    
    return z_interp


def interpolate_between_points(
    model: MusicVAE,
    z1: torch.Tensor,
    z2: torch.Tensor,
    num_steps: int,
    interpolation: str = 'linear',
    mean: float = 0.0,
    std: float = 1.0,
    ref_value: torch.Tensor = None,
    griffin_lim_iters: int = 64,
) -> List[Dict]:
    """
    Generate interpolated samples between two latent vectors.
    
    Args:
        model: Trained VAE model
        z1: Starting latent vector
        z2: Ending latent vector
        num_steps: Number of interpolation steps
        interpolation: 'linear' or 'slerp'
        mean, std: Normalization stats for denormalization
        ref_value: Reference value for audio conversion
        griffin_lim_iters: Griffin-Lim iterations
    
    Returns:
        List of dictionaries with 'alpha', 'z', 'mel', 'audio'
    """
    if ref_value is None:
        ref_value = torch.tensor(1.0)
    
    results = []
    alphas = np.linspace(0, 1, num_steps)
    
    for alpha in alphas:
        if interpolation == 'slerp':
            z_interp = spherical_interpolate(z1, z2, alpha)
        else:  # linear
            z_interp = linear_interpolate(z1, z2, alpha)
        
        mel, audio = decode_from_latent(
            model, z_interp, mean, std, ref_value, griffin_lim_iters
        )
        
        results.append({
            'alpha': alpha,
            'z': z_interp.cpu().numpy(),
            'mel': mel,
            'audio': audio,
        })
    
    return results


def walk_dimension(
    model: MusicVAE,
    base_z: torch.Tensor,
    dimension: int,
    value_range: Tuple[float, float],
    num_steps: int,
    mean: float = 0.0,
    std: float = 1.0,
    ref_value: torch.Tensor = None,
    griffin_lim_iters: int = 64,
) -> List[Dict]:
    """
    Walk along a single dimension of the latent space.
    
    Args:
        model: Trained VAE model
        base_z: Base latent vector to modify
        dimension: Which dimension to vary
        value_range: (min, max) values for the dimension
        num_steps: Number of steps
        mean, std: Normalization stats
        ref_value: Reference value for audio conversion
        griffin_lim_iters: Griffin-Lim iterations
    
    Returns:
        List of dictionaries with 'value', 'z', 'mel', 'audio'
    """
    if ref_value is None:
        ref_value = torch.tensor(1.0)
    
    results = []
    values = np.linspace(value_range[0], value_range[1], num_steps)
    
    for value in values:
        z_modified = base_z.clone()
        z_modified[0, dimension] = value
        
        mel, audio = decode_from_latent(
            model, z_modified, mean, std, ref_value, griffin_lim_iters
        )
        
        results.append({
            'value': value,
            'z': z_modified.cpu().numpy(),
            'mel': mel,
            'audio': audio,
        })
    
    return results


# ============================================================================
# VISUALIZATION AND SAVING
# ============================================================================

def plot_latent_walk(
    results: List[Dict],
    save_path: Optional[Path] = None,
    show: bool = True,
    title: str = "Latent Walk",
) -> plt.Figure:
    """Create a grid visualization of spectrograms from a latent walk."""
    num_samples = len(results)
    cols = min(5, num_samples)
    rows = (num_samples + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 2.5))
    if num_samples == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes if isinstance(axes, np.ndarray) else [axes]
    else:
        axes = axes.flatten()
    
    # Find shared color scale
    all_mels = [r['mel'] for r in results]
    vmin = min(m.min() for m in all_mels)
    vmax = max(m.max() for m in all_mels)
    
    for idx, result in enumerate(results):
        ax = axes[idx] if num_samples > 1 else axes[0]
        
        mel = result['mel']
        label = result.get('alpha', result.get('value', idx))
        
        librosa.display.specshow(
            mel,
            sr=TARGET_SAMPLE_RATE,
            hop_length=HOP_LENGTH,
            x_axis='time',
            y_axis='mel',
            cmap='magma',
            ax=ax,
            vmin=vmin,
            vmax=vmax,
        )
        
        if isinstance(label, float):
            ax.set_title(f'{label:.2f}', fontsize=10)
        else:
            ax.set_title(f'Step {idx}', fontsize=10)
        ax.set_xlabel('')
        ax.set_ylabel('')
    
    # Hide unused subplots
    for idx in range(num_samples, len(axes)):
        axes[idx].axis('off')
    
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  📊 Saved visualization to: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


def save_walk_audio(
    results: List[Dict],
    save_dir: Path,
    prefix: str = "walk",
) -> List[Path]:
    """Save audio files from a latent walk."""
    save_dir.mkdir(parents=True, exist_ok=True)
    audio_paths = []
    
    for idx, result in enumerate(results):
        label = result.get('alpha', result.get('value', idx))
        if isinstance(label, float):
            filename = f"{prefix}_step{idx:03d}_alpha{label:.3f}.wav"
        else:
            filename = f"{prefix}_step{idx:03d}.wav"
        
        audio_path = save_dir / filename
        sf.write(str(audio_path), result['audio'], TARGET_SAMPLE_RATE)
        audio_paths.append(audio_path)
    
    # Also save concatenated audio
    concatenated = np.concatenate([r['audio'] for r in results])
    concat_path = save_dir / f"{prefix}_concatenated.wav"
    sf.write(str(concat_path), concatenated, TARGET_SAMPLE_RATE)
    audio_paths.append(concat_path)
    
    print(f"  🎵 Saved {len(results)} audio files + concatenated version to: {save_dir}")
    
    return audio_paths


# ============================================================================
# MAIN FUNCTIONS
# ============================================================================

def run_interpolation_walk(
    checkpoint_path: Path,
    audio1_path: Optional[Path] = None,
    audio2_path: Optional[Path] = None,
    num_steps: int = 10,
    interpolation: str = 'linear',
    save_dir: Optional[Path] = None,
    show_plot: bool = True,
    griffin_lim_iters: int = 64,
) -> List[Dict]:
    """
    Run a latent walk by interpolating between two audio samples.
    """
    print(f"\n{'='*60}")
    print(f"  Latent Walk: Interpolation")
    print(f"{'='*60}")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Interpolation: {interpolation}")
    print(f"  Steps: {num_steps}")
    print(f"{'='*60}\n")
    
    # Load model
    print("  📦 Loading model...")
    model = load_model(checkpoint_path, DEVICE)
    
    # Select audio files if not provided
    if audio1_path is None or audio2_path is None:
        audio_files = list_audio_files(AUDIO_DIR)
        if len(audio_files) < 2:
            raise RuntimeError(f"Need at least 2 audio files, found {len(audio_files)}")
        
        if audio1_path is None:
            audio1_path = random.choice(audio_files)
        if audio2_path is None:
            remaining = [f for f in audio_files if f != audio1_path]
            audio2_path = random.choice(remaining)
    
    audio1_path = Path(audio1_path)
    audio2_path = Path(audio2_path)
    
    print(f"  🎵 Audio 1: {audio1_path.name}")
    print(f"  🎵 Audio 2: {audio2_path.name}")
    
    # Prepare inputs
    print("  🔄 Encoding audio to latent space...")
    prep1 = prepare_input(audio1_path, DEVICE)
    prep2 = prepare_input(audio2_path, DEVICE)
    
    z1 = encode_to_latent(model, prep1['input_tensor'])
    z2 = encode_to_latent(model, prep2['input_tensor'])
    
    # Use average normalization stats
    mean = (prep1['mean'] + prep2['mean']) / 2
    std = (prep1['std'] + prep2['std']) / 2
    ref_value = (prep1['ref_value'] + prep2['ref_value']) / 2
    
    # Generate walk
    print(f"  🚶 Generating {num_steps} interpolation steps...")
    results = interpolate_between_points(
        model, z1, z2, num_steps, interpolation,
        mean, std, ref_value, griffin_lim_iters
    )
    
    # Save outputs
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        plot_path = save_dir / f"interpolation_{interpolation}_walk.png"
        plot_latent_walk(
            results,
            save_path=plot_path,
            show=show_plot,
            title=f"Latent Walk: {audio1_path.stem} → {audio2_path.stem} ({interpolation})"
        )
        
        save_walk_audio(results, save_dir, prefix=f"interp_{interpolation}")
    elif show_plot:
        plot_latent_walk(
            results,
            title=f"Latent Walk: {audio1_path.stem} → {audio2_path.stem} ({interpolation})"
        )
    
    print(f"\n  ✅ Generated {len(results)} samples")
    
    return results


def run_dimension_walk(
    checkpoint_path: Path,
    base_audio_path: Optional[Path] = None,
    dimension: int = 0,
    value_range: Tuple[float, float] = (-3.0, 3.0),
    num_steps: int = 20,
    save_dir: Optional[Path] = None,
    show_plot: bool = True,
    griffin_lim_iters: int = 64,
) -> List[Dict]:
    """
    Run a latent walk along a single dimension.
    """
    print(f"\n{'='*60}")
    print(f"  Latent Walk: Dimension {dimension}")
    print(f"{'='*60}")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Dimension: {dimension}")
    print(f"  Range: {value_range[0]} to {value_range[1]}")
    print(f"  Steps: {num_steps}")
    print(f"{'='*60}\n")
    
    # Load model
    print("  📦 Loading model...")
    model = load_model(checkpoint_path, DEVICE)
    latent_dim = model.latent_dim
    
    if dimension < 0 or dimension >= latent_dim:
        raise ValueError(f"Dimension must be in [0, {latent_dim-1}], got {dimension}")
    
    # Select base audio if not provided
    if base_audio_path is None:
        audio_files = list_audio_files(AUDIO_DIR)
        if not audio_files:
            raise RuntimeError(f"No audio files found in {AUDIO_DIR}")
        base_audio_path = random.choice(audio_files)
    
    base_audio_path = Path(base_audio_path)
    print(f"  🎵 Base audio: {base_audio_path.name}")
    
    # Prepare input and encode
    print("  🔄 Encoding base audio to latent space...")
    prep = prepare_input(base_audio_path, DEVICE)
    base_z = encode_to_latent(model, prep['input_tensor'])
    
    # Generate walk
    print(f"  🚶 Walking along dimension {dimension}...")
    results = walk_dimension(
        model, base_z, dimension, value_range, num_steps,
        prep['mean'], prep['std'], prep['ref_value'], griffin_lim_iters
    )
    
    # Save outputs
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        plot_path = save_dir / f"dimension_{dimension}_walk.png"
        plot_latent_walk(
            results,
            save_path=plot_path,
            show=show_plot,
            title=f"Latent Walk: Dimension {dimension} (range {value_range[0]} to {value_range[1]})"
        )
        
        save_walk_audio(results, save_dir, prefix=f"dim_{dimension}")
    elif show_plot:
        plot_latent_walk(
            results,
            title=f"Latent Walk: Dimension {dimension} (range {value_range[0]} to {value_range[1]})"
        )
    
    print(f"\n  ✅ Generated {len(results)} samples")
    
    return results


# ============================================================================
# CLI
# ============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Generate latent walks through MusicVAE latent space.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Linear interpolation between two audio files
  python latent_walk.py --checkpoint <path_to_checkpoint> \\
      --audio1 Data/GTZAN-decompressed/audio/jazz/jazz.00001.wav \\
      --audio2 Data/GTZAN-decompressed/audio/metal/metal.00001.wav \\
      --steps 10

  # Spherical interpolation
  python latent_walk.py --checkpoint <path_to_checkpoint> \\
      --audio1 file1.wav --audio2 file2.wav --steps 10 --interpolation slerp

  # Walk along a single dimension
  python latent_walk.py --checkpoint <path_to_checkpoint> \\
      --dimension 0 --range -3 3 --steps 20

  # Save outputs
  python latent_walk.py --checkpoint <path_to_checkpoint> \\
      --audio1 file1.wav --audio2 file2.wav --steps 10 \\
      --save-dir outputs/latent_walks
        """
    )
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint (required)'
    )
    
    # Walk type
    parser.add_argument(
        '--dimension',
        type=int,
        default=None,
        help='Walk along a single dimension (if specified, dimension walk mode)'
    )
    
    # Interpolation mode
    parser.add_argument(
        '--audio1',
        type=str,
        default=None,
        help='First audio file for interpolation'
    )
    parser.add_argument(
        '--audio2',
        type=str,
        default=None,
        help='Second audio file for interpolation'
    )
    parser.add_argument(
        '--interpolation',
        type=str,
        choices=['linear', 'slerp'],
        default='linear',
        help='Interpolation method (linear or slerp)'
    )
    
    # Dimension walk mode
    parser.add_argument(
        '--base-audio',
        type=str,
        default=None,
        help='Base audio file for dimension walk'
    )
    parser.add_argument(
        '--range',
        type=float,
        nargs=2,
        default=[-3.0, 3.0],
        metavar=('MIN', 'MAX'),
        help='Value range for dimension walk'
    )
    
    # Common options
    parser.add_argument(
        '--steps',
        type=int,
        default=10,
        help='Number of steps in the walk'
    )
    parser.add_argument(
        '--save-dir',
        type=str,
        default=None,
        help='Directory to save outputs'
    )
    parser.add_argument(
        '--no-display',
        action='store_true',
        help='Do not display plots'
    )
    parser.add_argument(
        '--griffin-lim-iters',
        type=int,
        default=64,
        help='Number of Griffin-Lim iterations'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed'
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
    
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"  ❌ Error: Checkpoint not found: {checkpoint_path}")
        print(f"     Please provide a valid checkpoint path using --checkpoint")
        return
    
    save_dir = Path(args.save_dir) if args.save_dir else None
    
    if args.dimension is not None:
        # Dimension walk mode
        run_dimension_walk(
            checkpoint_path=checkpoint_path,
            base_audio_path=Path(args.base_audio) if args.base_audio else None,
            dimension=args.dimension,
            value_range=tuple(args.range),
            num_steps=args.steps,
            save_dir=save_dir,
            show_plot=not args.no_display,
            griffin_lim_iters=args.griffin_lim_iters,
        )
    else:
        # Interpolation mode
        run_interpolation_walk(
            checkpoint_path=checkpoint_path,
            audio1_path=Path(args.audio1) if args.audio1 else None,
            audio2_path=Path(args.audio2) if args.audio2 else None,
            num_steps=args.steps,
            interpolation=args.interpolation,
            save_dir=save_dir,
            show_plot=not args.no_display,
            griffin_lim_iters=args.griffin_lim_iters,
        )


if __name__ == '__main__':
    main()

