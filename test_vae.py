#!/usr/bin/env python3
"""
MusicVAE Test Script

Test a trained VAE model by reconstructing audio and visualizing spectrograms.
Provides both visual (spectrogram) and auditory (audio playback) comparison.

Usage:
    # Test with default checkpoint and random audio sample
    python test_vae.py

    # Test with specific checkpoint
    python test_vae.py --checkpoint checkpoints/music_vae/best.pt

    # Test with specific audio file
    python test_vae.py --audio Data/GTZAN-decompressed/audio/jazz/jazz.00001.wav

    # Save outputs without displaying
    python test_vae.py --save-dir outputs/test_results --no-display

    # Test multiple samples
    python test_vae.py --num-samples 5
"""

import argparse
import math
import random
import sys
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
# TEST FUNCTIONS
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
        - original_audio: Original audio waveform
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
        'original_audio': segment,
        'mean': mean,
        'std': std,
        'ref_value': ref_value,
    }


def reconstruct(
    model: MusicVAE,
    prepared_input: Dict,
    device: torch.device = DEVICE,
    griffin_lim_iters: int = 64,
) -> Dict:
    """
    Run reconstruction through the model.
    
    Returns dict with:
        - recon_log_mel: Reconstructed log mel spectrogram
        - recon_audio: Reconstructed audio waveform
        - mse: Mean squared error between original and reconstructed
        - mu, logvar: Latent space parameters
    """
    model.eval()
    
    input_tensor = prepared_input['input_tensor'].to(device)
    mean = prepared_input['mean']
    std = prepared_input['std']
    ref_value = prepared_input['ref_value']
    
    with torch.no_grad():
        recon, mu, logvar = model(input_tensor)
    
    # Denormalize reconstruction
    recon = recon.squeeze(0).cpu()  # (1, n_mels, time)
    recon_denorm = recon * (std + NORMALIZATION_EPS) + mean
    
    # Calculate MSE
    raw_log_mel = prepared_input['raw_log_mel'].cpu()
    mse = torch.mean((recon_denorm.squeeze(0) - raw_log_mel) ** 2).item()
    
    # Convert to audio
    ref_db = 10.0 * torch.log10(ref_value.cpu().clamp_min(MIN_AMP))
    recon_power = torch.pow(10.0, (recon_denorm + ref_db) / 10.0)
    recon_audio = mel_spectrogram_to_audio(
        recon_power, 
        sample_rate=TARGET_SAMPLE_RATE, 
        n_iters=griffin_lim_iters
    ).squeeze(0).numpy()
    
    return {
        'recon_log_mel': recon_denorm.squeeze(0).numpy(),
        'recon_audio': recon_audio,
        'mse': mse,
        'mu': mu.cpu().numpy(),
        'logvar': logvar.cpu().numpy(),
    }


def plot_comparison(
    original_mel: np.ndarray,
    recon_mel: np.ndarray,
    audio_path: str,
    mse: float,
    save_path: Optional[Path] = None,
    show: bool = True,
) -> plt.Figure:
    """Create side-by-side spectrogram comparison plot."""
    
    # Ensure 2D arrays
    if original_mel.ndim > 2:
        original_mel = np.squeeze(original_mel)
    if recon_mel.ndim > 2:
        recon_mel = np.squeeze(recon_mel)
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # Shared color scale
    vmin = min(original_mel.min(), recon_mel.min())
    vmax = max(original_mel.max(), recon_mel.max())
    
    # Original spectrogram
    im1 = librosa.display.specshow(
        original_mel,
        sr=TARGET_SAMPLE_RATE,
        hop_length=HOP_LENGTH,
        x_axis='time',
        y_axis='mel',
        cmap='magma',
        ax=axes[0],
        vmin=vmin,
        vmax=vmax,
    )
    axes[0].set_title('Original', fontsize=14, fontweight='bold')
    plt.colorbar(im1, ax=axes[0], format='%+2.0f dB')
    
    # Reconstructed spectrogram
    im2 = librosa.display.specshow(
        recon_mel,
        sr=TARGET_SAMPLE_RATE,
        hop_length=HOP_LENGTH,
        x_axis='time',
        y_axis='mel',
        cmap='magma',
        ax=axes[1],
        vmin=vmin,
        vmax=vmax,
    )
    axes[1].set_title('Reconstructed', fontsize=14, fontweight='bold')
    plt.colorbar(im2, ax=axes[1], format='%+2.0f dB')
    
    # Difference
    diff = np.abs(original_mel - recon_mel)
    im3 = librosa.display.specshow(
        diff,
        sr=TARGET_SAMPLE_RATE,
        hop_length=HOP_LENGTH,
        x_axis='time',
        y_axis='mel',
        cmap='hot',
        ax=axes[2],
    )
    axes[2].set_title('Absolute Difference', fontsize=14, fontweight='bold')
    plt.colorbar(im3, ax=axes[2], format='%+2.0f dB')
    
    # Title with file info and MSE
    filename = Path(audio_path).name
    fig.suptitle(
        f'VAE Reconstruction: {filename}\nMSE: {mse:.4f}',
        fontsize=16,
        fontweight='bold',
        y=1.02
    )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  📊 Saved spectrogram comparison to: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


def play_audio_comparison(
    original_audio: np.ndarray,
    recon_audio: np.ndarray,
    sample_rate: int = TARGET_SAMPLE_RATE,
) -> None:
    """
    Play original and reconstructed audio for comparison.
    Uses sounddevice if available, otherwise saves to temp files.
    """
    try:
        import sounddevice as sd
        
        print("\n  🔊 Playing original audio...")
        sd.play(original_audio, sample_rate)
        sd.wait()
        
        import time
        time.sleep(0.5)  # Brief pause between clips
        
        print("  🔊 Playing reconstructed audio...")
        sd.play(recon_audio, sample_rate)
        sd.wait()
        
    except ImportError:
        print("\n  ⚠️  sounddevice not installed. Install with: pip install sounddevice")
        print("     Saving audio files instead...")
        
        # Save to temp files
        import tempfile
        temp_dir = Path(tempfile.gettempdir())
        
        orig_path = temp_dir / 'vae_test_original.wav'
        recon_path = temp_dir / 'vae_test_reconstructed.wav'
        
        sf.write(str(orig_path), original_audio, sample_rate)
        sf.write(str(recon_path), recon_audio, sample_rate)
        
        print(f"     Original: {orig_path}")
        print(f"     Reconstructed: {recon_path}")
        
        # Try to open with system player on macOS
        try:
            import subprocess
            print("\n  🔊 Playing with system player...")
            subprocess.run(['afplay', str(orig_path)], check=True)
            subprocess.run(['afplay', str(recon_path)], check=True)
        except Exception:
            print("     Could not auto-play. Please open the files manually.")


def save_audio_files(
    original_audio: np.ndarray,
    recon_audio: np.ndarray,
    save_dir: Path,
    sample_name: str,
    sample_rate: int = TARGET_SAMPLE_RATE,
) -> Tuple[Path, Path]:
    """Save original and reconstructed audio to files."""
    save_dir.mkdir(parents=True, exist_ok=True)
    
    orig_path = save_dir / f'{sample_name}_original.wav'
    recon_path = save_dir / f'{sample_name}_reconstructed.wav'
    
    sf.write(str(orig_path), original_audio, sample_rate)
    sf.write(str(recon_path), recon_audio, sample_rate)
    
    print(f"  🎵 Saved original audio: {orig_path}")
    print(f"  🎵 Saved reconstructed audio: {recon_path}")
    
    return orig_path, recon_path


def run_test(
    checkpoint_path: Path,
    audio_path: Optional[Path] = None,
    save_dir: Optional[Path] = None,
    play_audio: bool = True,
    show_plot: bool = True,
    griffin_lim_iters: int = 64,
) -> Dict:
    """
    Run a complete test on a single audio file.
    
    Args:
        checkpoint_path: Path to model checkpoint
        audio_path: Path to audio file (random if None)
        save_dir: Directory to save outputs (None = don't save)
        play_audio: Whether to play audio comparison
        show_plot: Whether to display spectrogram plot
        griffin_lim_iters: Number of Griffin-Lim iterations
    
    Returns:
        Dictionary with test results
    """
    # Select audio file
    if audio_path is None:
        audio_files = list_audio_files(AUDIO_DIR)
        if not audio_files:
            raise RuntimeError(f"No audio files found in {AUDIO_DIR}")
        audio_path = random.choice(audio_files)
    
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    print(f"\n{'='*60}")
    print(f"  MusicVAE Reconstruction Test")
    print(f"{'='*60}")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Audio: {audio_path.name}")
    print(f"  Device: {DEVICE}")
    print(f"{'='*60}\n")
    
    # Load model
    print("  📦 Loading model...")
    model = load_model(checkpoint_path, DEVICE)
    
    # Prepare input
    print("  🎵 Processing audio...")
    prepared = prepare_input(audio_path, DEVICE)
    
    # Run reconstruction
    print("  🔄 Running reconstruction...")
    results = reconstruct(model, prepared, DEVICE, griffin_lim_iters)
    
    print(f"\n  📈 Results:")
    print(f"     MSE: {results['mse']:.4f}")
    print(f"     Latent dim: {results['mu'].shape[1]}")
    
    # Get sample name for saving
    sample_name = audio_path.stem
    
    # Save outputs
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save audio
        save_audio_files(
            prepared['original_audio'],
            results['recon_audio'],
            save_dir,
            sample_name,
        )
        
        # Save spectrogram plot
        plot_path = save_dir / f'{sample_name}_comparison.png'
        plot_comparison(
            prepared['raw_log_mel'].cpu().numpy(),
            results['recon_log_mel'],
            str(audio_path),
            results['mse'],
            save_path=plot_path,
            show=show_plot,
        )
    elif show_plot:
        # Just show plot without saving
        plot_comparison(
            prepared['raw_log_mel'].cpu().numpy(),
            results['recon_log_mel'],
            str(audio_path),
            results['mse'],
            show=True,
        )
    
    # Play audio
    if play_audio:
        play_audio_comparison(
            prepared['original_audio'],
            results['recon_audio'],
        )
    
    return {
        'audio_path': str(audio_path),
        'checkpoint': str(checkpoint_path),
        'mse': results['mse'],
        'original_audio': prepared['original_audio'],
        'recon_audio': results['recon_audio'],
        'original_mel': prepared['raw_log_mel'].cpu().numpy(),
        'recon_mel': results['recon_log_mel'],
    }


def run_batch_test(
    checkpoint_path: Path,
    num_samples: int = 5,
    save_dir: Optional[Path] = None,
    show_plots: bool = False,
    play_audio: bool = False,
) -> List[Dict]:
    """
    Run tests on multiple random audio samples.
    
    Args:
        checkpoint_path: Path to model checkpoint
        num_samples: Number of samples to test
        save_dir: Directory to save outputs
        show_plots: Whether to display plots
        play_audio: Whether to play audio
    
    Returns:
        List of test result dictionaries
    """
    audio_files = list_audio_files(AUDIO_DIR)
    if not audio_files:
        raise RuntimeError(f"No audio files found in {AUDIO_DIR}")
    
    # Select random samples
    num_samples = min(num_samples, len(audio_files))
    selected = random.sample(audio_files, num_samples)
    
    print(f"\n{'='*60}")
    print(f"  MusicVAE Batch Test - {num_samples} samples")
    print(f"{'='*60}")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"{'='*60}\n")
    
    # Load model once
    print("  📦 Loading model...")
    model = load_model(checkpoint_path, DEVICE)
    
    results = []
    mse_values = []
    
    for i, audio_path in enumerate(selected, 1):
        print(f"\n  [{i}/{num_samples}] Testing: {audio_path.name}")
        
        try:
            prepared = prepare_input(audio_path, DEVICE)
            recon_results = reconstruct(model, prepared, DEVICE)
            
            mse = recon_results['mse']
            mse_values.append(mse)
            print(f"     MSE: {mse:.4f}")
            
            sample_name = audio_path.stem
            
            # Save if directory specified
            if save_dir:
                sample_dir = Path(save_dir) / sample_name
                sample_dir.mkdir(parents=True, exist_ok=True)
                
                save_audio_files(
                    prepared['original_audio'],
                    recon_results['recon_audio'],
                    sample_dir,
                    sample_name,
                )
                
                plot_path = sample_dir / f'{sample_name}_comparison.png'
                plot_comparison(
                    prepared['raw_log_mel'].cpu().numpy(),
                    recon_results['recon_log_mel'],
                    str(audio_path),
                    mse,
                    save_path=plot_path,
                    show=show_plots,
                )
            
            if play_audio:
                play_audio_comparison(
                    prepared['original_audio'],
                    recon_results['recon_audio'],
                )
            
            results.append({
                'audio_path': str(audio_path),
                'mse': mse,
                'success': True,
            })
            
        except Exception as e:
            print(f"     ❌ Error: {e}")
            results.append({
                'audio_path': str(audio_path),
                'error': str(e),
                'success': False,
            })
    
    # Summary
    if mse_values:
        print(f"\n{'='*60}")
        print(f"  Summary")
        print(f"{'='*60}")
        print(f"  Samples tested: {len(mse_values)}/{num_samples}")
        print(f"  Average MSE: {np.mean(mse_values):.4f}")
        print(f"  Min MSE: {np.min(mse_values):.4f}")
        print(f"  Max MSE: {np.max(mse_values):.4f}")
        print(f"  Std MSE: {np.std(mse_values):.4f}")
        if save_dir:
            print(f"  Results saved to: {save_dir}")
        print(f"{'='*60}\n")
    
    return results


# ============================================================================
# CLI
# ============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Test a trained MusicVAE model with audio reconstruction.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test with default checkpoint and random audio
  python test_vae.py

  # Test with specific checkpoint
  python test_vae.py --checkpoint checkpoints/music_vae/best.pt

  # Test with specific audio file
  python test_vae.py --audio Data/GTZAN-decompressed/audio/jazz/jazz.00001.wav

  # Save outputs without displaying
  python test_vae.py --save-dir outputs/test_results --no-display

  # Test multiple samples
  python test_vae.py --num-samples 5 --save-dir outputs/batch_test
        """
    )
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='checkpoints/music_vae/best.pt',
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--audio',
        type=str,
        default=None,
        help='Path to audio file (random if not specified)'
    )
    parser.add_argument(
        '--save-dir',
        type=str,
        default=None,
        help='Directory to save outputs'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=1,
        help='Number of samples to test (for batch mode)'
    )
    parser.add_argument(
        '--no-audio',
        action='store_true',
        help='Do not play audio'
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
        help='Number of Griffin-Lim iterations for audio reconstruction'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed for sample selection'
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    if args.seed is not None:
        random.seed(args.seed)
    
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        # Try some common locations
        alternatives = [
            Path('checkpoints/music_vae/best.pt'),
            Path('checkpoints/music_vae/last.pt'),
            Path('checkpoints/music_vae_improved/best.pt'),
            Path('checkpoints/music_vae_improved/last.pt'),
        ]
        for alt in alternatives:
            if alt.exists():
                checkpoint_path = alt
                print(f"  ℹ️  Using checkpoint: {checkpoint_path}")
                break
        else:
            print(f"  ❌ Checkpoint not found: {args.checkpoint}")
            print(f"     Tried alternatives: {[str(a) for a in alternatives]}")
            sys.exit(1)
    
    save_dir = Path(args.save_dir) if args.save_dir else None
    
    if args.num_samples > 1:
        # Batch mode
        run_batch_test(
            checkpoint_path=checkpoint_path,
            num_samples=args.num_samples,
            save_dir=save_dir,
            show_plots=not args.no_display,
            play_audio=not args.no_audio,
        )
    else:
        # Single sample mode
        audio_path = Path(args.audio) if args.audio else None
        run_test(
            checkpoint_path=checkpoint_path,
            audio_path=audio_path,
            save_dir=save_dir,
            play_audio=not args.no_audio,
            show_plot=not args.no_display,
            griffin_lim_iters=args.griffin_lim_iters,
        )


if __name__ == '__main__':
    main()

