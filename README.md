# MusicVAE - Training Improvements Documentation

## Overview

This document describes the improvements made to address poor reconstruction quality in the MusicVAE model. The changes include enhanced loss functions, improved training strategies, and checkpoint resumption capabilities.

## Problems Identified

After extensive training, the model exhibited several reconstruction issues:

1. **Output Range Mismatch**: Reconstructions had a wider range than inputs ([-7.3, 6.8] vs [-4.2, 3.1])
2. **Missing High Frequencies**: Model failed to capture fine spectral details and high-frequency content
3. **Featureless Outputs**: Some reconstructions were completely dark (no signal), indicating model collapse

## Solutions Implemented

### 1. Improved Loss Function (`vae_loss_spectral`)

A new loss function that addresses the reconstruction issues:

- **Frequency-Weighted Loss**: Emphasizes important frequencies (fundamentals and harmonics) over noise
- **Output Range Regularization**: Penalizes outputs that fall outside the expected input range
- **Spectral Loss**: Frequency-domain loss that better captures frequency structure

**Key Features:**
- `freq_weight`: Tensor that weights different frequency bins (emphasizes fundamentals and harmonics)
- `output_reg_weight`: Weight for output range regularization (default: 0.01)
- `spectral_weight`: Weight for spectral loss component (default: 0.1)

### 2. Frequency Weighting (`create_frequency_weights`)

Creates a frequency weighting scheme that:
- **Low frequencies (0-20% of bins)**: Weighted 1.5x (fundamentals)
- **Mid-high frequencies (30-70% of bins)**: Weighted 2.0x (harmonics)
- **Very high frequencies (80-100% of bins)**: Weighted 0.8x (often noise)

This helps the model focus on musically important frequencies.

### 3. Improved Training Functions

#### `train_epoch_improved` and `evaluate_epoch_improved`
Training and evaluation functions that use the improved loss function with all enhancements.

#### `train_vae_model_improved`
Complete training function that:
- Uses frequency-weighted loss
- Applies output range regularization
- Includes spectral loss component
- Provides detailed loss breakdown (reconstruction, KL, total)

### 4. Recommended Configuration (`IMPROVED_CONFIG`)

Optimized hyperparameters for better reconstruction quality:

```python
IMPROVED_CONFIG = TrainingConfig(
    epochs=100,              # Train longer for better convergence
    lr=5e-5,                 # Lower learning rate for stability
    beta=0.05,               # Low beta to focus on reconstruction
    grad_clip=1.0,
    val_split=0.1,
    seed=123,
    checkpoint_dir=Path('checkpoints/music_vae_improved'),
)
```

**Key Changes:**
- Lower learning rate (5e-5 vs 1e-4) for more stable training
- Lower beta (0.05 vs 0.1) to prioritize reconstruction over regularization
- More epochs (100) for better convergence

### 5. Checkpoint Resumption (`resume_training`)

Added functionality to resume training from existing checkpoints:

- **Loads model and optimizer states**: Continues training seamlessly
- **Resumes from correct epoch**: Automatically continues from `checkpoint_epoch + 1`
- **Supports hyperparameter adjustment**: Can change learning rate, beta, etc. when resuming
- **Preserves best validation loss**: Tracks best performance across resumed training

**Usage:**
```python
resume_config = TrainingConfig(
    epochs=150,              # Total epochs (continues from checkpoint)
    lr=5e-5,                 # Can adjust learning rate
    beta=0.05,
    checkpoint_dir=Path('checkpoints/music_vae_improved'),
)

results = resume_training(
    checkpoint_path='checkpoints/music_vae_improved/last.pt',
    config=resume_config,
    save_outputs=True,
)
```

## Usage Guide

### Training with Improved Loss

1. **Set up the configuration:**
```python
from pathlib import Path
from dataclasses import dataclass

IMPROVED_CONFIG = TrainingConfig(
    epochs=100,
    lr=5e-5,
    beta=0.05,
    grad_clip=1.0,
    val_split=0.1,
    seed=123,
    checkpoint_dir=Path('checkpoints/music_vae_improved'),
)
```

2. **Train with improved loss:**
```python
results = train_vae_model_improved(
    config=IMPROVED_CONFIG,
    save_outputs=True,
    output_reg_weight=0.01,  # Penalize outputs outside input range
    spectral_weight=0.1,      # Weight for frequency-domain loss
)
```

3. **Monitor training:**
The function prints detailed metrics:
- Total loss
- Reconstruction loss
- KL divergence loss

### Resuming Training

1. **Check available checkpoints:**
```python
# Run the example cell to see available checkpoints
# It will list all checkpoints with their epoch numbers and loss values
```

2. **Resume from checkpoint:**
```python
results = resume_training(
    checkpoint_path='checkpoints/music_vae_improved/last.pt',
    config=IMPROVED_CONFIG,
    save_outputs=True,
)
```

### Adjusting Loss Weights

You can tune the loss components for your specific needs:

```python
# More emphasis on output range (if outputs are too extreme)
results = train_vae_model_improved(
    config=IMPROVED_CONFIG,
    output_reg_weight=0.05,  # Increased from 0.01
    spectral_weight=0.1,
)

# More emphasis on frequency structure (if high frequencies are missing)
results = train_vae_model_improved(
    config=IMPROVED_CONFIG,
    output_reg_weight=0.01,
    spectral_weight=0.2,  # Increased from 0.1
)
```

## Technical Details

### Loss Function Components

The total loss is computed as:

```
total_loss = recon_loss + beta * kl_loss + spectral_weight * spectral_loss + output_reg_weight * output_reg
```

Where:
- **recon_loss**: Frequency-weighted MSE between reconstruction and input
- **kl_loss**: KL divergence between latent distribution and prior
- **spectral_loss**: MSE in frequency domain (FFT magnitude)
- **output_reg**: Penalty for outputs outside input range

### Frequency Weighting Scheme

The frequency weights are applied per mel bin:
- Bins 0-25 (20%): Weight 1.5 (fundamentals)
- Bins 38-89 (30-70%): Weight 2.0 (harmonics)
- Bins 102-127 (80-100%): Weight 0.8 (high frequencies, often noise)

### Output Range Regularization

The regularization term penalizes outputs that fall outside:
```
[input_min - 0.5*range, input_max + 0.5*range]
```

This prevents the model from producing extreme values that don't match the input distribution.

## Expected Improvements

With these changes, you should see:

1. **Better output range**: Reconstructions should stay within input range
2. **Improved frequency content**: High frequencies should be better captured
3. **Fewer featureless outputs**: Model should produce meaningful reconstructions
4. **Lower reconstruction loss**: Overall MSE should decrease

## Files Modified

- `MusicVAE.ipynb`: Added new cells with improved loss functions and training utilities
  - Cell 38-39: Improved loss functions and frequency weighting
  - Cell 40-41: Improved training functions
  - Cell 42-43: Recommended configuration and usage examples
  - Cell 44: Summary and documentation
  - Cell 38-41 (earlier): Resume training functionality

## Next Steps

If reconstructions are still poor after using the improved loss:

1. **Increase training time**: Train for more epochs (150-200)
2. **Adjust loss weights**: Tune `output_reg_weight` and `spectral_weight`
3. **Consider architecture changes**: May need skip connections or larger model capacity
4. **Check data quality**: Ensure input data is properly normalized and preprocessed
5. **Monitor gradients**: Check if gradients are vanishing or exploding

## Notes

- The improved loss functions are backward compatible - you can still use the original `vae_loss` function
- Checkpoint resumption works with both original and improved training functions
- Frequency weights are automatically created if not provided
- All improvements are optional and can be tuned based on your specific needs

## Version History

- **Current Version**: Improved loss functions and checkpoint resumption
- **Previous**: Basic VAE training with standard MSE loss
- **Future**: Potential architecture improvements (skip connections, attention mechanisms)


