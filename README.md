# MusicVAE — Understanding Sound Through a Variational Autoencoder

This repository contains the full code, training utilities, figures, and documentation for my final project: building and improving a Variational Autoencoder (VAE) capable of reconstructing mel-spectrograms of musical audio.

This README serves as a complete, non-technical explanation of the project—how the model works, why it was built, the methodology behind training, and what the results look like. All behind-the-scenes implementation details, including code for the improved loss functions and training pipeline, are included in the repository files and described in more technical detail in the codebase. 

[Dataset link](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification)

[Blog Post](https://ajm-l.github.io/website/#/projects/musicvae)

## 1. Introduction and Problem Statement

Machine learning has made it possible to generate images, text, and even entire pieces of music—but understanding how models learn sound remains an intriguing challenge. This project explores that question by training a Variational Autoencoder (VAE) to learn the structure of audio through its mel-spectrogram representation.

The goal is not to generate finished music, but to investigate what a model “chooses” to remember about sound when it compresses it into a small latent space and reconstructs it. If the VAE produces coherent reconstructions, it suggests the model has meaningfully captured the harmonic, spectral, and timbral features in the data. If it fails, the failure modes reveal how neural networks struggle with audio and what improvements are necessary.

All code, logs, training curves, checkpoints, and figures used in this explanation are available in this repository.

## 2. Methodology and Data
Model Overview

A Variational Autoencoder consists of two parts:

Encoder — reduces a mel-spectrogram into a small latent vector.

Decoder — recreates a spectrogram from that compressed representation.

Mel-spectrograms are used instead of raw audio because they provide a structured, visually interpretable form of sound that neural networks can learn far more easily.

Training Challenges

Initial experiments revealed several issues:

- Reconstructions lacked high-frequency details.

- Output values drifted outside the true data range.

- Many training runs collapsed entirely, producing blank spectrograms.

These problems revealed that audio requires more than standard reconstruction loss.

![20 Epochs of training curves MSE](figures/training_curves_MSE.png)
![100 Epochs of training curves MSE](figures/training_curves_MSE100.png)
Training with MSE for 20 vs 100 epochs

### Improvements Implemented

To address training instability and reconstruction quality problems, I redesigned the loss function and training pipeline. The improvements include:

Frequency-Weighted Loss:
Weights important mel bands—fundamentals and harmonics—more highly than noisy high frequencies.

Output Range Regularization:
Prevents the model from producing unrealistic values far outside the input range.

Spectral Loss:
Adds a frequency-domain comparison (via FFT) to encourage accurate harmonic structure.

Stability-Driven Training Setup:
A lower learning rate, gradient clipping, more training epochs, and full checkpointing support ensure stable long-term training.

All implementation details can be found in the training script (train-improved-VAE.py) and technical notes in README.md 

![20 Epochs of training curves ELBO](figures/training_curves.png)
20 Epochs training with ELBO

Dataset

The dataset was scraped from a collection of short audio samples and converted to mel-spectrograms using Librosa. Instructions for preprocessing and reproducing the dataset are included here:
[Dataset](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification)

## 3. Results

After implementing the improved loss and training strategies, reconstruction quality increased significantly. Training curves show clearly when loss terms start behaving predictably, and reconstructions demonstrate major improvements in stability and detail.

Training Curves

Figures in this repository illustrate:

 - Total loss across epochs

- Reconstruction vs KL loss components

- The impact of the spectral loss on training stability

- These figures can be found in the /figures directory.

- Spectrogram Reconstructions



Before improvements, many reconstructions were blurred, desaturated, or entirely flat. After redesigning the loss:

- Harmonics became clearer.

- Low-frequency fundamentals were preserved.

- Extreme out-of-range values were reduced.

- Collapse events became rare.


![Reconstructed mel-spectrogram sample](figures/disco.00056_comparison.png)

[disco.00056_original.wav](figures/disco.00056_original.wav)

[disco.00056_reconstructed.wav](figures/disco.00056_reconstructed.wav)

Latent Interpolations

Interpolating between two latent vectors produces smooth transitions between their respective spectrograms. This demonstrates that the VAE learned a meaningful latent space rather than memorizing the inputs.

![Latent interpolation between samples](figures/interpolation_linear_walk.png)

## 4. Discussion

The improved VAE successfully captures the broad structure of sound—especially rythmic spacing, simple hamronics, and low-frequency components. However, high-frequency details remain difficult to model due to their noise-like structure and higher variability.

The project also highlights broader insights seen in contemporary research: VAEs often struggle with detailed or perceptually sharp audio, and require domain-specific losses to achieve convincing reconstructions. Diffusion models and transformers outperform VAEs in fidelity, but their complexity makes them less ideal for learning-oriented projects like this one.

Challenges such as model collapse, unstable gradients, and poor high-frequency reconstruction guided the design of the improved loss functions. Through steady iteration and careful experiment tracking, these issues were significantly reduced.

## 5. Conclusion

This project demonstrates that even a relatively small neural network can learn meaningful aspects of musical sound when guided with carefully designed loss functions and domain-aware training strategies. The improved MusicVAE reconstructs mel-spectrograms more faithfully, captures harmonic structure, and maintains stable training behavior.

The project not only produced better reconstructions, but deepened my understanding of how machine-learning models interpret sound, how loss design shapes learning behavior, and why audio modeling often requires approaches beyond standard image-based techniques.

Future directions include experimenting with larger bottleneck sizes, skip connections, conditional VAEs, and potentially turning the reconstructed spectrograms back into audio for listening tests.

All code, training utilities, figures, and checkpoints are available in this repository.

Dataset link: https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification