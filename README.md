# Unsupervised Neural Network for Multi-Genre Music Generation

This repository contains the implementation of deep unsupervised generative neural networks for generating novel music pieces across multiple genres, built for the CSE425/EEE474 course.

## Project Structure & File Descriptions

### Core Files
- `README.md`: Project overview and execution guide.
- `requirements.txt`: Python dependencies required to run the project.

### Source Code (`src/`)
- `src/config.py`: Centralized configuration for paths and hyperparameters.
- **Preprocessing (`src/preprocessing/`):**
  - `midi_parser.py`: Utilities for parsing raw MIDI files and extracting metadata.
  - `piano_roll.py`: Logic for converting MIDI to binary piano-roll matrices (Tasks 1 & 2).
  - `tokenize_maestro.py`: Tokenization pipeline using REMI (Task 3).
- **Models (`src/models/`):**
  - `autoencoder.py`: LSTM Autoencoder architecture.
  - `vae.py`: Variational Autoencoder architecture with reparameterization.
  - `transformer.py`: GPT-style decoder-only Transformer.
- **Training (`src/training/`):**
  - `train_ae.py`: Training loop for Task 1.
  - `train_vae.py`: Training loop for Task 2.
  - `train_transformer.py`: Training loop for Task 3.
- **Evaluation (`src/evaluation/`):**
  - `metrics.py`: Implementation of Rhythm Diversity, Repetition Ratio, and Pitch Similarity.
  - `baselines.py`: Markov Chain and Random Generator implementations.
- **Generation (`src/generation/`):**
  - `generate_music.py`: Main inference script for generating new compositions.
  - `midi_export.py`: Converts model outputs back to playable MIDI files.

## Setup & Requirements

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Dataset:**
   Download the MAESTRO dataset and place the unzipped contents in `data/raw_midi/maestro-v3.0.0/`.

## Tasks and Execution

- **Baselines (Random & Markov Chain):**
  Run `notebooks/baseline_markov.ipynb` to train the Markov model, generate baseline samples, and calculate evaluation metrics.
- **Task 1 (LSTM Autoencoder):**
  Run `notebooks/task1_autoencoder.ipynb` to train a single-genre reconstruction model.
- **Task 2 (Variational Autoencoder):**
  Run `notebooks/task2_vae.ipynb` for multi-genre diversity. (Note: Due to extreme piano-roll sparsity, the VAE exhibits posterior collapse. This is addressed by tokenization in Task 3.)
- **Task 3 (Autoregressive Transformer):**
  Run `notebooks/task3_transformer.ipynb` for the best generation quality using `miditok` REMI tokens.

## Evaluation

Evaluation metrics include Rhythm Diversity Score, Repetition Ratio, and Pitch Histogram Similarity. Outputs are saved in `outputs/plots/evaluation_results.json`.

## Generated Samples

For qualitative evaluation and playback, selected generated MIDI files from Task 1, Task 2, Task 3, and the baseline models are publicly shared here:
[Generated midi samples](https://drive.google.com/drive/folders/1vMjGSvsT4amtPq7IOp8xB_Z6ZmNBFocl?usp=drive_link)