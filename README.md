
This repository contains the implementation of an **Unbiased Label Distribution Learning** framework, designed to address label distribution biases in tasks such as sound source localization in circular spaces. The project leverages deep neural networks (DNNs) and PyTorch Lightning for training and evaluation, with a focus on handling noisy and reverberant audio data.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Configuration](#configuration)
- [Results](#results)
- [File Structure](#file-structure)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Overview
The `SoundLoc-EQ` project aims to develop a robust model for learning label distributions without bias, particularly in circular spatial contexts (e.g., 360-degree sound source localization). It integrates advanced encoding techniques (e.g., Gaussian smoothing) and distributed training to handle large-scale audio datasets like `Libri_Circular_4` and `Libri_adhoc_nodes10_splited`.

This project is now complete, with all core functionalities (training, evaluation, and data processing) fully implemented and tested.

The primary contribution of this work is the **plug-and-play `encoding_decoding.py` module**, which significantly enhances deep learning-based sound source localization. This module introduces two key functions:
- **`unbiased_encoding`**: Mitigates quantization errors during the training phase by applying unbiased label encoding with Gaussian smoothing.
- **`weighted_adjacent_decoding`**: Reduces decoding errors by employing a weighted adjacent decoding strategy, ensuring accurate localization in circular spaces.


## Features
- **Unbiased Encoding**: Implements `unbiased_encoding` with Gaussian smoothing to mitigate label bias.
- **Circular Space Support**: Configurable for 360-degree spatial tasks with a resolution of 72 cells (5 degrees each).
- **Distributed Training**: Supports multi-GPU training using PyTorch Lightning's DDP strategy.
- **Early Stopping and Checkpointing**: Ensures efficient training with minimal overfitting.
- **Comprehensive Evaluation**: Includes scripts for testing on diverse noisy and clean datasets.

## Installation
### Clone the Repository
```bash
git clone https://github.com/MetaAcoustix/SoundLoc-EQ.git
cd SoundLoc-EQ
```

### Install Dependencies
Ensure you have Python 3.7 or 3.8 installed.

```bash
pip install torch && ...
```
### Set Up GPU Environment (Optional)
Install CUDA-compatible PyTorch if using GPUs:
```bash
pip install torch --extra-index-url https://download.pytorch.org/whl/cu118
```

Configure available GPUs in `config.py` (e.g., `gpus = [5]`).

### Prepare Datasets
Download or prepare datasets (e.g., `Libri_Circular_4`, `Libri_adhoc_nodes10_splited`) and place them in the specified directories (see [Dataset](#dataset)).

## Usage
### Training
Run the training script with the following command:
```bash
python train.py
```
This will train the SS2net model on the `Libri_Circular_4` dataset using configurations from `config.py` (e.g., `epochs_max=40`, `learning_rate=5e-4`).

Checkpoints and logs are saved in `./circular/checkpoints/` and `./circular/logs/`, respectively.

### Evaluation
Run the evaluation script to test the model:
```bash
python eval.py
```
This evaluates the PhaseModel on the `Libri_adhoc_nodes10_splited` test dataset using a pre-trained checkpoint (e.g., `1hot_72.ckpt`).

### Customization
- Modify `config.py` to adjust hyperparameters (e.g., `sigma=8`, `angle_range=360`).
- Update dataset paths in `train.py` and `eval.py` if using different data sources.

## Dataset
The project uses the following datasets:

- **Training/Validation**: `Libri_Circular_4`
  - Located at `/home/liincuan/datasets_simu/Libri_Circular_4/train2/feats` and `val2/feats`.
- **Testing**: `Libri_adhoc_nodes10_splited`
  - Located at `/home/liincuan/datasets/Libri_adhoc_nodes10_splited/room2/test2/feats`.
- **Testing Samples**:
  - `testing_samples/` includes subdirectories:
    - `-10DB`: Test data with -10 dB noise.
    - `-20DB`: Test data with -20 dB noise.
    - `0DB`: Test data with 0 dB noise.
    - `clean`: Clean test data.
    - `reverb`: Test data with reverberation effects.

Datasets should be preprocessed into `.pt` files (PyTorch tensors) and placed in the respective directories.

## Configuration
Key configurations are defined in `config.py`:
```python
model_type = 'dnn'  # Uses a deep neural network
epochs_max = 40  # Maximum training epochs
gpus = [5]  # GPU configuration (supports single or multiple GPUs)
angle_range = 360  # Circular space range
cell_reso = 72  # Grid resolution (5 degrees per cell)
sigma = 8  # Gaussian smoothing parameter
space = 'circular'  # Spatial configuration
strategy = 'ddp'  # Distributed data parallel training
```

## Results
The project has been successfully completed with the following outcomes:
- **Training**: Achieved convergence on the `Libri_Circular_4` dataset with early stopping (`patience=10`) and checkpointing based on `val_mae`.
- **Evaluation**: Demonstrated robust performance on the `Libri_adhoc_nodes10_splited` test set using the `1hot_72.ckpt` checkpoint.
- **Performance**: Improved label distribution learning with unbiased encoding, particularly in noisy and reverberant conditions.

Detailed logs and checkpoints are available in `./circular/logs/` and `./circular/checkpoints/`.

## File Structure
```
SoundLoc-EQ/
├── config.py              # Configuration settings
├── dataset.py             # Dataset loading and preprocessing
├── encoding_decoding.py   # Encoding functions (e.g., onehot_encoding, unbiased_encoding)
├── eval.py                # Evaluation script
├── train.py               # Training script
├── model.py               # Model definitions (e.g., SS2net, PhaseModel)
├── README.md              # Project documentation
└── testing_samples/       # Test data subdirectories (-10DB, -20DB, 0DB, clean, reverb)
   ├── -10DB/
   ├── -20DB/
   ├── 0DB/
   ├── clean/
   └── reverb/

```

## Contributing
This project is now complete, and no further contributions are actively sought. However, if you have suggestions or improvements:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/<feature-name>`).
3. Commit your changes (`git commit -m "Add some feature"`).
4. Push to the branch (`git push origin feature/<feature-name>`).
5. Open a Pull Request with a clear description.

Please adhere to the Conventional Commits specification for commit messages.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

Special thanks to the dataset providers for `Libri_Circular_4` and `Libri_adhoc_nodes10_splited`.
Inspired by research in unbiased label distribution learning and sound source localization.
