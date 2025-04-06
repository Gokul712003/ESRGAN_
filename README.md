# Satellite Image Super-Resolution using ESRGAN


## Project Overview

This repository contains the implementation of Enhanced Super-Resolution Generative Adversarial Network (ESRGAN) for satellite image enhancement. The model is designed to improve the quality and resolution of satellite imagery without requiring upscaling, maintaining the same dimensions (256x256) while enhancing details and clarity.

## Dataset

The dataset used in this project is from [Kaggle: Super-Resolution using GANs and Satellite Images](https://www.kaggle.com/code/saraivaufc/super-resolution-using-gans-and-satellite-images/data).

### Dataset Preparation

1. Original data was in H5 format
2. Using `dataset_extractor.ipynb`, the data was extracted and converted to JPEG images
3. Images were contrast-enhanced for better visualization
4. Dataset was split into input (lower quality) and output (higher quality) image pairs

## Implementation Details

This repository contains two implementations of ESRGAN:

### Standard ESRGAN (`ESRgan_.ipynb`)
- Full implementation with Dense Residual Blocks
- More feature-rich but requires more GPU memory
- Uses VGG19 for perceptual loss calculation
- Includes checkpoint saving and visualization

## Model Architecture

The ESRGAN model consists of:
- Generator with Residual-in-Residual Dense Blocks (RRDB)
- Discriminator with VGG-style architecture
- VGG19-based perceptual loss network

## Requirements

- TensorFlow 2.x (2.10 preferred for GPU support)
- Python 3.7+
- NumPy
- Matplotlib
- PIL
- scikit-image
- h5py (for dataset extraction)

## Usage

1. **Dataset Extraction**:
   ```
   python dataset_extractor.ipynb
   ```

2. **Training**:

   ```
   python ESRgan_.ipynb  # For standard version
   ```


## Acknowledgments

- Dataset from [Kaggle](https://www.kaggle.com/code/saraivaufc/super-resolution-using-gans-and-satellite-images/data)
- Implementation based on the [ESRGAN paper](https://arxiv.org/abs/1809.00219)

