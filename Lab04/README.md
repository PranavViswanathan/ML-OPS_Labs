# CNN Image Classifier with Weights & Biases

A simple CNN for CIFAR-10 image classification with experiment tracking using WandB.

## What It Does

Trains a 3-layer convolutional neural network to classify images from CIFAR-10 into 10 categories (airplanes, cars, birds, cats, etc.).

## Requirements

```bash
pip install torch torchvision wandb
```

## Usage

1. **Login to WandB** (first time only):
```bash
wandb login
```

2. **Run the script**:
```bash
python train.py
```

3. **View results**: Check your WandB dashboard for real-time metrics, loss curves, and model parameters.

## Model Architecture

- 3 convolutional layers (32→64→128 filters)
- Max pooling after each conv layer
- 2 fully connected layers (512→10)
- Dropout for regularization

## What Gets Logged

- Training & validation loss/accuracy per epoch
- Batch-level metrics every 100 steps
- Model gradients and parameters
- Final model weights as an artifact

## Configuration

Default hyperparameters (edit in code):
- Learning rate: 0.001
- Batch size: 64
- Epochs: 10
- Optimizer: Adam

## Output

The trained model is saved as `model.pth` and uploaded to WandB as a versioned artifact.
