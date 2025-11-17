```
# Neural Network Training with TensorBoard

Simple neural network implementation using TensorFlow/Keras with TensorBoard logging for visualization.

## Setup

\`\`\`bash
pip install tensorflow numpy
\`\`\`

## What It Does

- Creates a synthetic dataset of 10,000 samples
- Splits data into 80% training, 20% testing
- Trains a simple Sequential neural network with:
  - Dense input layer (16 units)
  - Dense hidden layer (1 unit)
- Logs training metrics to TensorBoard for visualization

## Usage

1. Run the training script:
\`\`\`python
python train.py
\`\`\`

2. Launch TensorBoard:
\`\`\`bash
tensorboard --logdir=logs/scalars
\`\`\`

3. View results at \`http://localhost:6006\`

## Model Architecture

- Input: 2 features
- Hidden Layer: 16 units (Dense)
- Output Layer: 1 unit (Dense)
- Loss: Mean Squared Error (MSE)
- Optimizer: SGD (learning rate: 0.2)

## Training Configuration

- Epochs: 20
- Batch size: train_size
- Validation split: 20%
- TensorBoard callback enabled

## Output

Training logs are saved to \`logs/scalars/YYYYMMDD-HHMMSS/\` with timestamped directories for each run.
```
