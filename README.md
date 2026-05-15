# UE24CS645BC2_<USN>_Fashion_MNIST_CNN

## Demystify Convolutional Neural Networks - Fashion MNIST CNN From Scratch

This project implements a basic Convolutional Neural Network from first principles using NumPy.  
The model is trained and evaluated on the Fashion MNIST dataset.

The goal of this assignment is to understand how CNN layers work internally, including:

- Convolution operation
- Convolution layer forward pass
- Convolution layer backward pass
- Max pooling layer
- Flattening layer
- Fully connected layer
- Activation functions
- Softmax cross entropy loss
- Backpropagation
- Training and evaluation

> Note: PyTorch/Torchvision is used only for downloading/loading the Fashion MNIST dataset.  
> The CNN layers, forward pass, backward pass, loss, and training logic are implemented manually using NumPy.

---

## Dataset

Fashion MNIST contains 70,000 grayscale images of clothing and accessories.

- Training images: 60,000
- Test images: 10,000
- Image size: 28 x 28
- Number of classes: 10

Classes:

| Label | Class |
|---|---|
| 0 | T-shirt/top |
| 1 | Trouser |
| 2 | Pullover |
| 3 | Dress |
| 4 | Coat |
| 5 | Sandal |
| 6 | Shirt |
| 7 | Sneaker |
| 8 | Bag |
| 9 | Ankle boot |

---

## Project Structure

```text
UE24CS645BC2_<USN>_Fashion_MNIST_CNN/
│
├── README.md
├── requirements.txt
├── .gitignore
│
└── src/
    ├── layers.py
    ├── model.py
    ├── dataset.py
    ├── train.py
    └── utils.py
```

---

## How the CNN Works

The implemented CNN follows this flow:

```text
Input Image
   ↓
Convolution Layer
   ↓
ReLU Activation
   ↓
Max Pooling Layer
   ↓
Flatten Layer
   ↓
Fully Connected Layer
   ↓
Softmax Classifier
   ↓
Prediction
```

---

## Model Architecture

```text
Input: 1 x 28 x 28 grayscale image

Conv2D:
- Input channels: 1
- Output channels: 8
- Kernel size: 3 x 3
- Stride: 1
- Padding: 1

ReLU

MaxPool2D:
- Pool size: 2 x 2
- Stride: 2

Flatten

Dense:
- Input size: 8 x 14 x 14 = 1568
- Output size: 10 classes

Softmax Cross Entropy Loss
```

---

## Installation

Create a virtual environment:

```bash
python -m venv venv
```

Activate it:

### Windows

```bash
venv\Scripts\activate
```

### macOS / Linux

```bash
source venv/bin/activate
```

Install requirements:

```bash
pip install -r requirements.txt
```

---

## How to Run

Run training using a small subset first:

```bash
python src/train.py --epochs 3 --train-samples 2000 --test-samples 500 --batch-size 32
```

For a larger run:

```bash
python src/train.py --epochs 5 --train-samples 10000 --test-samples 2000 --batch-size 32
```

---

## Output

The program prints:

- Epoch number
- Training loss
- Training accuracy
- Test accuracy

Example:

```text
Epoch 1/3
Train Loss: 1.9243 | Train Acc: 35.25% | Test Acc: 42.60%

Epoch 2/3
Train Loss: 1.4128 | Train Acc: 52.40% | Test Acc: 56.20%

Epoch 3/3
Train Loss: 1.1037 | Train Acc: 63.15% | Test Acc: 64.80%
```

Actual results may vary depending on the number of training samples, epochs, and system speed.

---

## Explanation of Important Files

### `layers.py`

Contains the manually implemented neural network layers:

- `Conv2D`
- `ReLU`
- `MaxPool2D`
- `Flatten`
- `Dense`
- `SoftmaxCrossEntropyLoss`

Each trainable layer has:

- `forward()`
- `backward()`
- `update()`

### `model.py`

Builds the CNN by arranging layers in the correct order.

### `dataset.py`

Loads Fashion MNIST using Torchvision and converts it into NumPy arrays.

### `train.py`

Controls the full training pipeline:

1. Load dataset
2. Build model
3. Run forward pass
4. Calculate loss
5. Run backward pass
6. Update parameters
7. Evaluate model

### `utils.py`

Contains helper functions like accuracy calculation and mini-batch creation.

---

## Why CNN is Useful for Fashion MNIST

A CNN is suitable for image classification because it can learn local visual patterns such as:

- Edges
- Curves
- Shapes
- Clothing outlines
- Texture-like structures

The convolution layer detects useful features, pooling reduces spatial size, and the fully connected layer performs final classification.

---


