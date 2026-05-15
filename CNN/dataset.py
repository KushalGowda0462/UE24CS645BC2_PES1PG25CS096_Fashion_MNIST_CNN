import numpy as np
from torchvision.datasets import FashionMNIST
from torchvision import transforms


def load_fashion_mnist(data_dir="./data", train_samples=None, test_samples=None):
    """Load Fashion MNIST and return NumPy arrays.

    Torchvision is used only to download/load the dataset.
    CNN operations are implemented manually using NumPy.
    """

    transform = transforms.ToTensor()

    train_dataset = FashionMNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform,
    )

    test_dataset = FashionMNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform,
    )

    x_train = train_dataset.data.numpy().astype(np.float32) / 255.0
    y_train = train_dataset.targets.numpy().astype(np.int64)

    x_test = test_dataset.data.numpy().astype(np.float32) / 255.0
    y_test = test_dataset.targets.numpy().astype(np.int64)

    # Add channel dimension: (N, 28, 28) -> (N, 1, 28, 28)
    x_train = x_train[:, None, :, :]
    x_test = x_test[:, None, :, :]

    if train_samples is not None:
        x_train = x_train[:train_samples]
        y_train = y_train[:train_samples]

    if test_samples is not None:
        x_test = x_test[:test_samples]
        y_test = y_test[:test_samples]

    return x_train, y_train, x_test, y_test
