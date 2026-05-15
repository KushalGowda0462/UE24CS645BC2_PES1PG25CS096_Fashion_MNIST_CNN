import numpy as np


def accuracy(logits, labels):
    predictions = np.argmax(logits, axis=1)
    return np.mean(predictions == labels) * 100


def create_batches(x, y, batch_size, shuffle=True):
    indices = np.arange(len(x))

    if shuffle:
        np.random.shuffle(indices)

    for start in range(0, len(x), batch_size):
        end = start + batch_size
        batch_indices = indices[start:end]
        yield x[batch_indices], y[batch_indices]


def class_names():
    return [
        "T-shirt/top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle boot",
    ]
