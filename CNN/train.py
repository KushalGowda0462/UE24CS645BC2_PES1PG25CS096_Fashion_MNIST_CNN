import argparse
import numpy as np
from tqdm import tqdm

from dataset import load_fashion_mnist
from layers import SoftmaxCrossEntropyLoss
from model import SimpleCNN
from utils import accuracy, create_batches


def evaluate(model, x_test, y_test, batch_size):
    total_correct = 0
    total_samples = 0

    for x_batch, y_batch in create_batches(x_test, y_test, batch_size, shuffle=False):
        logits = model.forward(x_batch)
        predictions = np.argmax(logits, axis=1)
        total_correct += np.sum(predictions == y_batch)
        total_samples += len(y_batch)

    return (total_correct / total_samples) * 100


def train(args):
    np.random.seed(args.seed)

    print("Loading Fashion MNIST dataset...")
    x_train, y_train, x_test, y_test = load_fashion_mnist(
        train_samples=args.train_samples,
        test_samples=args.test_samples,
    )

    print(f"Training samples: {len(x_train)}")
    print(f"Test samples: {len(x_test)}")

    model = SimpleCNN()
    loss_function = SoftmaxCrossEntropyLoss()

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_samples = 0

        progress_bar = tqdm(
            create_batches(x_train, y_train, args.batch_size, shuffle=True),
            total=int(np.ceil(len(x_train) / args.batch_size)),
            desc=f"Epoch {epoch + 1}/{args.epochs}",
        )

        for x_batch, y_batch in progress_bar:
            # Forward pass
            logits = model.forward(x_batch)

            # Loss calculation
            loss = loss_function.forward(logits, y_batch)

            # Backward pass
            d_logits = loss_function.backward()
            model.backward(d_logits)

            # Parameter update
            model.update(args.learning_rate)

            # Metrics
            batch_acc = accuracy(logits, y_batch)
            epoch_loss += loss * len(y_batch)
            epoch_correct += np.sum(np.argmax(logits, axis=1) == y_batch)
            epoch_samples += len(y_batch)

            progress_bar.set_postfix({
                "loss": f"{loss:.4f}",
                "acc": f"{batch_acc:.2f}%"
            })

        train_loss = epoch_loss / epoch_samples
        train_accuracy = (epoch_correct / epoch_samples) * 100
        test_accuracy = evaluate(model, x_test, y_test, args.batch_size)

        print()
        print(f"Epoch {epoch + 1}/{args.epochs}")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_accuracy:.2f}% | Test Acc: {test_accuracy:.2f}%")
        print("-" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CNN from scratch on Fashion MNIST")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--train-samples", type=int, default=2000)
    parser.add_argument("--test-samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    train(args)
