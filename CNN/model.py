from layers import Conv2D, ReLU, MaxPool2D, Flatten, Dense


class SimpleCNN:
    """Small CNN model for Fashion MNIST.

    Architecture:
    Conv2D -> ReLU -> MaxPool2D -> Flatten -> Dense
    """

    def __init__(self):
        self.layers = [
            Conv2D(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1),
            ReLU(),
            MaxPool2D(pool_size=2, stride=2),
            Flatten(),
            Dense(input_size=8 * 14 * 14, output_size=10),
        ]

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, d_out):
        for layer in reversed(self.layers):
            d_out = layer.backward(d_out)

    def update(self, learning_rate):
        for layer in self.layers:
            layer.update(learning_rate)
