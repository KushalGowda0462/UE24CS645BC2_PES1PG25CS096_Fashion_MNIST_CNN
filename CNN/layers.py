import numpy as np


class Conv2D:
    """2D convolution layer implemented from scratch using NumPy.

    Input shape:  (batch_size, in_channels, height, width)
    Output shape: (batch_size, out_channels, out_height, out_width)
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # He initialization works well with ReLU.
        scale = np.sqrt(2.0 / (in_channels * kernel_size * kernel_size))
        self.weights = np.random.randn(
            out_channels, in_channels, kernel_size, kernel_size
        ) * scale
        self.biases = np.zeros(out_channels)

        self.input = None
        self.padded_input = None
        self.d_weights = None
        self.d_biases = None

    def forward(self, x):
        self.input = x
        batch_size, in_channels, height, width = x.shape

        if self.padding > 0:
            self.padded_input = np.pad(
                x,
                ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)),
                mode="constant",
            )
        else:
            self.padded_input = x

        padded_height, padded_width = self.padded_input.shape[2], self.padded_input.shape[3]

        out_height = (padded_height - self.kernel_size) // self.stride + 1
        out_width = (padded_width - self.kernel_size) // self.stride + 1

        output = np.zeros((batch_size, self.out_channels, out_height, out_width))

        for n in range(batch_size):
            for f in range(self.out_channels):
                for i in range(out_height):
                    for j in range(out_width):
                        h_start = i * self.stride
                        h_end = h_start + self.kernel_size
                        w_start = j * self.stride
                        w_end = w_start + self.kernel_size

                        region = self.padded_input[n, :, h_start:h_end, w_start:w_end]
                        output[n, f, i, j] = np.sum(region * self.weights[f]) + self.biases[f]

        return output

    def backward(self, d_out):
        batch_size, _, out_height, out_width = d_out.shape

        d_padded_input = np.zeros_like(self.padded_input)
        self.d_weights = np.zeros_like(self.weights)
        self.d_biases = np.zeros_like(self.biases)

        for n in range(batch_size):
            for f in range(self.out_channels):
                for i in range(out_height):
                    for j in range(out_width):
                        h_start = i * self.stride
                        h_end = h_start + self.kernel_size
                        w_start = j * self.stride
                        w_end = w_start + self.kernel_size

                        region = self.padded_input[n, :, h_start:h_end, w_start:w_end]

                        self.d_weights[f] += d_out[n, f, i, j] * region
                        self.d_biases[f] += d_out[n, f, i, j]
                        d_padded_input[n, :, h_start:h_end, w_start:w_end] += (
                            d_out[n, f, i, j] * self.weights[f]
                        )

        # Average gradients over batch for stable updates.
        self.d_weights /= batch_size
        self.d_biases /= batch_size

        if self.padding > 0:
            return d_padded_input[
                :, :, self.padding:-self.padding, self.padding:-self.padding
            ]

        return d_padded_input

    def update(self, learning_rate):
        self.weights -= learning_rate * self.d_weights
        self.biases -= learning_rate * self.d_biases


class ReLU:
    """Rectified Linear Unit activation function."""

    def __init__(self):
        self.input = None

    def forward(self, x):
        self.input = x
        return np.maximum(0, x)

    def backward(self, d_out):
        return d_out * (self.input > 0)

    def update(self, learning_rate):
        pass


class MaxPool2D:
    """Max pooling layer implemented from scratch."""

    def __init__(self, pool_size=2, stride=2):
        self.pool_size = pool_size
        self.stride = stride
        self.input = None

    def forward(self, x):
        self.input = x
        batch_size, channels, height, width = x.shape

        out_height = (height - self.pool_size) // self.stride + 1
        out_width = (width - self.pool_size) // self.stride + 1

        output = np.zeros((batch_size, channels, out_height, out_width))

        for n in range(batch_size):
            for c in range(channels):
                for i in range(out_height):
                    for j in range(out_width):
                        h_start = i * self.stride
                        h_end = h_start + self.pool_size
                        w_start = j * self.stride
                        w_end = w_start + self.pool_size

                        region = x[n, c, h_start:h_end, w_start:w_end]
                        output[n, c, i, j] = np.max(region)

        return output

    def backward(self, d_out):
        batch_size, channels, out_height, out_width = d_out.shape
        d_input = np.zeros_like(self.input)

        for n in range(batch_size):
            for c in range(channels):
                for i in range(out_height):
                    for j in range(out_width):
                        h_start = i * self.stride
                        h_end = h_start + self.pool_size
                        w_start = j * self.stride
                        w_end = w_start + self.pool_size

                        region = self.input[n, c, h_start:h_end, w_start:w_end]
                        max_value = np.max(region)
                        mask = region == max_value

                        d_input[n, c, h_start:h_end, w_start:w_end] += (
                            d_out[n, c, i, j] * mask
                        )

        return d_input

    def update(self, learning_rate):
        pass


class Flatten:
    """Converts 4D CNN output into 2D matrix for dense layer."""

    def __init__(self):
        self.original_shape = None

    def forward(self, x):
        self.original_shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, d_out):
        return d_out.reshape(self.original_shape)

    def update(self, learning_rate):
        pass


class Dense:
    """Fully connected layer."""

    def __init__(self, input_size, output_size):
        scale = np.sqrt(2.0 / input_size)
        self.weights = np.random.randn(input_size, output_size) * scale
        self.biases = np.zeros((1, output_size))

        self.input = None
        self.d_weights = None
        self.d_biases = None

    def forward(self, x):
        self.input = x
        return np.dot(x, self.weights) + self.biases

    def backward(self, d_out):
        batch_size = self.input.shape[0]

        self.d_weights = np.dot(self.input.T, d_out) / batch_size
        self.d_biases = np.sum(d_out, axis=0, keepdims=True) / batch_size

        d_input = np.dot(d_out, self.weights.T)
        return d_input

    def update(self, learning_rate):
        self.weights -= learning_rate * self.d_weights
        self.biases -= learning_rate * self.d_biases


class SoftmaxCrossEntropyLoss:
    """Softmax activation combined with cross entropy loss."""

    def __init__(self):
        self.probabilities = None
        self.labels = None

    def forward(self, logits, labels):
        self.labels = labels

        # Numerical stability trick.
        shifted_logits = logits - np.max(logits, axis=1, keepdims=True)
        exp_scores = np.exp(shifted_logits)
        self.probabilities = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        batch_size = logits.shape[0]
        correct_log_probs = -np.log(self.probabilities[np.arange(batch_size), labels] + 1e-12)
        loss = np.mean(correct_log_probs)

        return loss

    def backward(self):
        batch_size = self.probabilities.shape[0]
        d_logits = self.probabilities.copy()
        d_logits[np.arange(batch_size), self.labels] -= 1
        return d_logits
