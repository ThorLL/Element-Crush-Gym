import flax.nnx as nnx
import jax.lax


class ResidualLayer(nnx.Module):
    def __init__(self, features, kernel_size=(3, 3), strides=(1, 1), rng=None):
        super().__init__()
        rng = rng or nnx.Rngs(0)
        self.conv1 = nnx.Conv(features, features, kernel_size, strides, rngs=rng)
        self.bn1 = nnx.BatchNorm(features, rngs=rng)
        self.conv2 = nnx.Conv(features, features, kernel_size, strides, rngs=rng)
        self.bn2 = nnx.BatchNorm(features, rngs=rng)

    def __call__(self, x):
        residual = x  # Save skip connection
        x = self.conv1(x)
        x = self.bn1(x)
        x = nnx.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = x + residual  # Add skip connection
        x = nnx.relu(x)
        return x


class ResidualBlock(nnx.Module):
    def __init__(self, features, num_layers, kernel_size=(3, 3), strides=(1, 1), rng=None):
        super().__init__()
        rng = rng or nnx.Rngs(0)
        self.residual_layers = [ResidualLayer(features, kernel_size, strides, rng=rng) for _ in range(num_layers)]

    def __call__(self, x):
        for layer in self.residual_layers:
            x = layer(x)
        return x
