import flax.nnx as nnx


class ConvLayer(nnx.Module):
    def __init__(self, in_features, out_features, kernel_size=(3, 3), strides=(1, 1), rng=None):
        super().__init__()
        rng = rng or nnx.Rngs(0)
        self.conv = nnx.Conv(in_features, out_features, kernel_size, strides, rngs=rng)
        self.bn = nnx.BatchNorm(out_features, rngs=rng)

    def __call__(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return nnx.relu(x)


class PolicyHead(nnx.Module):
    def __init__(self, in_features, height, width, action_space, rng=None):
        super().__init__()
        rng = rng or nnx.Rngs(0)
        self.conv = nnx.Conv(in_features, out_features=2, kernel_size=(1, 1), strides=(1, 1), rngs=rng)
        self.bn = nnx.BatchNorm(num_features=2, rngs=rng)
        self.dense = nnx.Linear(in_features=2 * height * width, out_features=action_space, rngs=rng)

    def __call__(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = nnx.relu(x)
        x = x.reshape(x.shape[0], -1)
        x = self.dense(x)
        return x


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


class ValueHead(nnx.Module):
    def __init__(self, in_features, height, width, features, rng=None):
        super().__init__()
        rng = rng if rng else nnx.Rngs(0)
        self.conv = nnx.Conv(in_features, 1, (1, 1), (1, 1), rngs=rng)
        self.bn = nnx.BatchNorm(num_features=1, rngs=rng)
        self.dense1 = nnx.Linear(height * width, out_features=features, rngs=rng)
        self.dense2 = nnx.Linear(in_features=features, out_features=1, rngs=rng)

    def __call__(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = nnx.relu(x)
        x = x.reshape(x.shape[0], -1)
        x = nnx.relu(self.dense1(x))
        x = nnx.relu(self.dense2(x))
        return x