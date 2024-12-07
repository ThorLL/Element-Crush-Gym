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
