import flax.nnx as nnx


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
