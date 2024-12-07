import flax.nnx as nnx


class ValueHead(nnx.Module):
    def __init__(self, in_features, rng=None):
        super().__init__()
        rng = rng if rng else nnx.Rngs(0)
        self.conv = nnx.Conv(in_features, 1, (1, 1), (1, 1), rngs=rng)
        self.bn = nnx.BatchNorm(num_features=1, rngs=rng)
        self.dense1 = nnx.Linear(in_features, out_features=256, rngs=rng)
        self.dense2 = nnx.Linear(in_features=256, out_features=1, rngs=rng)

    def __call__(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = nnx.relu(x)
        x = x.reshape(x.shape[0], -1)
        x = nnx.relu(self.dense1(x))
        x = nnx.tanh(self.dense2(x))
        return x
