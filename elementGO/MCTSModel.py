import math

import flax.nnx as nnx
from optax import sgd
from tqdm import tqdm

from elementGO.blocks.conv import ConvLayer
from elementGO.blocks.policyHead import PolicyHead
from elementGO.blocks.residual import ResidualBlock
from elementGO.blocks.valueHead import ValueHead
from elementGO.util import policy_loss, value_loss, l2_regularization
from util.plotter import LivePlotter


class Model(nnx.Module):
    def __init__(self, height, width, action_space, channels, features, learning_rate, momentum, rng=None):
        super().__init__()
        rng = rng or nnx.Rngs(0)

        self.channels = 2 ** (int(math.ceil(math.log2(channels))) + 2)

        self.conv = ConvLayer(self.channels,  out_features=features, rng=rng)

        self.residual_block = ResidualBlock(features, num_layers=5, rng=rng)

        self.value_head = ValueHead(features, height, width, rng=rng)
        self.policy_head = PolicyHead(features, height, width, action_space, rng=rng)

        self.optimizer = nnx.Optimizer(self, sgd(learning_rate, momentum, nesterov=False))
        self.metrics = nnx.MultiMetric(
            loss=nnx.metrics.Average('loss'),
            value_loss=nnx.metrics.Average('value_loss'),
            policy_loss=nnx.metrics.Average('policy_loss'),
            regularization=nnx.metrics.Average('regularization'),
            value_MAE=nnx.metrics.Average('value_MAE'),
            policy_MAE=nnx.metrics.Average('policy_MAE'),
        )

    @nnx.jit
    def __call__(self, x):
        x = nnx.one_hot(x, self.channels)
        x = self.conv(x)
        x = self.residual_block(x)
        value = self.value_head(x)
        policy = self.policy_head(x)
        return value, policy

    @nnx.jit
    def loss(self, model, observation, values, policies):
        value, policy = self(observation)
        p_loss = policy_loss(policy, policies)
        v_loss = value_loss(value, values)
        r = l2_regularization(model)
        total_loss = v_loss + p_loss + r
        return total_loss, (v_loss, p_loss, r, (value, policy))

    @nnx.jit
    def update_metrics(self, aux_data, values, policies):
        (v_loss, p_loss, r, (value, policy)) = aux_data
        self.metrics.update(
            loss=v_loss + p_loss + r,
            value_loss=v_loss,
            policy_loss=p_loss,
            regularization=r,
            value_MAE=(values - value).mean(),
            policy_MAE=(policies - policy).mean()
        )

    @nnx.jit
    def eval(self, observation, values, policies):
        """Evaluate the model on the batch and update metrics."""
        _, aux_data = self.loss(self, observation, values, policies)
        self.update_metrics(aux_data, values, policies)

    @nnx.jit
    def training_step(self, observation, values, policies):
        grad_fn = nnx.value_and_grad(self.loss, has_aux=True)
        (_, aux_data), grads = grad_fn(self, observation, values, policies)
        self.optimizer.update(grads)
        self.update_metrics(aux_data, values, policies)

    def train(self, train_ds, test_ds, epochs, eval_every, plot=True):
        print('starting training')

        if plot:
            plotter = LivePlotter()
            for label in self.metrics._metric_names:
                plot = plotter.add_view('steps', label)
                plot.add_plot(f'train_{label}', x_step=eval_every)
                plot.add_plot(f'test_{label}', x_step=eval_every)
            plotter.build()

        with tqdm(total=epochs * len(train_ds)) as pbar:
            for epoch in range(epochs):
                for step, batch in enumerate(train_ds):
                    self.training_step(batch['observations'], batch['values'], batch['policies'])
                    pbar.n += 1
                    pbar.refresh()
                    if step % eval_every == 0 and step != 0:
                        
                        if plot:
                            for metric, value in self.metrics.compute().items():
                                plotter.add_value_for(f'train_{metric}', value)

                        self.metrics.reset()
                        for test_batch in test_ds:
                            self.eval(test_batch['observations'], test_batch['values'], test_batch['policies'])

                        if plot:
                            for metric, value in self.metrics.compute().items():
                                plotter.add_value_for(f'test_{metric}', value)
                        self.metrics.reset()

                        if plot: plotter.update()

        if plot: plotter.show()
