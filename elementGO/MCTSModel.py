import flax.nnx as nnx
import jax.numpy as jnp

from matplotlib import pyplot as plt

from optax import sgd
from tqdm import tqdm
from jax.nn import log_softmax
from jax.tree_util import tree_leaves

from elementGO.blocks.conv import ConvLayer
from elementGO.blocks.policyHead import PolicyHead
from elementGO.blocks.residual import ResidualBlock
from elementGO.blocks.valueHead import ValueHead
from util.plotter import LivePlotter


class Model(nnx.Module):
    def __init__(self, action_space, channels, features, learning_rate, momentum, rng=None):
        super().__init__()
        rng = rng or nnx.Rngs(0)
        self.channels = channels
        self.conv = ConvLayer(channels,  out_features=256, rng=rng)

        self.residual_block = ResidualBlock(features, num_layers=40, rng=rng)

        self.value_head = ValueHead(features, rng=rng)
        self.policy_head = PolicyHead(features, action_space, rng=rng)

        self.optimizer = nnx.Optimizer(self, sgd(learning_rate, momentum, nesterov=False))
        self.metrics = nnx.MultiMetric(
            accuracy=nnx.metrics.Accuracy(),
            loss=nnx.metrics.Average('loss'),
            value_loss=nnx.metrics.Average('value_loss'),
            policy_loss=nnx.metrics.Average('policy_loss'),
            regularization=nnx.metrics.Average('regularization')
        )

    def __call__(self, x):
        x = nnx.one_hot(x, self.channels)
        x = self.conv(x)
        x = self.residual_block(x)
        value = self.value_head(x)
        policy = self.policy_head(x)
        return value, policy

    def loss(self, W, batch):
        x = jnp.array(batch['observations'])
        value, policy = self(x)
        policy_loss = self.policy_loss(policy, batch)
        value_loss = self.value_loss(value, batch)
        regularization = self.regularization(W)
        return value_loss, policy_loss, regularization, policy

    def value_loss(self, hat_value, batch):
        true_value = batch['values']
        return jnp.mean((hat_value - true_value) ** 2)

    def policy_loss(self, hat_policy, batch):
        true_policy = batch['policies']
        log_prob = log_softmax(hat_policy, axis=-1)
        return -jnp.sum(true_policy * log_prob, axis=-1).mean()

    def regularization(self, W, c=1e-4):
        return c * sum(jnp.sum(jnp.square(w)) for w in tree_leaves(W))

    def eval(self, batch: dict[str, list[any]]):
        """Evaluate the model on the batch and update metrics."""
        x = jnp.array(batch['observations'])
        value, policy = self(x)
        policy_loss = self.policy_loss(policy, batch)
        value_loss = self.value_loss(value, batch)

        total_loss = value_loss + policy_loss

        self.metrics.update(
            loss=total_loss,
            value_loss=value_loss,
            policy_loss=policy_loss,

            logits=policy,
            labels=jnp.array(batch['policies'])
        )

    def training_step(self, batch):
        grad_fn = nnx.value_and_grad(self.loss, has_aux=False)
        (value_loss, policy_loss, regularization, policy), grads = grad_fn(self, batch)
        loss = value_loss + policy_loss + regularization

        self.optimizer.update(grads)

        self.metrics.update(
            loss=loss,
            value_loss=value_loss,
            policy_loss=policy_loss,

            logits=policy,
            labels=jnp.array(batch['policies'])
        )

        return loss

    def train(self, train_ds, test_ds, epochs=2, eval_every=1):
        metrics_history = {
            'train_loss': [],
            'train_accuracy': [],
            'test_loss': [],
            'test_accuracy': [],
        }

        plotter = LivePlotter()
        loss_view = plotter.add_view('Steps', 'Loss')
        training_loss = loss_view.add_plot('Train Loss')
        testing_loss = loss_view.add_plot('Test Loss')

        accuracy_view = plotter.add_view('Steps', 'Accuracy')
        training_accuracy = accuracy_view.add_plot('Train Accuracy')
        testing_accuracy = accuracy_view.add_plot('Test Accuracy')

        # Set up live plotting
        plt.ion()
        with tqdm(total=epochs * len(train_ds)) as pbar:
            for epoch in range(epochs):
                for step, batch in enumerate(train_ds):

                    self.training_step(batch)

                    if (step * (1 + epoch)) % eval_every == 0:
                        for metric, value in self.metrics.compute().items():  # Compute the metrics.
                            metrics_history[f'train_{metric}'].append(value)  # Record the metrics.

                        self.metrics.reset()  # Reset the metrics for the test set.

                        # Compute the metrics on the test set after each training epoch.
                        for test_batch in test_ds[:10]:
                            self.eval(test_batch)

                        # Log the test metrics.
                        for metric, value in self.metrics.compute().items():
                            metrics_history[f'test_{metric}'].append(value)
                        self.metrics.reset()  # Reset the metrics for the next training epoch.

                        # Update plots
                        training_loss.set_y_data(metrics_history['train_loss'], eval_every)
                        testing_loss.set_y_data(metrics_history['test_loss'], eval_every)
                        training_accuracy.set_y_data([acc * 100 for acc in metrics_history['train_accuracy']], eval_every)
                        testing_accuracy.set_y_data([acc * 100 for acc in metrics_history['test_accuracy']], eval_every)

                    pbar.n += 1
                    pbar.refresh()

        plotter.show()
