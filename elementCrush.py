import math
import os
from typing import Optional

import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax import tree
from optax import sgd
from tqdm import tqdm
import orbax.checkpoint as ocp


import elementGOModules as elmGO
from match3tile import metadata
from util.prompter import ask_for, chose
from visualisers.plotter import LivePlotter


CKPT_PATH = os.getcwd() + "/models"
checkpointer = ocp.PyTreeCheckpointer()


@nnx.jit
def value_loss(hat_value, values):
    return jnp.mean((hat_value - values) ** 2)


@nnx.jit
def policy_loss(hat_policy, policies):
    log_prob = nnx.log_softmax(hat_policy, axis=-1)
    return -jnp.sum(policies * log_prob, axis=-1).mean()


@nnx.jit
def l2_regularization(model, alpha=1e-4):
    state = nnx.state(model)
    weights = (tree.leaves(state['conv']) +
               tree.leaves(state['residual_block']) +
               tree.leaves(state['value_head']) +
               tree.leaves(state['policy_head']))

    def l2_loss(x):
        return alpha * (x ** 2).sum()

    l2 = tree.map(lambda w: l2_loss(w), weights)
    l2 = jnp.array(l2).sum()
    return l2


class ElementCrush(nnx.Module):
    def __init__(self,
                 residual_layer=40,
                 features=256,
                 optimizer=sgd(1e-5, 0.9, nesterov=False),
                 rng=None
                 ):
        super().__init__()
        rng = rng or nnx.Rngs(0)

        height, width, action_space = metadata.rows, metadata.columns, metadata.action_space

        types = metadata.types

        self.channels = 2 ** (int(math.ceil(math.log2(types))) + 2)

        self.conv = elmGO.ConvLayer(self.channels, out_features=features, rng=rng)

        self.residual_block = elmGO.ResidualBlock(features, residual_layer, rng=rng)

        self.value_head = elmGO.ValueHead(features, height, width, features, rng=rng)
        self.policy_head = elmGO.PolicyHead(features, height, width, action_space, rng=rng)

        self.optimizer = nnx.Optimizer(self, optimizer)
        self.metrics = nnx.MultiMetric(
            loss=nnx.metrics.Average('loss'),
            value_loss=nnx.metrics.Average('value_loss'),
            policy_loss=nnx.metrics.Average('policy_loss'),
            regularization=nnx.metrics.Average('regularization'),
            value_MAE=nnx.metrics.Average('value_MAE'),
            policy_MAE=nnx.metrics.Average('policy_MAE'),
        )

        self.to_string = lambda: f'elementCrush/{height}x{width}x{types}/{residual_layer}_{features}'

    def to_string(self):
        pass

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
        if plot: plotter.save(self.to_string())

    def save(self, suffix=None, force: bool = False):
        suffix = '_' + suffix if suffix else ''
        name = f'{CKPT_PATH}/{self.to_string()}{suffix}'
        try:
            checkpointer.save(name, nnx.state(self), force=force)
        except ValueError as e:
            if not force:
                if 'yes' == ask_for('Model with same params already exists, overwrite?', ['yes', 'no']):
                    self.save(suffix, True)
            else:
                raise e

    @staticmethod
    def load(file_name=None) -> 'ElementCrush':
        if file_name is None:
            models_folder = f'{CKPT_PATH}/elementCrush/'
            model_groups = os.listdir(models_folder)
            model_groups = [model_group for model_group in model_groups if os.path.isdir(models_folder+model_group)]
            if len(model_groups) > 1:
                model_group = chose('Chose shape', model_groups)
            else:
                model_group = model_groups[0]

            model_group = models_folder+model_group
            models = [model for model in os.listdir(model_group) if os.path.isdir(model_group)]
            file_name = model_group + '/' + chose('Chose model', models)

        # extract abstract model from file
        _, shape, model = tuple(file_name.split('elementCrush')[-1].split('/'))

        h, w, t = tuple(shape.split('x'))
        metadata.set_shape(int(h), int(w))
        metadata.set_types(int(t))

        layers, feats = tuple(model.split('_'))
        model = nnx.eval_shape(lambda: ElementCrush(int(layers), int(feats)))
        state = nnx.state(model)

        state = checkpointer.restore(file_name, state)

        nnx.update(model, state)
        return model

    def __eq__(self, other: Optional['ElementCrush']) -> bool:
        if other is None:
            return False

        state1 = nnx.state(self)
        state2 = nnx.state(other)

        try:
            tree.map(np.testing.assert_array_equal, state1, state2)
        except AssertionError:
            return False
        return True
