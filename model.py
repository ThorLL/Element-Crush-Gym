import tqdm
from jax.numpy import array
from flax.nnx import Module, Rngs, Conv, Linear, one_hot, relu, MultiMetric, jit, value_and_grad, Optimizer, metrics, grad
from optax import softmax_cross_entropy_with_integer_labels, adamw

class Model(Module):
    def __init__(self, height, width, channels, action_space, learning_rate, momentum):
        rngs = Rngs(0)
        self.shape = (height, width)
        self.channels = channels

        self.conv1 = Conv(in_features=channels, out_features=32, kernel_size=(3, 3), padding='SAME', rngs=rngs)
        self.conv2 = Conv(in_features=32, out_features=64, kernel_size=(3, 2), padding='SAME', rngs=rngs)
        self.conv3 = Conv(in_features=64, out_features=128, kernel_size=(3, 2), padding='SAME', rngs=rngs)
        self.conv4 = Conv(in_features=128, out_features=128, kernel_size=(3, 2), padding='SAME', rngs=rngs)

        self.dense1 = Linear(in_features=height*width*128, out_features=512, rngs=rngs)
        self.dense2 = Linear(in_features=512, out_features=256, rngs=rngs)
        self.output_layer = Linear(in_features=256, out_features=action_space, rngs=rngs)

        self.optimizer = Optimizer(self, adamw(learning_rate, momentum))
        self.metrics = MultiMetric(
            accuracy=metrics.Accuracy(),
            loss=metrics.Average('loss')
        )


    def __call__(self, x):
        if x.shape == self.shape:
            return self.predict_on_batch(x)
        if len(x.shape) == 3 and x.shape[1] == self.shape[0] and x.shape[2] == self.shape[1]:
            return self.predict(x)
        raise TypeError(f'Invalid shape, shape must either ({self.shape[0]}, {self.shape[1]}), ({self.shape[0]}, {self.shape[1]}, {self.shape[2]}), (n_batches, {self.shape[1]}, {self.shape[2]}), or (n_batches, {self.shape[0]}, {self.shape[1]}, {self.shape[2]})')


    def predict(self, x):
        x = one_hot(x, self.channels)
        x = self.conv1(x)
        x = relu(x)

        # Conv layer 2
        x = self.conv2(x)
        x = relu(x)

        # Conv layer 3
        x = self.conv3(x)
        x = relu(x)

        # Optional Conv layer 4
        x = self.conv4(x)
        x = relu(x)

        # Flatten the output of the final convolutional layer
        x = x.reshape(x.shape[0], -1)  # Shape becomes (batch_size, 9 * 7 * 128)

        # Fully connected layer 1
        x = self.dense1(x)
        x = relu(x)

        # Fully connected layer 2
        x = self.dense2(x)
        x = relu(x)

        # Output layer: One score per possible move
        x = self.output_layer(x)

        return x  # The final output: scores for each possible move

    def predict_on_batch(self, x):
        return self.predict(x.reshape(1, *x.shape)).reshape(-1)

    @staticmethod
    def loss(logits, labels):
        return softmax_cross_entropy_with_integer_labels(logits=logits, labels=labels).mean()

    def loss_fn(self, _, batch: dict[str, list[any]]):
        x = array(batch['observations'])
        logits = self(x)
        labels = array(batch['actions'])
        loss = self.loss(logits, labels)
        return loss, logits

    def eval(self, batch: dict[str, list[any]]):
        loss, logits = self.loss_fn(self, batch)
        labels = array(batch['actions'])
        self.metrics.update(loss=loss, logits=logits, labels=labels)  # In-place updates

    def _train_step(self, batch):
        grad_fn = value_and_grad(self.loss_fn, has_aux=True, allow_int=True)
        (loss, logits), grads = grad_fn(self, batch)
        self.metrics.update(loss=loss, logits=logits, labels=array(batch['actions']))
        self.optimizer.update(grads)

    def fit(self, train_ds, test_ds, epochs=1, eval_every=100):
        metrics_history = {
            'train_loss': [],
            'train_accuracy': [],
            'test_loss': [],
            'test_accuracy': [],
        }

        for epoch in range(epochs):
            for step, batch in enumerate(train_ds):
                self._train_step(batch)

                if step > 0 and (step % eval_every == 0 or step == len(train_ds) - 1):
                    for metric, value in self.metrics.compute().items():  # Compute the metrics.
                        metrics_history[f'train_{metric}'].append(value)  # Record the metrics.
                    self.metrics.reset()  # Reset the metrics for the test set.

                    # Compute the metrics on the test set after each training epoch.
                    for test_batch in test_ds:
                        self.eval(test_batch)

                    # Log the test metrics.
                    for metric, value in self.metrics.compute().items():
                        metrics_history[f'test_{metric}'].append(value)
                    self.metrics.reset()  # Reset the metrics for the next training epoch.

                    print(
                        f"[train] step: {step}, "
                        f"loss: {metrics_history['train_loss'][-1]}, "
                        f"accuracy: {metrics_history['train_accuracy'][-1] * 100}"
                    )
                    print(
                        f"[test] step: {step}, "
                        f"loss: {metrics_history['test_loss'][-1]}, "
                        f"accuracy: {metrics_history['test_accuracy'][-1] * 100}"
                    )

