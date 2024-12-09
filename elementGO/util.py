import flax.nnx as nnx
import jax.numpy as jnp
import jax.tree as tree


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
