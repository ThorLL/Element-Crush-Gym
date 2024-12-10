import jax
import numpy as np
import orbax.checkpoint as ocp
from flax import nnx

from elementGO.MCTSModel import Model
from match3tile.env import Match3Env


def save_model(
    model: Model, filename: str = "state", validate: bool = True, verbose: bool = False
):
    """
    Saves a models state to a file.
        - model: The model to save
        - name: The name of the file
        - validate: If True, the model is saved and then loaded again to check if the state is the same
    """

    # TODO: Don't use erase_and_create_empty. Use a proper Path object (pathlib)
    path = ocp.test_utils.erase_and_create_empty("/tmp/my-checkpoints/")

    _, state = nnx.split(model.state)
    if verbose:
        print("NNX State to save: ")
        nnx.display(state)

    checkpointer = ocp.StandardCheckpointer()
    checkpointer.save(path / filename, state)
    print(f"Model saved to {path / filename}")

    if validate:
        env = Match3Env()
        abstract_state = nnx.eval_shape(
            lambda: Model(
                action_space=env.action_space,
                channels=model.channels,
                features=256,
                learning_rate=0.005,
                momentum=0.9,
            )
        )
        _, state_restored = nnx.split(
            checkpointer.restore(path / "state", abstract_state)
        )
        jax.tree.map(np.testing.assert_array_equal, state, state_restored)


def load_model(path, abstract_model=None, verbose=False) -> Model:
    """
    Loads a model from a file.
        - path: The path to the file
        - abstract_model: The abstract model to use. Should be created using the same parameters as the model that was saved. If None is passed an abstract model will be created using default variables
    """

    checkpointer = ocp.StandardCheckpointer()

    if abstract_model is None:
        abstract_model = create_abstract_model()

    graphdef, abstract_state = nnx.split(abstract_model)
    if verbose:
        print("The abstract NNX model (all leaves are abstract arrays):")
        nnx.display(abstract_model)

    state_restored = checkpointer.restore(path / "state", abstract_state)
    if verbose:
        print("NNX State restored: ")
        nnx.display(state_restored)

    model = nnx.merge(graphdef, state_restored)
    return model


def create_abstract_model(
    width=9, height=9, num_types=6, features=256, learning_rate=0.005, momentum=0.9
):
    """
    Creates an abstract model with the given parameters used for loading a model.
    If model parameters isn't given, default values are used.
    """

    env = Match3Env(width=width, height=height, num_types=num_types)
    height, width, channels = env.observation_space

    return nnx.eval_shape(
        lambda: Model(
            action_space=env.action_space,
            channels=channels,
            features=features,
            learning_rate=learning_rate,
            momentum=momentum,
        )
    )


# Using the example from Flax repository rather than their documentation
def save(model: Model, path: str, filename: str = "state"):
    state = nnx.state(model)
    checkpointer = ocp.PyTreeCheckpointer()
    checkpointer.save(f"{path}/{filename}", state)


def load(path: str, filename: str = "state") -> Model:
    # create that model with abstract shapes
    model = nnx.eval_shape(lambda: create_model())
    state = nnx.state(model)

    checkpointer = ocp.PyTreeCheckpointer()
    state = checkpointer.restore(f"{path}/{filename}", state)

    nnx.update(model, state)
    return model


def create_model():
    env = Match3Env()

    model = Model(
        action_space=env.action_space,
        channels=6,
        features=256,
        learning_rate=0.005,
        momentum=0.9,
    )

    return model

def validate():
    model_to_save = create_model()
    path = "/tmp/my-checkpoints"
    filename = "state"

    save(model_to_save, path, filename)

    model_loaded = load(path, filename)