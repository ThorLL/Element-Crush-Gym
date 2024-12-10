import os
import jax
import numpy as np
import orbax.checkpoint as ocp
from flax import nnx

from elementGO.MCTSModel import Model
import match3tile

CKPT_TEST_PATH = os.getcwd() + "/test_models"
CKPT_PATH = os.getcwd() + "/models"
checkpointer = ocp.PyTreeCheckpointer()


def save(model: Model, name: str = "state", testing: bool = False):
    """
    Saves the model to the checkpoint path (/models)
    - model: The model to save
    - name: The name of the model to save
    - testing: If true, it wipes the testing directory and saves the model there (/test_models/[name])
    """
    state = nnx.state(model)

    if testing:
        ocp.test_utils.erase_and_create_empty(CKPT_TEST_PATH)
        checkpointer.save(f"{CKPT_TEST_PATH}/{name}", state)
    else:
        checkpointer.save(f"{CKPT_PATH}/{name}", state)


def load(name: str = "state", testing: bool = False) -> Model:
    """
    Loads a specified model from the checkpoint path (/models)
    - name: The name of the model to load
    - testing: If True, loads the test model from the test directory (/test_models/test_state)
    """
    model = nnx.eval_shape(lambda: abstract_model())
    state = nnx.state(model)

    path = CKPT_TEST_PATH if testing else CKPT_PATH
    state = checkpointer.restore(f"{path}/{name}", state)

    nnx.update(model, state)
    return model


def abstract_model():
    """Creates a default model used for loading models"""
    return Model(
        match3tile.metadata.rows,
        match3tile.metadata.columns,
        match3tile.metadata.action_space,
        channels=match3tile.metadata.types,
        features=64,
        learning_rate=0.005,
        momentum=0.9,
    )


def compare_models(model1, model2):
    """A check to see if two models are the same"""
    state1 = nnx.state(model1)
    state2 = nnx.state(model2)

    try:
        jax.tree.map(np.testing.assert_array_equal, state1, state2)
    except AssertionError:
        return False
    return True
