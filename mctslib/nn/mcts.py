from collections.abc import Callable
from typing import Optional
import jax.numpy as jnp

from elementCrush import ElementCrush
from mctslib.abc.mcts import BaseMCTS, State, BaseNode


class NNNode(BaseNode):
    def __init__(self,
                 state: State,
                 policy_fn: Callable[[jnp.array], tuple[float, jnp.array]],
                 parent: Optional['NNNode'] = None,
                 policy: float = 1.0
                 ):
        super().__init__(state, parent)
        self.policy = policy

        self.policy_fn = policy_fn
        _, child_policy = policy_fn(jnp.array([state.array]))
        self.child_action_probs = {i: p for i, p in enumerate(child_policy.flatten()) if i in state.legal_actions}
        
        self.untried_actions = list(self.child_action_probs.keys())

    def ucb1(self, c: float):
        return super().ucb1(c * self.policy)

    @property
    def is_fully_expanded(self) -> bool:
        return len(self.untried_actions) == 0

    def expand(self) -> 'NNNode':
        action = self.untried_actions.pop()
        next_state = self.state.apply_action(action)
        child_node = NNNode(
            state=next_state,
            policy_fn=self.policy_fn,
            parent=self,
            policy=self.child_action_probs[action]
        )
        self.children[action] = child_node
        return child_node


class NeuralNetworkMCTS(BaseMCTS):
    def __init__(self, model: ElementCrush, state: State, exploration_weight: float, simulations: int, verbose: bool):
        self.model = model
        root = NNNode(state, self.model.__call__)

        super().__init__(root, exploration_weight, simulations, verbose)

    def rollout(self, state: State) -> float:
        if state.is_terminal:
            return state.reward
        value, _ = self.model(jnp.array([state.array]))
        return value
