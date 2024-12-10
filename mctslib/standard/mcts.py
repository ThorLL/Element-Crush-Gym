import random
from typing import Optional

import numpy as np

from mctslib.abc.mcts import BaseMCTS, State, BaseNode


class MCTS(BaseMCTS):
    def __init__(self, state: State, exploration_weight: float, simulations: int, verbose: bool, deterministic=False):
        super().__init__(Node(state), exploration_weight, simulations, verbose)
        self.deterministic = deterministic

    def rollout(self, state: State) -> float:
        np.random.seed(state.seed if self.deterministic else random.randint(0, 2 ** 31 - 1))
        while not state.is_terminal:
            action = np.random.choice(state.legal_actions)
            state = state.apply_action(action)
        return state.reward


class Node(BaseNode):
    def __init__(self, state: State, parent: Optional['Node'] = None):
        super().__init__(state, parent)
        self.untried_actions = list(state.legal_actions)

    @property
    def is_fully_expanded(self) -> bool:
        return len(self.untried_actions) == 0

    def expand(self) -> 'Node':
        # Pick random action
        action = self.untried_actions.pop()
        # self.untried_actions.remove(action)

        # create next state and child
        next_state = self.state.apply_action(action)
        child_node = Node(next_state, self)

        # add child to children list
        self.children[action] = child_node
        return child_node
