import math
from abc import ABC, abstractmethod
from typing import Optional, List, Any

from tqdm import tqdm


class State(ABC):
    @property
    @abstractmethod
    def legal_actions(self) -> List[Any]:
        raise NotImplementedError

    @abstractmethod
    def apply_action(self, action) -> 'State':
        raise NotImplementedError

    @property
    @abstractmethod
    def is_terminal(self) -> bool:
        raise NotImplementedError

    @property
    @abstractmethod
    def reward(self) -> float:
        raise NotImplementedError

    @abstractmethod
    def clone(self):
        raise NotImplementedError


class BaseNode(ABC):
    def __init__(self, state: State, parent: Optional['BaseNode'] = None):
        self.state: State = state.clone()
        self.parent: BaseNode = parent
        self.children: dict[Any, BaseNode] = {}
        self.visits = 0
        self.reward = 0

    @property
    @abstractmethod
    def is_fully_expanded(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def expand(self) -> 'BaseNode':
        raise NotImplementedError

    @property
    def policies(self):
        return [child.visits / self.visits for child in self.children.values()]

    def update(self, reward: float):
        self.visits += 1
        self.reward += reward

    def best_child(self, c):
        return max(self.children.values(), key=lambda child: child.ucb1(c))

    @property
    def exploitation(self):
        return self.reward / self.visits

    @property
    def exploration(self):
        return math.sqrt(math.log(self.parent.visits) / (1 + self.visits))

    def ucb1(self, c: float):
        if self.visits == 0:
            return float('inf')
        return self.exploitation + c * self.exploration


class BaseMCTS(ABC):
    def __init__(self, root: BaseNode, exploration_weight: float, simulations: int, verbose: bool):
        self._root = root
        self._simulations = simulations
        self._verbose = verbose
        self._exploration_weight = exploration_weight

        self._root.expand()

    def __call__(self):
        # create progress par
        if self._verbose:
            pbar = tqdm(total=self._simulations)

        # core mcts logic
        node = self._root
        for iteration in range(self._simulations):
            # Selection
            while not node.state.is_terminal and node.is_fully_expanded:
                node = node.best_child(self._exploration_weight)

            # Expansion
            if not (node.state.is_terminal or node.is_fully_expanded):
                node = node.expand()

            # Simulation
            reward = self.rollout(node.state)

            # Backpropagation
            while node is not None:
                node.update(reward)
                node = node.parent
            node = self._root
            if self._verbose:
                pbar.n += 1
                pbar.refresh()

        # Select best child
        best_child = max(self._root.children.values(), key=lambda c: c.visits)

        action = 0
        for a, child in self._root.children.items():
            if best_child == child:
                action = a
                break

        policies = best_child.policies
        value = best_child.reward

        # update root for next call
        self._root = best_child

        # dispose of pbar
        if self._verbose:
            del pbar

        return action, value, policies

    @abstractmethod
    def rollout(self, state: State) -> float:
        raise NotImplemented
