import math
import random

from tqdm import tqdm

from match3tile.board import Board
import numpy as np


class Node:
    def __init__(self, state: Board, moves, game_score, parent=None, config=(0, 6)):
        self.state: Board = state.clone()
        self.parent: Node = parent
        self.children = []
        self.visits = 0
        self.reward = 0
        self.moves_left = moves
        self.game_score = game_score
        self.seed, self.types = config
        self.actions = self.state.actions

    def best_child(self, c):
        return max(self.children, key=lambda child: child.ucb1(c))

    def step(self, action, deterministic=True):
        if deterministic:
            np.random.seed(self.seed)
        else:
            np.random.seed(random.randint(0, 2**32 - 1))

        score, _ = self.state.swap(action)
        self.actions = self.state.actions
        self.game_score += score

    def ucb1(self, c) -> float:
        if self.visits == 0:
            return float('inf')
        return self.reward / self.visits + c * math.sqrt(math.log(self.parent.visits / self.visits))

    def expand(self):
        for action in self.actions:
            child_node = Node(self.state, self.moves_left-1, self.game_score, self, (self.seed, self.types))
            child_node.step(action, True)
            self.children.append(child_node)

    def __del__(self):
        for i in range(len(self.children)):
            del self.children[0]
        del self.state
        del self.parent
        del self


class MCTS:
    def __init__(self, env, simulations=100, verbal=True, deterministic=False):
        self.deterministic = deterministic
        self.root = Node(
            state=env.board,
            moves=env.num_moves - env.moves_taken,
            game_score=env.score,
            config=(env.seed, env.num_types)
        )

        self.simulations = simulations

        self.goal = env.env_goal
        self.root.expand()

        self.verbal = verbal

    def __call__(self):
        action, _, _ = self.analyse()
        return action

    def analyse(self):
        if self.verbal:
            for _ in tqdm(range(self.simulations)):
                self.simulation_step()
        else:
            for _ in range(self.simulations):
                self.simulation_step()

        next_root = self.root.best_child(0)

        policy = [child.ucb1(0) for child in self.root.children]
        value = next_root.ucb1(0)

        next_root_idx = self.root.children.index(next_root)
        action = self.root.actions[next_root_idx]
        next_root.parent = None
        # clean up and prep for next call
        self.root.children.remove(next_root)
        del self.root
        self.root = next_root

        return action, value, policy

    def simulation_step(self):
        node = self.tree_traversal(self.root)
        if node.visits > 0:
            node.expand()
            node = node.children[0]
        reward = self.rollout(node)
        self.backpropagation(node, reward)

    def tree_traversal(self, node: Node):
        while len(node.children) != 0:
            node = node.best_child(3)
        return node

    def rollout(self, node):
        node_state = node.state.clone()
        node_actions = node.actions
        node_score = node.game_score

        # Ensures that the simulation is deterministic
        if self.deterministic:
            np.random.seed(self.root.seed)
            for i in range(node.moves_left):  # Limit the depth of simulation
                node.step(np.random.choice(node.state.actions), True)
        else:
            # Ensures that the simulation is non-deterministic (also change the deterministic boolean above):
            for i in range(node.moves_left):  # Limit the depth of simulation
                node.step(random.choice(node.state.actions))

        if node.game_score >= self.goal:
            diff = node.game_score - self.goal
            c = 5
            node.game_score += diff ** c

        reward = node.game_score - node_score
        node.state = node_state
        node.actions = node_actions
        node.game_score = node_score
        return reward

    def backpropagation(self, node, reward):
        while node:
            node.visits += 1
            node.reward += reward
            node = node.parent
