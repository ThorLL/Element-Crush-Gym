import math
import random
import time

from tqdm import tqdm

from match3tile.board import Board
import numpy as np


class Node:
    def __init__(self, state, moves, game_score, parent=None, config=(0, 6)):
        self.state = np.copy(state)
        self.parent: Node = parent
        self.children = []
        self.visits = 0
        self.reward = 0
        self.moves_left = moves
        self.game_score = game_score
        self.seed, self.types = config
        self.actions = []

    def best_child(self, c):
        return max(self.children, key=lambda child: child.ucb1(c))

    def step(self, action, deterministic=False):
        if deterministic:
            np.random.seed(self.seed)
        else:
            np.random.seed(random.randint(0, 2**32 - 1))
        self.state = Board.swap_tokens(action, self.state)

        tokens_matched = 0
        chain_matches = 0

        actions = []
        while True:
            matches = Board.get_matches(self.state)
            if len(matches) == 0 and len(actions) == 0:
                pass
            if len(matches) == 0:
                break
            chain_matches += len(matches)
            tokens_matched += sum([len(match) for match in matches])
            self.state = Board.remove_matches(matches, self.state)
            self.state = Board.drop(self.state)

            while True:
                safe_cpy = np.copy(self.state)
                zero_mask = (self.state == 0)
                random_values = np.random.randint(1, self.types + 1, size=self.state.shape, dtype=np.int32)
                self.state[zero_mask] = random_values[zero_mask]
                self.actions = Board.valid_actions(self.state)
                if len(self.actions) > 0:
                    break
                else:
                    self.state = safe_cpy
        self.game_score += tokens_matched + tokens_matched * (chain_matches - 1)

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
    def __init__(self, env, simulations=100, verbal=True):
        self.root = Node(
            state=np.copy(env.board.array),
            moves=env.num_moves - env.moves_taken,
            game_score=env.score,
            config=(env.seed, env.num_types)
        )
        self.root.actions = env.board.actions

        self.simulations = simulations

        self.goal = env.env_goal
        self.root.expand()

        self.times = {
            'tree traversal ': 0,
            'expand         ': 0,
            'rollout        ': 0,
            'backpropagation': 0,
        }

        self.verbal = verbal

        del env

    def print_times(self):
        print('MCTS TIME TABLE')
        print('      Step      - Total - Avg pr call')
        for name, v in self.times.items():
            avg = self.times[name] / self.simulations
            v = "%.2f" % v
            if len(v) == 4:
                v = '0'+v
            print(name, v, "%.2f" % avg, sep=' - ')

    def simulation_step(self):
        node = self.tree_traversal(self.root)
        if node.visits > 0:
            begin = time.time()
            node.expand()
            node = node.children[0]
            self.times['expand         '] += time.time() - begin
        reward = self.rollout(node)
        self.backpropagation(node, reward)

    def __call__(self):
        if self.verbal:
            for _ in tqdm(range(self.simulations)):
                self.simulation_step()
        else:
            for _ in range(self.simulations):
                self.simulation_step()

        next_root = self.root.best_child(0)
        next_root_idx = self.root.children.index(next_root)
        action = self.root.actions[next_root_idx]
        next_root.parent = None
        # clean up and prep for next call
        self.root.children.remove(next_root)
        del self.root
        self.root = next_root

        return action

    def tree_traversal(self, node: Node):
        begin = time.time()
        while len(node.children) != 0:
            node = node.best_child(3)
        self.times['tree traversal '] += time.time() - begin
        return node

    def rollout(self, node):
        begin = time.time()

        node_state = np.copy(node.state)
        node_actions = node.actions
        node_score = node.game_score

        for i in range(node.moves_left):  # Limit the depth of simulation
            node.step(random.choice(node.actions))

        reward = node.game_score - node_score
        node.state = node_state
        node.actions = node_actions
        node.game_score = node_score
        self.times['rollout        '] += time.time() - begin
        return reward

    def backpropagation(self, node, reward):
        begin = time.time()
        while node:
            node.visits += 1
            node.reward += reward
            node = node.parent
        self.times['backpropagation'] += time.time() - begin
