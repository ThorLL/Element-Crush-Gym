from MCTS import MCTS
from match3tile.env import Match3Env
from model import Model
from trainingDataGen import get_train_and_test_data


def mcts(moves, goal):
    mcts_env = Match3Env(num_moves=moves, env_goal=goal)
    best_env = Match3Env(num_moves=moves, env_goal=goal)
    naive_env = Match3Env(num_moves=moves, env_goal=goal)
    random_env = Match3Env(num_moves=moves, env_goal=goal)

    mcts_score = [0, 0]
    best_move_score = 0
    naive_score = 0
    random_score = 0
    done, won = False, False

    mcts_algo = MCTS(mcts_env)

    won_at = 0

    while mcts_env.moves_taken != mcts_env.num_moves:
        mcts_action, required_next_state = mcts_algo()

        best_action = best_env.board.best_action()
        naive_action = naive_env.board.naive_action()
        random_action = random_env.board.random_action()

        idx = 1 if won else 0
        _, reward, done, won, _ = mcts_env.step(mcts_action)
        mcts_score[idx] += reward

        _, reward, _, _, _ = best_env.step(best_action)
        best_move_score += reward

        _, reward, _, _, _ = naive_env.step(naive_action)
        naive_score += reward

        _, reward, _, _, _ = random_env.step(random_action)
        random_score += reward

        if not won:
            won_at += 1
    if done:
        if won:
            print('Won game after', won_at, 'moves')
        else:
            print('Lost game')
        print('mcts actions score:', mcts_score, sum(mcts_score))
        print('best actions score:', best_move_score)
        print('naive actions score:', naive_score)
        print('random actions score:', random_score)
        print('')

        mcts_algo.print_times()
    return mcts_score, best_move_score, naive_score, random_score


def train_model():
    env = Match3Env()

    height, width, channels = env.observation_space
    model = Model(
        height=height,
        width=width,
        channels=channels,
        action_space=env.action_space,
        learning_rate=0.005,
        momentum=0.9,
    )

    train_ds, test_ds = get_train_and_test_data()
    model.fit(train_ds, test_ds)


if __name__ == '__main__':
    mcts(20, 1000)

