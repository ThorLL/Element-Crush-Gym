from match3tile.env import Match3Env
from model import Model
from trainingDataGen import get_train_and_test_data


if __name__ == '__main__':
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
