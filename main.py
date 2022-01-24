from keras_q_learning import DDqnAgent, DqnAgent, DuelingDqnAgent, DuelingDeepQNetworkModel, DuellingDDqnAgent
import gym
from connectx_game import connectx_agents, GameBoardEnv

# from connectx_game.connectx_agents import MinimaxAgent
import tensorflow.keras.layers as L
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import numpy as np


def prepare_agent_env():

    # hyperparameters
    LEARNING_RATE = 0.1
    UPDATE_TARGET_NET_STEPS = 900
    epsilon = 1
    epsilon_decay = 0.999
    epsilon_end = 0.01

    # replay buffer for training
    MAX_BUFFER_SIZE = 100000
    MIN_DATA_TO_COLLECT = 2000

    env = gym.make("LunarLander-v2")
    # env = gym.make('CartPole-v0')
    # env = GameBoardEnv(connectx_agents.MinimaxAgent(1))

    n_actions = env.action_space.n
    input_shape = env.observation_space.shape

    def build_qnet(num_output, input_shape, lr):
        layer_list = [
            L.InputLayer(input_shape=input_shape),
            # L.Conv2D(16, (3, 3), padding="same", activation="tanh"),
            # L.Conv2D(16, (3, 3), padding="same", activation="tanh"),
            # L.Conv2D(8, (3, 3), padding="same", activation="tanh"),
            L.Flatten(),
            L.Dense(32, activation="tanh"),
            L.Dense(32, activation="tanh"),
            L.Dense(num_output, activation="linear"),
        ]
        # layer_list = [
        #     L.InputLayer(input_shape=input_shape),
        #     L.Conv2D(16, (3, 3), padding="same", activation="tanh"),
        #     L.Conv2D(16, (3, 3), padding="same", activation="tanh"),
        #     L.Conv2D(8, (3, 3), padding="same", activation="tanh"),
        #     L.Flatten(),
        #     L.Dense(32, activation="tanh"),
        #     L.Dense(32, activation="tanh"),
        #     L.Dense(num_output, activation="linear"),
        # ]
        model = tf.keras.Sequential(layer_list)
        model.compile(loss="mean_squared_error", optimizer=Adam(learning_rate=lr), metrics=["accuracy"])
        return model

    # def build_qnet(num_output, input_shape, lr):
    #     model = DuelingDeepQNetworkModel(num_output, input_shape, 128, 128)
    #     model.trainable = True
    #     model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(
    #         learning_rate=lr), metrics=['accuracy'])
    #     return model

    agent_config = dict(
        n_actions=n_actions,
        input_shape=input_shape,
        learning_rate=LEARNING_RATE,
        update_target_steps=UPDATE_TARGET_NET_STEPS,
        epsilon=epsilon,
        epsilon_decay=epsilon_decay,
        epsilon_end=epsilon_end,
        max_buffer_size=MAX_BUFFER_SIZE,
        min_data_to_collect=MIN_DATA_TO_COLLECT,
        build_q_network=lambda: build_qnet(n_actions, input_shape, LEARNING_RATE),
    )

    # agent = DuellingDDqnAgent(**agent_config)
    agent = DDqnAgent(**agent_config)

    agent.collect_random_samples(env)
    return agent, env


def train_agent(agent, env, batch_size=32, discount=0.99, n_episodes=5000):
    # training parameters
    BATCH_SIZE = batch_size
    DISCOUNT_FACTOR = discount
    N_EPISODES = n_episodes

    agent.train(
        collect_env=env,
        N_EPISODES=N_EPISODES,
        batch_size=BATCH_SIZE,
        DISCOUNT_FACTOR=DISCOUNT_FACTOR,
        show_result=5,
    )


if __name__ == "__main__":
    agent, env = prepare_agent_env()
    train_agent(agent, env, 32, 0.99, 1000000)
