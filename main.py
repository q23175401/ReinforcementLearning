from tensorflow.python.keras.layers.core import Flatten
from keras_q_learning.keras_ddqn_agent import DDqnAgent
from keras_q_learning.keras_dqn_agent import DqnAgent
from keras_q_learning.keras_dueling_dqn import DuelingDqnAgent, DuelingDeepQNetworkModel
from keras_q_learning.keras_dueling_qqdn import DuellingDDqnAgent
import gym
from connectx_game.connectx_game_board import GameBoardEnv
from connectx_game.connectx_agents import MinimaxAgent
import tensorflow.keras.layers as L
import tensorflow as tf
import numpy as np


if __name__ == "__main__":
    
    
    #hyperparameters
    LEARNING_RATE = 0.01
    UPDATE_TARGET_NET_STEPS = 100
    epsilon = 0.5
    epsilon_decay = 0.996
    epsilon_end = 0.01

    # replay buffer for training
    MAX_BUFFER_SIZE = 100000
    MIN_DATA_TO_COLLECT = 200

    # training parameters
    BATCH_SIZE = 64
    DISCOUNT_FACTOR = 0.99
    N_EPISODES = 50000


    env = gym.make('CartPole-v0')
    # env = GameBoardEnv(MinimaxAgent(1))
    n_actions  = env.action_space.n
    input_shape = env.observation_space.shape

    # def build_qnet(num_output, input_shape, lr):
    #     layer_list = [
    #         L.InputLayer(input_shape=input_shape),
    #         L.Flatten(),
    #         L.Dense(128, activation='relu'),
    #         L.Dense(128, activation='relu'),
    #         L.Dense(num_output, activation='linear'),
    #     ]
    #     model = tf.keras.Sequential(layer_list)
    #     model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(learning_rate=lr), metrics=['accuracy'])
    #     return model

    def build_qnet(num_output, input_shape, fc1, fc2, lr):
        model = DuelingDeepQNetworkModel(num_output, input_shape, fc1, fc2)
        model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(learning_rate=lr), metrics=['accuracy'])
        return model

    # my_model = build_qnet(num_output, input_shape, 128, 128, LEARNING_RATE)
    # test_input = tf.random.uniform(shape=[1] + list(input_shape))
    # ori_output = my_model.predict(test_input)
    # print(ori_output)

    agent = DuellingDDqnAgent(
        n_actions           = n_actions,
        input_shape         = input_shape,
        learning_rate       = LEARNING_RATE,
        update_target_steps = UPDATE_TARGET_NET_STEPS,
        epsilon             = epsilon,
        epsilon_decay       = epsilon_decay,
        epsilon_end         = epsilon_end,
        max_buffer_size     = MAX_BUFFER_SIZE,
        min_data_to_collect = MIN_DATA_TO_COLLECT,
        q_network           = build_qnet(n_actions, input_shape, 128, 128, LEARNING_RATE),
    )

    agent.collect_random_samples(env)
    agent.train(
        collect_env     = env,
        N_EPISODES      = N_EPISODES,
        batch_size      = BATCH_SIZE,
        DISCOUNT_FACTOR = DISCOUNT_FACTOR,
        show_result     = 10,
    )
