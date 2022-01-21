from abc import ABC, abstractmethod
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as L
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from .q_learning_replay_buffer import OneTrajectoryData, MyReplayBuffer
from typing import Callable

class QBaseAgent(ABC):
    def __init__(self, n_actions, input_shape, learning_rate, update_target_steps, epsilon, epsilon_decay, epsilon_end, max_buffer_size, min_data_to_collect, build_q_network: Callable=None):
        self.n_actions = n_actions
        self.input_shape = input_shape

        self.train_step_counter = 0
        self.learning_rate = learning_rate
        self.UPDATE_TARGET_NET_STEPS = update_target_steps
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_end = epsilon_end

        # replay buffer for training
        self.MAX_BUFFER_SIZE = max_buffer_size
        self.MIN_DATA_TO_COLLECT = min_data_to_collect

        # change build_q_network function
        self.build_my_qnet = build_q_network or self.build_my_qnet
        self.value_net  = self.build_my_qnet()
        self.target_net = self.build_my_qnet()
        self.update_target_net() # pass value net's weights to target net

        self.replay_buffer = MyReplayBuffer(self.MAX_BUFFER_SIZE)

    def get_action(self, one_state):
        return np.argmax(self.value_net.predict(np.array([one_state])), axis=1)[0]

    def choose_action(self, collect_env, one_state): #choose action by epsilon greedy method
        if np.random.sample() > self.epsilon: 
            action = self.get_action(one_state)
        else:
            action = collect_env.action_space.sample()

        self.epsilon *= self.epsilon_decay
        self.epsilon = self.epsilon_end if self.epsilon <= self.epsilon_end else self.epsilon

        return action


    def update_target_net(self):
        self.target_net.set_weights(self.value_net.get_weights())

    def train(self, collect_env, N_EPISODES, batch_size=64, DISCOUNT_FACTOR=0.99, show_result=10):
        
        scores = []
        for i_episode in range(N_EPISODES):
            RENDER_RESULT = i_episode % show_result == 0 if show_result>0 else False

            done = False
            accumulated_reward = 0
            state = collect_env.reset()
            while not done:
                action = self.choose_action(collect_env, state)

                if RENDER_RESULT:
                    collect_env.render()

                next_state, reward, done, info = collect_env.step(action)
                accumulated_reward += reward
                self.store_transition(state, action, reward, next_state, done)

                if len(self.replay_buffer) > self.MIN_DATA_TO_COLLECT: # 要超過了最少的量才開訓練
                
                    train_ok = self.train_one_batch(batch_size, DISCOUNT_FACTOR)

                    if train_ok: # 有訓練到才改變counter
                        self.train_step_counter += 1
                        if self.train_step_counter >= self.UPDATE_TARGET_NET_STEPS:
                            self.train_step_counter = 0
                            self.update_target_net()

                state = next_state

            scores.append(accumulated_reward)

            if RENDER_RESULT:
                avg_score = np.mean(scores[-show_result:])
                print(f'Episode {i_episode} end with {accumulated_reward} time steps',
                    'average score', avg_score)

        self.plot_result(N_EPISODES, scores)
        collect_env.close()

    @abstractmethod
    def train_one_batch(self, batch_size, DISCOUNT_FACTOR) -> bool:
        pass

    def build_my_qnet(self):
        layer_list = [
            L.InputLayer(input_shape=self.input_shape),
            L.Flatten(),
            L.Dense(128, activation='relu'),
            L.Dense(128, activation='relu'),
            L.Dense(self.n_actions, activation='linear'),
        ]
        model = tf.keras.Sequential(layer_list)
        model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=self.learning_rate), metrics=['accuracy'])
        return model

    def plot_result(self, n_episodes, scores):
        print('plot training history, not implement yet')

    def evaluate_policy(self, eval_env, test_episode=10):
        if test_episode<0: return

        total_acc_reward = 0
        for _ in range(test_episode):

            done = False
            state = eval_env.reset()
            accumulated_reward = 0
            while not done:
                action = self.get_action(state)
                next_state, reward, done, info = eval_env.step(action)
                accumulated_reward += reward
                state = next_state
        
            total_acc_reward += accumulated_reward
        print(f'Evaluate {test_episode} episodes and get {total_acc_reward/test_episode} average steps')

    def collect_random_samples(self, env, num_samples=-1):
        collect_samples = num_samples if num_samples>0 else self.MIN_DATA_TO_COLLECT
        
        collect_counter = 0
        while collect_counter<collect_samples:

            done = False
            state = env.reset()
            accumulated_reward = 0
            while not done:
                action = env.action_space.sample()

                next_state, reward, done, info = env.step(action)
                accumulated_reward += reward

                self.store_transition(state, action, reward, next_state, done)

                collect_counter += 1

    def store_transition(self, state, action, reward, next_state, done):
        one_trajectory_data = \
            OneTrajectoryData(
                cur_state  = state,
                action     = action,
                reward     = reward,
                next_state = next_state,
                done       = done
            )

        self.replay_buffer.push_trajectory(one_trajectory_data)
    
    def save_agent(self, file_name):
        self.value_net.save(file_name)
    
    def load_agent(self, file_name):
        self.value_net = load_model(file_name)
        self.target_net = load_model(file_name)

