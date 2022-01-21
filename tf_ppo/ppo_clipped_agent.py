from collections import namedtuple
import tensorflow as tf
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.backend import dtype
from tensorflow.python.keras.layers.serialization import deserialize
from tensorflow.python.ops import ragged
from tensorflow.python.util.nest import _yield_sorted_items
import tensorflow_probability as tfp
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from ppo_actor_model import PPOActorModel
from ppo_critic_model import PPOCriticModel
from ppo_replay_buffer import PPOReplayBuffer  
from ppo_env_worker import PPOEnvWorker

class PPOAgent():
    def __init__(self,
            policy_network: PPOActorModel,
            critic_network: PPOCriticModel,
            gamma=0.99,
            gae_lambda=0.95,
        ):

        self.gamma          = gamma
        self.gae_lambda     = gae_lambda

        self.policy         = policy_network
        self.critic         = critic_network
        self.replay_buffer  = PPOReplayBuffer()

    def train(self, env, iterations, total_steps, n_epoches, batch_size, show_score=True):
        env_workwer = PPOEnvWorker(env) # using one worker to collect experiences

        for iteration in range(iterations):
            env_workwer.collect_trajectories(self.replay_buffer, self.policy, total_steps)
            a_loss, c_loss = self.learn(n_epoches, batch_size)
            self.replay_buffer.reset()

            total_game = env_workwer.n_games
            if show_score:
                print(f'Iteration {iteration:3d} total_game {total_game:3d} avg_score {np.mean(env_workwer.score_list[-10:]):.1f} \
                        actor_loss {" " if a_loss>=0 else ""}{a_loss:.3f}   critic_loss {c_loss:.3f}')

        plt.plot(env_workwer.score_list)
        plt.show()

    def calculate_discount_rewards(self, rewards, dones, discount_factor, normalized=True):
        returns = np.zeros_like(rewards)
        acc_a_value = 0

        # As[t] = R[t] + discount_factor**1 * R[t+1] + discount_factor**2 * R[t+2] + ... discount_factor**T-t * R[T]
        for ti in reversed(range(len(rewards))): # calculate reversely
            acc_a_value = rewards[ti] + discount_factor * acc_a_value * (not dones[ti])
            
            returns[ti] = acc_a_value

        if normalized:
            returns = (returns - returns.mean()) / (returns.std() + 1e-10)
        return returns

    def calculate_deltas(self, rewards, values, next_values, dones, discount_factor):
        deltas = [r - v + discount_factor*nv*(not d) for r, v, nv, d in zip(rewards, values, next_values, dones)]
        return deltas

    def calculate_gae_advantages_and_targets(self, rewards, values, next_values, dones, discount_fator, gae_lambda, normalized=True):
        deltas     = self.calculate_deltas(rewards, values, next_values, dones, discount_fator)  # reward shaping
        advantages = self.calculate_discount_rewards(deltas, dones, discount_fator*gae_lambda, normalized=False)

        targets = advantages + values
        if normalized:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10)
        return advantages, targets

    def generate_batches(self, states, actions, advantages, targets, old_action_porbs, old_values, batch_size):
        batch_indices = list(range(0, len(states), batch_size))
        np.random.shuffle(batch_indices)
        
        for start_idx in batch_indices:
            end_idx = start_idx + batch_size
            yield states          [start_idx:end_idx], \
                  actions         [start_idx:end_idx], \
                  advantages      [start_idx:end_idx], \
                  targets         [start_idx:end_idx], \
                  old_action_porbs[start_idx:end_idx], \
                  old_values      [start_idx:end_idx], 

    def learn(self, epochs, batch_size):
        # train model using TD method
        states           , \
        next_states      , \
        rewards          , \
        dones            , \
        old_actions      , \
        old_action_probs , = self.replay_buffer.get_all_tf_tensor_datas()

        old_values       = self.critic(states)       # values before changing critic model variables
        old_next_values  = self.critic(next_states)  # values before changing critic model variables

        # old advantages and targets before changing any variable
        old_advantages, old_value_targets = self.calculate_gae_advantages_and_targets(rewards, old_values, old_next_values, dones, self.gamma, self.gae_lambda, normalized=True)
        old_advantages    = tf.convert_to_tensor(old_advantages,    dtype=tf.float32) 
        old_value_targets = tf.convert_to_tensor(old_value_targets, dtype=tf.float32) 
        
        # training using all old experiences for n epochs with m batch size
        a_loss = self.policy.fit(x=states, y=[old_actions, old_advantages, old_action_probs], epochs=epochs, verbose=0, batch_size=batch_size)
        c_loss = self.critic.fit(x=states, y=[old_values,  old_value_targets],                epochs=epochs, verbose=0, batch_size=batch_size)
        return np.mean(a_loss.history['actor_loss']), np.mean(c_loss.history['critic_loss'])

        # training from scratch
        # a_loss_sum = 0
        # c_loss_sum = 0
        # for _ in range(epochs):

        #     a_loss_list = []
        #     c_loss_list = []
        #     for b_states,           \
        #         b_old_actions,      \
        #         b_old_advantages,   \
        #         b_old_targets,      \
        #         b_old_action_probs, \
        #         b_old_values        in self.generate_batches(states, old_actions, old_advantages, old_value_targets, old_action_probs, old_values, batch_size):

        #         with tf.GradientTape() as a_tape, tf.GradientTape() as c_tape:
        #             a_loss = self.policy.actor_loss (b_states, b_old_actions, b_old_advantages, b_old_action_probs, self.policy.epsilon, self.policy.entropy_coefficient)
        #             c_loss = self.critic.critic_loss(b_states, b_old_values,  b_old_targets, self.critic.critic_loss_coefficient)
                    
        #         a_gradients = a_tape.gradient(a_loss, self.policy.trainable_variables)
        #         self.policy.optimizer.apply_gradients(zip(a_gradients, self.policy.trainable_variables))

        #         c_gradients = c_tape.gradient(c_loss, self.critic.trainable_variables)
        #         self.critic.optimizer.apply_gradients(zip(c_gradients, self.critic.trainable_variables))

        #         a_loss_list.append(a_loss)
        #         c_loss_list.append(c_loss)
            
        #     a_loss_sum += np.mean(a_loss_list)
        #     c_loss_sum += np.mean(c_loss_list)

        # return a_loss_sum/epochs if epochs>0 else 0, c_loss_sum/epochs if epochs>0 else 0
        
        
if __name__ == "__main__":
    import gym

    env = gym.make('CartPole-v0')
    # env = gym.make('MountainCar-v0')

    # hyperparameters
    iterations                = 300
    n_steps                   = 250
    n_epochs                  = 10
    batch_size                = 25
    n_actions                 = env.action_space.n
    learning_rate             = 1e-4
    gamma                     = 0.99
    gae_lambda                = 0.95
    clip_epsilon              = 0.2
    critic_loss_coefficient   = 0.5
    entropy_bonus_coefficient = 0.001

    policy = PPOActorModel(n_actions, entropy_bonus_coefficient, learning_rate, epsilon=clip_epsilon)
    critic = PPOCriticModel(critic_loss_coefficient, learning_rate, epsilon=clip_epsilon)
    agent  = PPOAgent(policy, critic, gamma, gae_lambda)

    agent.train(env, iterations, n_steps, n_epochs, batch_size)


