import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import matplotlib.pyplot as plt
from policy_gradient_model import PolicyGradientModel
    
class PolicyGradientAgent():
    def __init__(self, n_actions, policy_network:PolicyGradientModel, gamma=0.99):
        self.gamma     = gamma
        self.n_actions = n_actions

        self.policy    = policy_network

        self.reset_buffers()

    def reset_buffers(self):
        self.state_buffer  = []
        self.reward_buffer = []
        self.action_buffer = []

    def choose_action(self, one_state):
        state = tf.convert_to_tensor([one_state], dtype=tf.float32)
        action_probs = self.policy(state)

        action_dist = tfp.distributions.Categorical(probs=action_probs)
        action = action_dist.sample()

        return action.numpy()[0]

    def store_transition(self, state, reward, action):
        self.state_buffer.append(state)
        self.action_buffer.append(action)
        self.reward_buffer.append(reward)

    def train(self, env, n_episodes, show_score=10):
        
        scores = []
        for i_episode in range(n_episodes):
            SHOE_RESULT = True if show_score>0 and i_episode % show_score == 0 else False

            done = False
            score = 0
            state = env.reset()
            while not done:
                action = self.choose_action(state)
                next_state, reward, done, info = env.step(action)
                self.store_transition(state, reward, action)

                state = next_state
                score += reward

            # monte carlo method trains model after each episode
            self.learn()
            # use on policy method, clear buffer after training
            self.reset_buffers()
            

            scores.append(score)
            if SHOE_RESULT:
                print(f'Episode {i_episode} score {score} average score {np.mean(scores[-show_score:])}')
            
        plt.plot(range(n_episodes), scores)
        plt.show()

    def calculate_discount_rewards(self, rewards, discount_factor, normalized=True):
        Gs = np.zeros_like(rewards)
        acc_g_value = 0

        # Gs[t] = R[t] + discount_factor**1 * R[t+1] + discount_factor**2 * R[t+2] + ... discount_factor**T-t * R[T]
        for ti in reversed(range(len(rewards))): # calculate reversely
            acc_g_value = rewards[ti] + discount_factor * acc_g_value
            
            Gs[ti] = acc_g_value

        if normalized:
            Gs = (Gs - Gs.mean()) / Gs.std()
        return Gs

    def learn(self):
        # train model using monte carlo method
        states  = tf.convert_to_tensor(self.state_buffer,  dtype=tf.float32)
        actions = tf.convert_to_tensor(self.action_buffer, dtype=tf.float32)
        rewards = tf.convert_to_tensor(self.reward_buffer)

        Gs = self.calculate_discount_rewards(rewards, self.gamma, normalized=True)
        Gs = tf.convert_to_tensor(Gs, dtype=tf.float32)

        loss = self.policy.train_on_batch(x=states, y=[actions, Gs])
        # loss = self.policy.fit(x=states, y=[actions, Gs], verbose=0)
        return loss

        

if __name__ == "__main__":
    import gym

    env = gym.make('CartPole-v0')

    # hyperparameters
    n_episodes = 500
    n_actions = env.action_space.n
    LEARNING_RATE = 3e-4
    gamma = 0.99

    policy = PolicyGradientModel(n_actions, LEARNING_RATE)
    
    agent = PolicyGradientAgent(n_actions, policy, gamma)

    agent.train(env, n_episodes)


