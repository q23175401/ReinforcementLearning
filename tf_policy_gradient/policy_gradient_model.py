import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam

class PolicyGradientModel(Model):
    def __init__(self, n_actions, learning_rate=0.001, fc1_units=128, fc2_units=128):
        super().__init__()
        self.n_actions = n_actions
        
        self.F = Flatten()
        self.D1 = Dense(fc1_units, activation='relu')
        self.D2 = Dense(fc2_units, activation='relu')
        self.PI = Dense(self.n_actions, activation='softmax')

        self.compile(optimizer=Adam(learning_rate=learning_rate))

    def call(self, state):
        x = state
        x = self.F(x)
        x = self.D1(x)
        x = self.D2(x)
        action_probs = self.PI(x)

        return action_probs

    def policy_gradient_loss(self, states, actions, advantages, entropy_coefficient=0.001):
        action_probs = self(states)
        probs = tfp.distributions.Categorical(probs=action_probs)
        entropy = probs.entropy()

        log_probs = probs.log_prob(actions)   # get log probabilities of each action in this episode
        all_loss = advantages*log_probs + entropy*entropy_coefficient
        # loss = -tf.math.reduce_sum(all_loss)  # gradient ascent => to maximize average total reward of this model
        loss = -tf.math.reduce_mean(all_loss)
        return loss

    def train_step(self, data):
        states, [actions, advantages] = data
        
        with tf.GradientTape() as tape:
            
            loss = self.policy_gradient_loss(states, actions, advantages)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return {'loss': loss}
