import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam


class PPOActorModel(Model):
    def __init__(self, n_actions, entropy_coefficient=0.001, learning_rate=0.001, fc1_units=128, fc2_units=128, epsilon=0.2):
        super().__init__()
        self.n_actions = n_actions
        self.epsilon = epsilon
        self.entropy_coefficient = entropy_coefficient

        self.F = Flatten()
        self.D1 = Dense(fc1_units, activation='relu')
        self.D2 = Dense(fc2_units, activation='relu')
        self.PI = Dense(self.n_actions, activation='softmax')

        self.compile(optimizer=Adam(learning_rate=learning_rate))

    def choose_action(self, one_state):
        state = tf.convert_to_tensor([one_state], dtype=tf.float32)
        action_probs = self.call(state)

        action_dist = tfp.distributions.Categorical(probs=action_probs)
        action = action_dist.sample() 
        action_prob = action_dist.prob(action)
        return action.numpy()[0], action_prob[0]

    def call(self, state):
        x = state
        x = self.F(x)
        x = self.D1(x)
        x = self.D2(x)
        action_probs = self.PI(x)

        return action_probs

    def actor_loss(self, states, actions, advantages, old_action_probs, epsilon=0.2, entropy_coefficient=0.001):
        action_dist = self(states)
        probs = tfp.distributions.Categorical(probs=action_dist)
        action_entropies = probs.entropy()
        action_probs = probs.prob(actions)  # get new probabilities of each action in this episode

        action_probs     = tf.clip_by_value(action_probs,     1e-10, 1) # to avoid value 0 
        old_action_probs = tf.clip_by_value(old_action_probs, 1e-10, 1) # to avoid value 0 
        log_action_prob     = tf.math.log(action_probs)
        log_old_action_prob = tf.math.log(old_action_probs)
        ratio = tf.math.exp(log_action_prob-log_old_action_prob)  # ratio = action_probs / old_action_probs = e^(log(p) - log(old_p))

        all_loss_v1 = advantages * ratio
        all_loss_v2 = advantages * tf.clip_by_value(ratio, 1-epsilon, 1+epsilon)
        all_loss = tf.math.minimum(all_loss_v1, all_loss_v2) + action_entropies * entropy_coefficient

        loss = -tf.math.reduce_mean(all_loss)  # gradient ascent => to maximize average total reward of this model
        return loss

    def train_step(self, data):
        states, [actions, advantages, old_action_probs] = data

        with tf.GradientTape() as tape:
            loss = self.actor_loss(states, actions, advantages, old_action_probs, self.epsilon, self.entropy_coefficient)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return {'actor_loss': loss}