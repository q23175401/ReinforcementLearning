import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam

class PPOCriticModel(Model):
    def __init__(self, critic_loss_coefficient=0.5, learning_rate=0.001, fc1_units=128, fc2_units=128, epsilon=0.2):
        super().__init__()
        self.epsilon = epsilon
        self.critic_loss_coefficient = critic_loss_coefficient

        self.F = Flatten()
        self.D1 = Dense(fc1_units, activation='relu')
        self.D2 = Dense(fc2_units, activation='relu')
        self.V = Dense(1, activation='linear')

        self.compile(optimizer=Adam(learning_rate=learning_rate))

    def call(self, state):
        x = state
        x = self.F(x)
        x = self.D1(x)
        x = self.D2(x)
        value = self.V(x)

        return value

    def get_value(self, one_state):
        state = tf.convert_to_tensor([one_state], dtype=tf.float32)
        return self.call(state)

    def critic_loss(self, states, old_values, targets, critic_loss_coefficient=0.5):
        # var_2 = 

        new_values = self.call(states)

        all_critic_loss = (new_values - targets)**2
        # critic_loss = tf.math.reduce_sum(all_critic_loss)
        critic_loss = tf.math.reduce_mean(all_critic_loss)
        return critic_loss * critic_loss_coefficient

    def train_step(self, data):
        states, [old_values, targets] = data

        with tf.GradientTape() as tape:
            loss = self.critic_loss(states, old_values, targets, self.critic_loss_coefficient)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return {'critic_loss': loss}
