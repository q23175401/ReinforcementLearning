from tensorflow.keras.models import Model, load_model
import tensorflow.keras.layers as L
import tensorflow as tf
import numpy as np
from typing import Callable
from .keras_dqn_agent import DqnAgent

class DuelingDeepQNetworkModel(Model):
    def __init__(self, n_actions, input_shape, fc1_units, fc2_units):
        super().__init__()
        self.trainable=True
        self.n_actions = n_actions
        self.in_shape = input_shape
        self.fc1_units = fc1_units
        self.fc2_units = fc2_units

        self.model_body_list = [
                L.InputLayer(self.in_shape),
                L.Flatten(),
                L.Dense(fc1_units, activation='relu'),
                L.Dense(fc2_units, activation='relu'),
            ]

        self.V = L.Dense(1, activation='linear')
        self.A = L.Dense(self.n_actions, activation='linear')

        self.call(tf.random.uniform(shape=[1] + list(input_shape))) # force net to build its weights
        # self.build(tuple(input_shape)) # 要build才會實際生成符合網路架構的weights

    def call(self, state):
        x = state
        for layer in self.model_body_list:
            x = layer(x)

        v = self.V(x)
        a = self.A(x)

        # q value equals to v + a (force mean_a equals to 0 to force model to change v, instead of changing a)
        # 要維持住dim才不會減少axis 1
        #  => 
        # [[1, 2],   [[1.5],
        #  [3, 4]] => [3.5]]
        mean_a = tf.math.reduce_mean(a, axis=1, keepdims=True) 
        q_pred = v + (a - mean_a)

        return q_pred

    def get_advantage(self, state):
        x = state
        for layer in self.model_body_list:
            x = layer(x)

        a = self.A(x)
        return a

    def get_config(self):
        return {
            "n_actions"   : self.n_actions,
            "input_shape" : self.in_shape,
            'fc1_units'   : self.fc1_units,
            'fc2_units'   : self.fc2_units,
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)
        
class DuelingDqnAgent(DqnAgent):
    def __init__(self, n_actions, input_shape, learning_rate, update_target_steps, epsilon, epsilon_decay, epsilon_end, max_buffer_size, min_data_to_collect, build_q_network: Callable=None):
      super().__init__(n_actions, input_shape, learning_rate, update_target_steps, epsilon, epsilon_decay, epsilon_end, max_buffer_size, min_data_to_collect, build_q_network)

    def get_action(self, one_state):
        advantage = self.value_net.get_advantage(np.array([one_state])) # dueling net get action score from A network path
        action = np.array(tf.math.argmax(advantage, axis=1))[0]
        return action

    def build_my_qnet(self):
        model = DuelingDeepQNetworkModel(
                n_actions=self.n_actions,
                input_shape=self.input_shape,
                fc1_units=128,
                fc2_units=128,
            )

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate), loss='mse', metrics=['accuracy'])
        return model

    def load_agent(self, file_name):
        self.value_net = load_model(file_name, custom_objects={"DuelingDeepQNetworkModel": DuelingDeepQNetworkModel})
        self.target_net = load_model(file_name, custom_objects={"DuelingDeepQNetworkModel": DuelingDeepQNetworkModel})

if __name__ == "__main__":
    n_actions = 6
    input_shape = np.array([6, 7, 1])

    custom_model = DuelingDeepQNetworkModel(n_actions, input_shape, 128, 128)
    custom_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss='mse', metrics=['accuracy'])
    test_input = tf.random.uniform(shape=[1] + list(input_shape))
    
    print(test_input)
    ori_output = custom_model.predict(test_input)
    print(ori_output)
    # save_file_name = "test_save_custom_model"
    # custom_model.save(save_file_name)

    # loaded_model = load_model(
    #                 save_file_name, custom_objects={"DuelingDeepQNetworkModel": DuelingDeepQNetworkModel}
    #             )
    
    # prd_output = loaded_model.predict(test_input)
    # np.testing.assert_allclose(ori_output, prd_output)
    
    # created_model = DuelingDeepQNetworkModel(n_actions, input_shape, 128, 128)
    # created_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss='mse', metrics=['accuracy'])
    # created_model.set_weights(custom_model.get_weights())

    # prd_output = created_model.predict(test_input)
    # np.testing.assert_allclose(ori_output, prd_output)