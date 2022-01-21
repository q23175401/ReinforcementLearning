from keras_dueling_dqn import DuelingDqnAgent, DuelingDeepQNetworkModel
from keras_ddqn_agent import DDqnAgent


class DuellingDDqnAgent(DDqnAgent, DuelingDqnAgent):
    def __init__(self, n_actions, input_shape, learning_rate, update_target_steps, epsilon, epsilon_decay, epsilon_end, max_buffer_size, min_data_to_collect, q_network: DuelingDeepQNetworkModel):
      super().__init__(n_actions, input_shape, learning_rate, update_target_steps, epsilon, epsilon_decay, epsilon_end, max_buffer_size, min_data_to_collect, q_network)