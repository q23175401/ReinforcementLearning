import tensorflow as tf

class PPOReplayBuffer():
    def __init__(self):
        self.reset()

    def reset(self):
        self.stored_steps = 0
        self.state_buffer              = []
        self.next_state_buffer         = []
        self.old_action_buffer         = []
        self.old_action_probs_buffer   = []
        self.reward_buffer             = []
        self.done_buffer               = []

    def store_transition(self, state, next_state, action, action_prob, reward, done):
        self.state_buffer.append(state)
        self.next_state_buffer.append(next_state)
        self.old_action_buffer.append(action)
        self.old_action_probs_buffer.append(action_prob)
        self.reward_buffer.append(reward)
        self.done_buffer.append(done)
        self.stored_steps += 1

    def get_all_tf_tensor_datas(self):
        states           = tf.convert_to_tensor(self.state_buffer,            dtype=tf.float32)
        next_states      = tf.convert_to_tensor(self.next_state_buffer,       dtype=tf.float32)
        rewards          = tf.convert_to_tensor(self.reward_buffer,           dtype=tf.float32)
        dones            = tf.convert_to_tensor(self.done_buffer,             dtype=tf.float32)
        old_actions      = tf.convert_to_tensor(self.old_action_buffer,       dtype=tf.float32)
        old_action_probs = tf.convert_to_tensor(self.old_action_probs_buffer, dtype=tf.float32)

        return states           , \
               next_states      , \
               rewards          , \
               dones            , \
               old_actions      , \
               old_action_probs 

    def __len__(self):
        return self.stored_steps