from ppo_actor_model import PPOActorModel
from ppo_replay_buffer import PPOReplayBuffer  

class PPOEnvWorker():
    def __init__(self, env):
        self.env = env
        self.reset()

    def reset(self):
        self.state = self.env.reset()
        self.done  = False
        self.score = 0
        self.score_list = []

    @property
    def n_games(self):
        return len(self.score_list)

    def collect_trajectories(self, replay_buffer:PPOReplayBuffer, policy:PPOActorModel, length):

        while len(replay_buffer)<length:

            while not self.done:
                action, action_prob = policy.choose_action(self.state)
                next_state, reward, self.done, info = self.env.step(action)

                replay_buffer.store_transition(self.state, next_state, action, action_prob, reward, self.done)
                self.state = next_state
                self.score += reward

                if len(replay_buffer) >= length:
                    return

            self.score_list.append(self.score)
            
            # reset game after each episode
            self.state = self.env.reset()
            self.done  = False
            self.score = 0
    