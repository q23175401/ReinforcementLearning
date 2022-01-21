# from connectx_agents import MinimaxAgent
import random as r
from gym import Env
from kaggle_environments import make
from gym.spaces import Discrete, Box
import numpy as np
import os

class GameBoardEnv(Env):
    def __init__(self, opponent_agent):
        ks_env = make("connectx", debug=True)
        self.train_env = ks_env.train([None, opponent_agent])
        self.row = ks_env.configuration['rows']
        self.col = ks_env.configuration['columns']   

        self.action_space = Discrete(self.col)
        self.game_board_shape = (self.row, self.col, 1)
        self.observation_space = Box(low=0, high=2, shape=self.game_board_shape, dtype=np.float32)
        self.reset()

    @property
    def state(self):
        return np.array(self.obs['board']).reshape(self.game_board_shape)

    def step(self, action):
        is_valid_move = self.state[0, action] == 0

        if is_valid_move: # Play the move
            self.obs, old_reward, done, _ = self.train_env.step(int(action))
            reward = self.change_reward(old_reward, done)
        else: # End the game and penalize agent
            reward, done, _ = -10, True, {}
        
        return self.state, reward, done, {}

    def change_reward(self, old_reward, done):
        if old_reward == 1: # The agent won the game
            return 1
        elif done: # The opponent won the game
            return -1
        else: # Reward 1/42
            return 1/(self.row*self.col)

    def render(self):
        for r in range(self.row):
            col_str = ""
            for c in range(self.col):
                # col_str += str(self.state[r][c]) + " "
                # continue

                if self.state[r][c]==2:
                    col_str+="X "
                elif self.state[r][c]==1:
                    col_str+="O "
                else:
                    col_str+="_ "
            print(col_str)

    def reset(self):
        self.obs = self.train_env.reset()
        return self.state

    def print_board(self):
        os.system("cls")
        self.render()

if __name__=="__main__":
    game_board = GameBoardEnv('random')
    import time
    from tf_agents.environments.gym_wrapper import GymWrapper
    from tf_agents.environments import TFPyEnvironment

    py_env = GymWrapper(game_board)
    tf_env = TFPyEnvironment(py_env)
    # game_board.print_board()

    # # time.sleep(1)
    # # game_board.step(0)
    # # game_board.print_board()

    # # time.sleep(1)
    # # game_board.step(0)
    # # game_board.print_board()
    # for i in range(8):
    #     time.sleep(1)
    #     game_board.step(0)
    #     game_board.print_board()

    