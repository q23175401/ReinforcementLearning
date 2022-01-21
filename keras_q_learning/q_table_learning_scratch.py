import gym
import numpy as np
import math
import matplotlib.pyplot as plt

# env = gym.make('MountainCar-v0')
env = gym.make('CartPole-v0')

print(env.observation_space.high)
print(env.observation_space.low)

## state bounds
feature_bounds = list(zip(env.observation_space.low, env.observation_space.high))
feature_bounds[3] = [-math.radians(50), math.radians(50)]

# cut this observation space from continuous space to discrete space
# DISCRETE_OBS_AMOUNT = [2] * len(env.observation_space.high)
DISCRETE_OBS_AMOUNT = [7, 2, 16, 9]
# DISCRETE_OBS_INTERVAL = (env.observation_space.high / DISCRETE_OBS_AMOUNT - env.observation_space.low / DISCRETE_OBS_AMOUNT) 
# # convert original state to discrete to create a qtable to store whole rewards of all state-action 
# def convert_state_to_discrete(state):
#     discrete_state = (state - env.observation_space.low) // DISCRETE_OBS_INTERVAL
#     return tuple(np.array(discrete_state, np.int))

def convert_state_to_discrete(observation):
    discrete_state = [0] * len(observation)
    for i, s in enumerate(observation):
        l, u = feature_bounds[i] # lower- and upper-bounds for each feature in observation
        if s <= l:
            discrete_state[i] = 0
        elif s >= u:
            discrete_state[i] = DISCRETE_OBS_AMOUNT[i] - 1
        else:
            discrete_state[i] = int((s - l) / (u/DISCRETE_OBS_AMOUNT[i] - l/DISCRETE_OBS_AMOUNT[i]))

    return tuple(discrete_state)


# create q table to store all possible rewards
# q_table = np.random.uniform(low=0, high=0, size=DISCRETE_OBS_AMOUNT + [env.action_space.n])
q_table = np.zeros(DISCRETE_OBS_AMOUNT + [env.action_space.n])

EPISODES = 200000
LEARNING_RATE = 1e-1
DISCOUNT_FACTOR = 0.99
SHOW_RESULT_INTERVAL = 2000

RANDOM_ACTION_RATIO = 0.9
RANDOM_ACTION_DECAY = 0.98
START_DECAY_RANDOM_ACTION_EPISODE = 0
END_DECAY_RANDOM_ACTION_EPISODE = EPISODES//2

# get_epsilon = lambda i: max(0.01, min(1,   1.0 - math.log10( (i+1) / 25 ) ) )
# get_lr      = lambda i: max(0.01, min(0.5, 1.0 - math.log10( (i+1) / 25 ) ) )

def get_epsilon(episode):
    max_value = min(1,   1.0 - math.log10( (episode + 1) / 25 ) ) # 最大不超過1
    min_value = max(0.01, max_value)                              # 最小不低於0.01
    return min_value

def get_learning_rate(episode):
    max_value = min(0.5,   1.0 - math.log10( (episode + 1) / 25 ) ) # 最大不超過0.5 
    min_value = max(0.01, max_value)                                # 最小不低於0.01, 隨著episode增加逐漸靠近0.01
    return min_value


def get_action(epsilon, discrete_state):
    if np.random.random_sample() > epsilon:
        action = np.argmax(q_table[discrete_state])
    else:
        action = env.action_space.sample()
    return action


time_steps = []
for episode in range(EPISODES):
    LEARNING_RATE = get_learning_rate(episode)      # update learning rate 
    RANDOM_ACTION_RATIO = get_epsilon(episode) # update epsilon

    if episode % SHOW_RESULT_INTERVAL == 0:
        render_result = True
    else:
        render_result = False
    

    done = False
    survive_time_step = 0
    state = env.reset()
    while not done:
        discrete_state = convert_state_to_discrete(state)

        action = get_action(RANDOM_ACTION_RATIO, discrete_state)
        new_state, reward, done, info = env.step(action)
        survive_time_step += reward

        if render_result:
            env.render()

        
        # q learning algorithm to calculate new q value
        action_index = discrete_state + (action, )
        if not done:
            max_furture_q_value = np.amax(q_table[convert_state_to_discrete(new_state)])

            old_q_value = q_table[action_index]
            new_q_value = (1-LEARNING_RATE) * old_q_value + LEARNING_RATE * (reward + DISCOUNT_FACTOR * max_furture_q_value)
            q_table[action_index] = new_q_value
        else:
            q_table[action_index] = survive_time_step
        # q_table[action_index] += LEARNING_RATE * (reward + DISCOUNT_FACTOR * max_furture_q_value - q_table[action_index])

        state = new_state

    # if START_DECAY_RANDOM_ACTION_EPISODE <= episode <= END_DECAY_RANDOM_ACTION_EPISODE:
    #     RANDOM_ACTION_RATIO = RANDOM_ACTION_RATIO * RANDOM_ACTION_DECAY

    if render_result:
        print(f'Episode {episode} Finished with {survive_time_step} time steps')

    time_steps.append(survive_time_step)
    if len(time_steps) >= SHOW_RESULT_INTERVAL:
        # plt.plot(time_steps)
        # plt.show()
        time_steps = []

env.close()