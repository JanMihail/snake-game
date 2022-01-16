import time

import gym

from snakeenv import SnakeEnv

env = SnakeEnv()
print(env.action_space)
print(env.observation_space)

episodes = 20
for episode in range(1, episodes + 1):
    env.reset()
    episode_reward = 0
    done = False

    while not done:
        env.render()
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        episode_reward += reward
        print(obs)
        time.sleep(0.001)

    env.render()
    time.sleep(1.)
    print("Episode {}. Reward: {}".format(episode, episode_reward))

env.close()
