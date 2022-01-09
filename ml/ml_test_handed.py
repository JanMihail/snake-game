import time

import numpy
import tensorflow as tf

from funcs import *

ENV_NAME = 'Acrobot-v1'
# ENV_NAME = 'CartPole-v1'

print('=== Create environment ===')
env = gym.make(ENV_NAME)
input_cnt = env.observation_space.shape[0]
output_cnt = env.action_space.n

print('=== Create model ===')
model = create_model(input_cnt, output_cnt)
model.summary()

print('=== Load weights ===')
model.load_weights('weights-{}.h5f'.format(ENV_NAME))

episodes = 10
for episode in range(1, episodes + 1):
    done = False
    episode_reward = 0
    obs = env.reset()

    while not done:
        env.render()
        x = tf.Variable([[obs.tolist()]])
        y = model(x)
        action = numpy.argmax(y.numpy())
        print("out: {}. Action: {}".format(y.numpy(), action))
        obs, reward, done, info = env.step(action)
        episode_reward += reward

    # time.sleep(1)
    env.render()
    print("Episode: {}. Reward: {}".format(episode, episode_reward))
    time.sleep(1)

env.close()
