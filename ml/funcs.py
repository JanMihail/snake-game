import gym
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input, Flatten
from tensorflow.keras.optimizers import Adam
# from rl.agents import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import BoltzmannQPolicy



def test_env_without_nn_random(env_name):
    env = gym.make(env_name)

    episodes = 10
    for episode in range(1, episodes + 1):
        env.reset()
        episode_reward = 0
        done = False

        while not done:
            env.render()
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            episode_reward += reward

        print("Episode {}. Reward: {}".format(episode, episode_reward))

    env.close()


def create_model(input_cnt, output_cnt):
    model = Sequential()
    model.add(Input(shape=(1, input_cnt)))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(output_cnt, activation='linear'))
    return model
    # model = Sequential()
    # model.add(Input(shape=(1, input_cnt)))
    # model.add(Flatten())
    # model.add(Dense(24, activation='relu'))
    # model.add(Dense(24, activation='relu'))
    # model.add(Dense(output_cnt, activation='linear'))
    # return model


def create_agent(model, output_cnt, fit_steps_cnt):
    policy = BoltzmannQPolicy()
    memory = SequentialMemory(limit=fit_steps_cnt, window_length=1)
    agent = DQNAgent(model=model, memory=memory, policy=policy,
                     nb_actions=output_cnt, nb_steps_warmup=10, target_model_update=1e-2)
    agent.compile(Adam(learning_rate=1e-3), metrics=['accuracy'])
    return agent
