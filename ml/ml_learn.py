from funcs import *

ENV_NAME = 'Acrobot-v1'
FIT_STEPS_CNT = 100000

print('=== Test environment without neural network ===')
test_env_without_nn_random(ENV_NAME)

print('=== Create environment ===')
env = gym.make(ENV_NAME)
input_cnt = env.observation_space.shape[0]
output_cnt = env.action_space.n

print('=== Create model ===')
model = create_model(input_cnt, output_cnt)
model.summary()

print('=== Create learning agent ===')
agent = create_agent(model, output_cnt, FIT_STEPS_CNT)

print('=== Learning ===')
agent.fit(env, nb_steps=FIT_STEPS_CNT, visualize=False, verbose=1)

print('=== Testing ===')
result = agent.test(env, nb_episodes=100, visualize=False, verbose=1)
print('Average reward: {0}'.format(np.mean(result.history['episode_reward'])))

print('=== Save ===')
agent.save_weights('weights-{}.h5f'.format(ENV_NAME), overwrite=True)

env.close()
