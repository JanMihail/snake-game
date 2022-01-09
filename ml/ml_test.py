from funcs import *

ENV_NAME = 'Acrobot-v1'
FIT_STEPS_CNT = 50000

print('=== Create environment ===')
env = gym.make(ENV_NAME)
input_cnt = env.observation_space.shape[0]
output_cnt = env.action_space.n

print('=== Create model ===')
model = create_model(input_cnt, output_cnt)
model.summary()

print('=== Load weights ===')
model.load_weights('weights-{}.h5f'.format(ENV_NAME))

print('=== Create learning agent ===')
agent = create_agent(model, output_cnt, FIT_STEPS_CNT)

print('=== Testing ===')
result = agent.test(env, nb_episodes=10, visualize=True, verbose=1)
print('Average reward: {0}'.format(np.mean(result.history['episode_reward'])))

env.close()
