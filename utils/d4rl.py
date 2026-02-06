import minari 
#import gymnasium as gym
#import gymnasium_robotics

"""gym.register_envs(gymnasium_robotics)

env = gym.make('AdroitHandPen-v1', max_episode_steps=400)

# d4rl abides by the OpenAI gym interface
env.reset()
env.step(env.action_space.sample())"""

# Alternatively, use d4rl.qlearning_dataset which
# also adds next_observations.
dataset = minari.load_dataset('D4RL/pen/human-v2', download=True)

for episode_data in dataset.iterate_episodes():
    observations = episode_data.observations
    actions = episode_data.actions
    rewards = episode_data.rewards
    terminations = episode_data.terminations
    truncations = episode_data.truncations
    infos = episode_data.infos

    print(f"Episode length: {len(observations)}")
    print(type(observations), type(actions), type(rewards))
    break  # Just process the first episode for demonstration

