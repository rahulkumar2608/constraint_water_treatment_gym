from datetime import datetime
from os import makedirs

import gym
import numpy as np
from gym.wrappers.monitoring.video_recorder import VideoRecorder
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from constraint_water_treatment_gym.models import MarkovChain
from experiments.plotting.networks import visualize_nets

np.random.seed(0)
inflow = MarkovChain(np.linspace(0, 1, 5),
                     .9 * np.eye(5) +
                     .1 * np.array([[0, .5, .5, 0, 0],
                                    [0, 0, 0, .2, .8],
                                    [0, 1, 0, 0, 0],
                                    [0, 0, 1, 0, 0],
                                    [0, 1, 0, 0, 0]]))

vol_max = 10

timestamp = datetime.now().strftime('%Y.%b.%d %X')
makedirs(timestamp)
env = gym.make('constraint_water_treatment_gym:distillation-plant-v0',
               inflow=inflow.step,
               out_ref=inflow.expectation(), penalty_scale=120,
               safety_factor=.2, volume_max=vol_max, max_eps_len=4000)
env = gym.make('MountainCarContinuous-v0')
with open(f'{timestamp}/env.txt', 'w') as f:
    print(repr(env), file=f)
env = Monitor(env)

model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=f'{timestamp}/')
model.learn(total_timesteps=100000)
model.save(f'{timestamp}/model')

domain = np.rot90(np.array([env.observation_space.high, env.observation_space.low]), k=3)
visualize_nets(env, model, timestamp, domain=domain)

rec = VideoRecorder(env, f'{timestamp}/vid.mp4')
obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.unwrapped.render()
    rec.capture_frame()
    if done:
        obs = env.reset()

env.close()
rec.close()
