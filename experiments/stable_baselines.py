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
                     .5 * np.eye(5) +
                     .5 * np.array([[0, .5, .5, 0, 0],
                                    [0, 0, 0, .2, .8],
                                    [0, 1, 0, 0, 0],
                                    [0, 0, 1, 0, 0],
                                    [0, 1, 0, 0, 0]]))

env_cfg = dict(safety_factor=.1, vol_init=None, vol_max=10, penalty=1, steps_max=100)

timestamp = datetime.now().strftime(f'%Y.%b.%d %X {env_cfg}')
makedirs(timestamp)
env = gym.make('constraint_water_treatment_gym:distillation-plant-v0',
               inflow=inflow.step,
               out_ref=inflow.expectation(), **env_cfg)
with open(f'{timestamp}/env.txt', 'w') as f:
    print(str(env), file=f)
with open(f'{timestamp}/inflow.txt', 'w') as f:
    print(str(inflow), file=f)
env = Monitor(env)

model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=f'{timestamp}/', gamma=.5)
model.learn(total_timesteps=5000000)
model.save(f'{timestamp}/model')

visualize_nets(env, model, timestamp)

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
