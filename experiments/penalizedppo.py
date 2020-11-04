from datetime import datetime
from os import makedirs

import gym
import numpy as np
from gym.wrappers.monitoring.video_recorder import VideoRecorder
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList, EveryNTimesteps, BaseCallback
from stable_baselines3.common.monitor import Monitor

from constraint_water_treatment_gym.agents.penalized_ppo import PenalizedPPO
from constraint_water_treatment_gym.models import MarkovChain
from constraint_water_treatment_gym.penality.simple import SimplePenalty
from plotting.networks import visualize_nets

np.random.seed(0)
inflow = MarkovChain(np.linspace(0, 1, 5),
                     .5 * np.eye(5) +
                     .5 * np.array([[0, .5, .5, 0, 0],
                                    [0, 0, 0, .2, .8],
                                    [0, 1, 0, 0, 0],
                                    [0, 0, 1, 0, 0],
                                    [0, 1, 0, 0, 0]]))

env_cfg = dict(safety_factor=.1, vol_init=None, vol_max=10, penalty=0, steps_max=500)
model_cfg = dict(penalty_coef=5000)

timestamp = datetime.now().strftime(f'%Y.%b.%d %X {env_cfg}')
makedirs(timestamp)
env = gym.make('constraint_water_treatment_gym:distillation-plant-v0',
               inflow=inflow.step,
               out_ref=inflow.expectation(), **env_cfg)
with open(f'{timestamp}/env.txt', 'w') as f:
    print(str(env), file=f)
with open(f'{timestamp}/model.txt', 'w') as f:
    print(str(model_cfg), file=f)
with open(f'{timestamp}/inflow.txt', 'w') as f:
    print(str(inflow), file=f)
env_wrap = Monitor(env)


class VisCallback(BaseCallback):
    def __init__(self, timestamp):
        super().__init__()
        self.timestamp = timestamp

    def _on_step(self):
        visualize_nets(self.training_env, self.model, f'{self.timestamp}/{self.num_timesteps}')


class ExecCallback(BaseCallback):
    def __init__(self, timestamp):
        super().__init__()
        self.timestamp = timestamp

    def _on_step(self):
        rec = VideoRecorder(env, f'{self.timestamp}/{self.num_timesteps}_vid.mp4')
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


model = PenalizedPPO('MlpPolicy', env_wrap, penalty=SimplePenalty(env), verbose=1,
                     tensorboard_log=f'{timestamp}/', **model_cfg)
callbacks = [
    CheckpointCallback(save_freq=100000, save_path=f'{timestamp}/checkpoints/'),
    EveryNTimesteps(n_steps=50000, callback=VisCallback(timestamp)),
    # EveryNTimesteps(n_steps=50000, callback=ExecCallback(timestamp)),
]

model.learn(total_timesteps=5000000, callback=CallbackList(callbacks))
