import torch as th
from stable_baselines3.common.policies import BasePolicy

from constraint_water_treatment_gym.envs import WaterTreatmentEnv
from constraint_water_treatment_gym.models.envmodel import DivertModel
from constraint_water_treatment_gym.penality.base import Penalty


class SimplePenalty(Penalty):
    def __init__(self, env: WaterTreatmentEnv, env_model=None):
        super().__init__()
        self.model = env_model or DivertModel(env)
        self.limits = (th.Tensor(env.observation_space.low), th.Tensor(env.observation_space.high))

    def __call__(self, policy: BasePolicy, obs: th.Tensor):
        acts, _, _ = policy.forward(obs)
        return th.stack(
            [self.penalty(act_.squeeze(), obs_.squeeze()) for (act_, obs_) in zip(th.split(acts, 1), th.split(obs, 1))])

    def penalty(self, act: th.Tensor, obs: th.Tensor):
        obs = self.model(act, obs)
        # clip can currently not be used for vectors https://github.com/pytorch/pytorch/issues/2793
        return th.sum((obs - th.max(th.min(obs, self.limits[1]), self.limits[0])) ** 2)
