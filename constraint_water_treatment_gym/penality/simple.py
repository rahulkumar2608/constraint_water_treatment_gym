import torch as th
from stable_baselines3.common.policies import BasePolicy

from constraint_water_treatment_gym.models.envmodel.divert import DivertModel
from constraint_water_treatment_gym.penality.base import Penalty


class SimplePenalty(Penalty):
    def __init__(self, low, high, env_model=None):
        super().__init__()
        self.model = env_model or DivertModel()
        self.limits = (th.Tensor(low), th.Tensor(high))

    def __call__(self, policy: BasePolicy, obs: th.tensor):
        act = policy.predict(obs)
        obs = self.model(act, obs)
        return (obs - th.clamp(th.Tensor(obs), *self.limits)) ** 2
