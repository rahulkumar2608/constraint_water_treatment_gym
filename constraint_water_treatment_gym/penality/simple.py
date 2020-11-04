import torch as th
from stable_baselines3.common.policies import BasePolicy

from constraint_water_treatment_gym.envs import WaterTreatmentEnv
from constraint_water_treatment_gym.models.envmodel import DivertModel
from constraint_water_treatment_gym.penality.base import Penalty


class SimplePenalty(Penalty):
    def __init__(self, env: WaterTreatmentEnv, env_model=None, low=None, high=None):
        super().__init__()
        self.model = env_model or DivertModel(env)
        self.env = env
        if low is None:
            self.low = th.Tensor([env.observation_space.low[0],
                                  env.observation_space.low[1] * env.vol_safety_factor])
        else:
            self.low = low
        if high is None:
            self.high = th.Tensor([env.observation_space.high[0],
                                   1 - env.observation_space.high[1] * env.vol_safety_factor])
        else:
            self.high = high

    def __call__(self, policy: BasePolicy, obs: th.Tensor):
        acts, _, _ = policy.forward(obs, deterministic=True)
        # actions stem from -1,1 interval because of tanh. in policy.predict the function "unscale" is used. This needs to be reimplemented here as its only defined for numpy
        acts = self.unscale_action(acts)
        return th.stack(
            [self.penalty(act_.squeeze(), obs_.squeeze()) for (act_, obs_) in
             zip(th.split(acts, 1), th.split(obs, 1))])

    def penalty(self, act: th.Tensor, obs: th.Tensor):
        obs = self.model(act, obs)
        # clip can currently not be used for vectors https://github.com/pytorch/pytorch/issues/2793
        return th.sum((obs - th.max(th.min(obs, self.high), self.low)) ** 2)

    def unscale_action(self, scaled_action: th.Tensor) -> th.Tensor:
        """
        copied from stablebaselines.Policy, converted to work with tensors

        Rescale the action from [-1, 1] to [low, high]
        (no need for symmetric action space)

        :param scaled_action: Action to un-scale
        """
        low, high = th.Tensor(self.env.action_space.low), th.Tensor(self.env.action_space.high)
        return low + (0.5 * (scaled_action + 1.0) * (high - low))
