import torch as th

from constraint_water_treatment_gym.envs import WaterTreatmentEnv
from constraint_water_treatment_gym.penality.base import Penalty


class DirectPenalty(Penalty):
    """
    This Penalty function directly calculates the penalty from the existing state and ignores the penalty
    """

    def __init__(self, env: WaterTreatmentEnv, low=None, high=None):
        super().__init__()
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

    def __call__(self, policy, obs: th.Tensor):
        return th.sum((obs - th.max(th.min(obs, self.high), self.low)) ** 2)
