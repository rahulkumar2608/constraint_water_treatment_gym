import torch as th

from constraint_water_treatment_gym.envs import WaterTreatmentEnv


class DivertModel:
    """
    selects inflow to push to the closest boundary (close inflow if low volume, max inflow above 50% vol)
    """

    def __init__(self, env: WaterTreatmentEnv):
        self.env = env

    def __call__(self, act: th.Tensor, obs: th.Tensor):
        if obs[1] < .5:
            return th.stack([th.tensor(0), obs[1] - act / self.env.vol_max])
        else:
            return th.stack([th.tensor(1), obs[1] + (1 - act) / self.env.vol_max])
