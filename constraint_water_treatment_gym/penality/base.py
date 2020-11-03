import numpy as np
from stable_baselines3.common.policies import BasePolicy


class Penalty:
    def __init__(self):
        pass

    def __call__(self, policy: BasePolicy, obs: np.ndarray):
        pass
