from itertools import islice
from typing import Generator

import numpy as np
import torch as th
from stable_baselines3.common.policies import BasePolicy

from constraint_water_treatment_gym.envs import WaterTreatmentEnv
from constraint_water_treatment_gym.penality.simple import SimplePenalty


class StateSetPenalty(SimplePenalty):
    def __init__(self, env: WaterTreatmentEnv, gen: Generator[th.Tensor, None, None] = None, n: int = 64,
                 env_model=None, low=None,
                 high=None):
        super().__init__(env, env_model, low, high)
        if gen is None:
            self.gen = self._state_generator
        else:
            self.gen = gen
        self.state_generator = self.gen()
        self.n = n

    def __call__(self, policy: BasePolicy, obs: th.Tensor):
        # sample test states to base the penalty calculation on and calculate penalty using parent implementation
        test_states = th.stack(list(islice(self.state_generator, self.n)))
        return super().__call__(policy, test_states)

    def _state_generator(self):
        """default implementation"""
        while True:
            # a region arround the savety boundary with twice the width of the maximal step size
            dist = th.distributions.Uniform(self.env.vol_safety_factor - 1 / self.env.vol_max,
                                            self.env.vol_safety_factor + 1 / self.env.vol_max)

            if np.random.random() < .5:
                # top boundary
                yield th.stack([th.rand(1).squeeze(), 1 - dist.sample()])
            else:
                # bottom boundary
                yield th.stack([th.rand(1).squeeze(), dist.sample()])
