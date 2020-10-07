import gym
import numpy as np


def test_step():
    env = gym.make('constraint_water_treatment_gym:distillation-plant-v0',
                   inflow=lambda: 0,
                   out_ref=0, volume_init=5, penalty_scale=1000)
    env.reset()
    assert env.step(np.zeros(1))[1] < -1000
