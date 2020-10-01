import gym
import numpy as np
from gym import error, spaces, utils
from gym.utils import seeding


# TODO: time resolution


class WaterTreatmentEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, inflow=lambda: .5):
        self.inflow = inflow
        self.out_max = 1
        self.out_ref = .5
        self.volume = None
        self.last_in = None
        self.volume_max = 100

        self.viewer = None

        self.action_space = gym.spaces.Box(np.zeros(1), np.ones(1))
        self.observation_space = gym.spaces.Box(np.full(2, -1), np.full(2, 1))
        self.seed()

    def reset(self):
        self.volume = 50
        self.last_in = self.inflow()
        return self._get_obs()

    def step(self, action: np.ndarray):
        self.last_in = self.inflow()
        out = self.out_max * action.squeeze()

        delta = self.last_in - out
        self.volume += delta

        # validate and fail overflow
        if self.volume > self.volume_max:
            return self._get_obs(), -100, True, dict(reason='overflow')

        # validate and correct underflow
        underflow = min(self.volume, 0)
        out += underflow
        self.volume -= underflow

        cost = (self.out_ref - out) ** 2

        return self._get_obs(), 1 - cost, False, {}

    def render(self, mode='human'):
        from gym.envs.classic_control import rendering

        if self.viewer is None:
            self.viewer = rendering.Viewer(500, 500)
            self.viewer.set_bounds(-1, 1, -1.2, 1.2)

        l, r, t, b = .1, -.1, 1, - 1
        self.viewer.draw_polygon([(l, b), (l, t), (r, t), (r, b)], filled=False)
        t = 2 * (self.volume / self.volume_max) - 1
        self.viewer.draw_polygon([(l, b), (l, t), (r, t), (r, b)])

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def _get_obs(self):
        return np.array([self.last_in, self.volume])
