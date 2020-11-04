from numbers import Real
from typing import Callable, Union, Optional

import gym
import numpy as np


# TODO: time resolution


class WaterTreatmentEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, inflow=lambda: .5, out_ref=.5, out_max=1, safety_factor: float = .1,
                 vol_init: Optional[Union[int, float, Callable]] = None, vol_max: float = 100.0,
                 penalty=100, steps_max=400):
        self.inflow = inflow
        self.out_ref = out_ref
        self.out_max = out_max
        self.last_in = None
        self.vol_init = vol_init
        self.vol = None
        self.vol_safety_factor = safety_factor
        self.vol_max = vol_max
        self.penaltyscale = penalty
        self.steps = None
        self.steps_max = float('inf') if steps_max is None else steps_max

        self.viewer = None

        self.action_space = gym.spaces.Box(np.zeros(1), np.ones(1))
        self.observation_space = gym.spaces.Box(np.zeros(2), np.ones(2))
        self.seed()

    def reset(self):
        if self.vol_init is None:
            self.vol = np.random.uniform(self.vol_safety_factor, (1 - self.vol_safety_factor)) \
                       * self.vol_max
        elif isinstance(self.vol_init, Real):
            self.vol = self.vol_init * self.vol_max
        else:
            self.vol = self.vol_init() * self.vol_max
        self.last_in = self.inflow()
        self.steps = 0
        return self._get_obs()

    def step(self, action: np.ndarray):
        self.steps += 1
        if self.steps > self.steps_max:
            raise ValueError("called an exceded env. please reset env before calling step again")

        self.last_in = self.inflow()
        out = self.out_max * action.squeeze()

        delta = self.last_in - out
        self.vol += delta

        # validate safety limits
        if self.vol < 0:
            info = dict(safety='empty')
            out += self.vol
            self.vol = 0
        elif self.vol < self.vol_safety_factor:
            info = dict(safety='lower')
        elif self.vol > self.vol_max:
            info = dict(safety='overflow')
            out += self.vol - self.vol_max
            self.vol = self.vol_max
        elif self.vol > (1 - self.vol_safety_factor) * self.vol_max:
            info = dict(safety='upper')
        else:
            info = dict(safety='ok')

        reward = 1 - (self.out_ref - out) ** 2 / self.out_max ** 2

        # calculating penalty
        limit = self.vol_safety_factor * self.vol_max
        violation = np.maximum(np.clip(limit - self.vol, 0, limit),
                               np.clip(self.vol + limit - self.vol_max, 0, limit)) ** 2
        penalty = self.penaltyscale * violation

        return self._get_obs(), reward - penalty, self.steps >= self.steps_max, info

    def render(self, mode='human'):
        from gym.envs.classic_control import rendering

        if self.viewer is None:
            self.viewer = rendering.Viewer(500, 500)
            self.viewer.set_bounds(-5, 5, -1, 9)
        # widths
        side, tip = 2.5, .4
        # heights
        top = 8
        top_margin = top - top * self.vol_safety_factor
        bot_margin = top * self.vol_safety_factor
        bot = 0
        # frame
        self.viewer.draw_polygon(
            [(tip, bot), (side, bot_margin), (side, top), (-side, top), (-side, bot_margin), (-tip, bot)], filled=False)

        # dashes
        for height in [top_margin, bot_margin]:
            for pair in np.linspace(side, -side, 20).reshape((-1, 2)).tolist():
                self.viewer.draw_line(*zip(pair, [height, height]))

        # filling
        top *= self.vol / self.vol_max
        if top < bot_margin:
            # correct drawing width if we are in the tapered end
            side = tip + (side - tip) * (top - bot) / (bot_margin - bot)
            bot_margin = top
        self.viewer.draw_polygon(
            [(tip, bot), (side, bot_margin), (side, top), (-side, top), (-side, bot_margin), (-tip, bot)])

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def __str__(self):
        return f'{self.__class__}(out_ref={self.out_ref}, ' \
               f'vol=[0,{self.vol_safety_factor},{(1 - self.vol_safety_factor) * self.vol_max},{self.vol_max}], ' \
               f'max_steps={self.steps_max}, penalty={self.penaltyscale}'

    def _get_obs(self):
        return np.array([self.last_in, self.vol / self.vol_max])
