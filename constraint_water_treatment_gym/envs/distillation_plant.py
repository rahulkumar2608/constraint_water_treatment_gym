import gym
import numpy as np


# TODO: time resolution


class WaterTreatmentEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, inflow=lambda: .5, out_ref=.5, out_max=1, safety_factor: float = .1, volume_max: float = 100.0):
        self.inflow = inflow
        self.out_ref = out_ref
        self.out_max = out_max
        self.last_in = None
        self.volume = None
        self.volume_safety_factor = safety_factor
        self.volume_max = volume_max

        self.viewer = None

        self.action_space = gym.spaces.Box(np.zeros(1), np.ones(1))
        self.observation_space = gym.spaces.Box(np.full(2, -1), np.full(2, 1))
        self.seed()

    def reset(self):
        self.volume = 30
        self.last_in = self.inflow()
        return self._get_obs()

    def step(self, action: np.ndarray):
        self.last_in = self.inflow()
        out = self.out_max * action.squeeze()

        delta = self.last_in - out
        self.volume += delta

        # validate and fail overflow
        if self.volume > self.volume_max:
            return self._get_obs(), -100, True, dict(safety='overflow')

        # validate safety limits
        if self.volume < 0:
            info = dict(safety='empty')
        if self.volume < self.volume_safety_factor:
            info = dict(safety='lower')
        elif self.volume > self.volume_max - self.volume_safety_factor:
            info = dict(safety='upper')
        else:
            info = dict(safety='ok')

        # validate and correct underflow
        underflow = min(self.volume, 0)
        out += underflow
        self.volume -= underflow

        cost = (self.out_ref - out) ** 2

        return self._get_obs(), 1 - cost, False, info

    def render(self, mode='human'):
        from gym.envs.classic_control import rendering

        if self.viewer is None:
            self.viewer = rendering.Viewer(500, 500)
            self.viewer.set_bounds(-3, 3, -1, 9)
        # widths
        side, tip = 2.5, .4
        # heights
        top = 8
        top_margin = top - top * self.volume_safety_factor
        bot_margin = top * self.volume_safety_factor
        bot = 0
        # frame
        self.viewer.draw_polygon(
            [(tip, bot), (side, bot_margin), (side, top), (-side, top), (-side, bot_margin), (-tip, bot)], filled=False)

        # dashes
        for height in [top_margin, bot_margin]:
            for pair in np.linspace(side, -side, 20).reshape((-1, 2)).tolist():
                self.viewer.draw_line(*zip(pair, [height, height]))

        # filling
        top *= self.volume / self.volume_max
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

    def _get_obs(self):
        return np.array([self.last_in, self.volume])
