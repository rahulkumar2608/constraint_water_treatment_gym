import gym
import numpy as np


# TODO: time resolution


class WaterTreatmentEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, inflow=lambda: .5, out_ref=.5, out_max=1, safety_factor: float = .1, volume_init: float = None,
                 volume_max: float = 100.0,
                 penalty_scale=100, max_eps_len=400):
        self.inflow = inflow
        self.out_ref = out_ref
        self.out_max = out_max
        self.volume_init = volume_init
        self.last_in = None
        self.volume = None
        self.volume_safety_factor = safety_factor
        self.volume_max = volume_max
        self.penaltyscale = penalty_scale
        self.steps = None
        self.max_eps_len = float('inf') if max_eps_len is None else max_eps_len

        self.viewer = None

        self.action_space = gym.spaces.Box(np.zeros(1), np.ones(1))
        self.observation_space = gym.spaces.Box(np.zeros(2), np.ones(2))
        self.seed()

    def reset(self):
        if self.volume_init is None:
            self.volume = np.random.uniform(self.volume_safety_factor,
                                            (1 - self.volume_safety_factor) * self.volume_max)
        else:
            self.volume = self.volume_init * self.volume_max
        self.last_in = self.inflow()
        self.steps = 0
        return self._get_obs()

    def step(self, action: np.ndarray):
        self.steps += 1
        if self.steps > self.max_eps_len:
            raise ValueError("called an exceded env. please reset env before calling step again")

        self.last_in = self.inflow()
        out = self.out_max * action.squeeze()

        delta = self.last_in - out
        self.volume += delta

        # validate safety limits
        if self.volume < 0:
            info = dict(safety='empty')
            out += self.volume
            self.volume = 0
        elif self.volume < self.volume_safety_factor:
            info = dict(safety='lower')
        elif self.volume > self.volume_max:
            info = dict(safety='overflow')
            out += self.volume - self.volume_max
            self.volume = self.volume_max
        elif self.volume > (1 - self.volume_safety_factor) * self.volume_max:
            info = dict(safety='upper')
        else:
            info = dict(safety='ok')

        reward = 1 - (self.out_ref - out) ** 2 / self.out_max ** 2

        # calculating penalty
        limit = self.volume_safety_factor * self.volume_max
        violation = np.maximum(np.clip(limit - self.volume, 0, limit),
                               np.clip(self.volume + limit - self.volume_max, 0, limit)) ** 2
        penalty = self.penaltyscale * violation

        return self._get_obs(), reward - penalty, self.steps >= self.max_eps_len, info

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

    def __repr__(self):
        return f'{self.__class__}(out_ref={self.out_ref}, ' \
               f'vol=[0,{self.volume_safety_factor},{(1 - self.volume_safety_factor) * self.volume_max},{self.volume_max}], ' \
               f'max_steps={self.max_eps_len}, penalty={self.penaltyscale}'

    def _get_obs(self):
        return np.array([self.last_in, self.volume / self.volume_max])
