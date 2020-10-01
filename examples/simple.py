import gym
import numpy as np

if __name__ == '__main__':
    env = gym.make('constraint_water_treatment_gym:distillation-plant-v0')

    env.reset()
    for _ in range(1000):
        env.render()
        # _, r, _, _ = env.step(env.action_space.sample())  # pick three continous control actions randomly
        _, r, _, _ = env.step(np.full(1, .5))  # pick three continous control actions randomly
        print(r)
    env.close()
