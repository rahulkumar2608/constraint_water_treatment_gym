import gym
from gym.wrappers.monitoring.video_recorder import VideoRecorder

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

env = gym.make('constraint_water_treatment_gym:distillation-plant-v0')
env = Monitor(env)

model = PPO('MlpPolicy', env, verbose=1, tensorboard_log='trainlog/')
model.learn(total_timesteps=100000)

rec = VideoRecorder(env, 'vid.mp4')
obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.unwrapped.render()
    rec.capture_frame()
    if done:
        obs = env.reset()

env.close()
rec.close()
