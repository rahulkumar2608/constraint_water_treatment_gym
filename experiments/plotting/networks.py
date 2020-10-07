import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch


def create_heatmap(timestamp, v, name, domain=None):
    if domain is None:
        domain = [[0, 1], [0, 1]]
    inflow_idx = np.linspace(*domain[0], 51)
    vol_idx = np.linspace(*domain[1], 51)

    fig, ax = plt.subplots()
    V = np.array([[v(i, vol) for i in inflow_idx] for vol in vol_idx])
    df = pd.DataFrame(V, columns=pd.Index([f'{v:,.1f}' for v in inflow_idx], name='inflow'),
                      index=pd.Index([f'{v:,.1f}' for v in vol_idx], name='volume'))
    sns.heatmap(df, ax=ax, cmap='icefire', xticklabels=5, yticklabels=5)
    ax.set_title(name)
    ax.invert_yaxis()
    fig.savefig(f'{timestamp}/{name.lower()}.pdf')


def visualize_nets(env, model, timestamp, domain=None, value=True, policy=True):
    if value:
        def v(i, vol):
            obs = np.array([i, vol])
            observation = obs.reshape((-1,) + env.observation_space.shape)
            observation = torch.as_tensor(observation)
            return model.policy.forward(observation, deterministic=True)[1].item()

        create_heatmap(timestamp, v, 'Value', domain=domain)

    if policy:
        def pi(i, vol):
            return model.predict(np.array([i, vol]), deterministic=True)[0].item()

        create_heatmap(timestamp, pi, 'Policy', domain=domain)
