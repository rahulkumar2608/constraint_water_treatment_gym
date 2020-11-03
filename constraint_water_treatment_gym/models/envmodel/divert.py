import torch as th


class DivertModel:
    """
    selects inflow to push to the closest boundary (close inflow if low volume, max inflow above 50% vol)
    """

    def __call__(self, act: th.tensor, obs: th.tensor):
        if obs[1] < .5:
            return th.tensor([0, obs[1] + act])
        else:
            return th.tensor([1, obs[1] + act + 1])
