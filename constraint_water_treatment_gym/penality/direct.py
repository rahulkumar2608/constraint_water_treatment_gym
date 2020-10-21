from constraint_water_treatment_gym.penality.base import Penalty


class DirectPenalty(Penalty):
    """
    This Penalty function directly calculates the penalty from the existing state and ignores the penalty
    """

    def __init__(self, low, high):
        super().__init__()
        self.limits = (low, high)

    def value(self, policy, obs):
        # err = (obs - np.clip(obs,*self.limits))**2
        pass
