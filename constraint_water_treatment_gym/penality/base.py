class Penalty:
    def __init__(self):
        pass

    def value(self, policy, obs):
        pass

    def grad(self, policy, obs):
        """
        TODO: this might be technically not so easy. The penalty needs to be a function but when backpropagating
        TODO: i will need to calculate the gradient of the policy parameters

        :param policy: Pytorch tensor
        :param obs:
        :return: gradient of the penalty function
        """
        pass
