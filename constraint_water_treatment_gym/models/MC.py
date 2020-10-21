from typing import List, Union, Optional

import numpy as np


class MarkovChain:
    def __init__(self, values: Union[np.ndarray, List[float]], transitions: Union[np.ndarray, list],
                 start: Optional[int] = None):
        transitions = np.array(transitions)
        if len(transitions.shape) != 2 or not (
                len(values) == transitions.shape[0] == transitions.shape[1]) or (
                start is not None and not 0 <= start < len(values)):
            raise ValueError('shape missmatch')

        self.n = len(values)
        self.values = values
        self.trans = transitions
        if start is not None:
            self.state = start
        else:
            self.reset()

    @property
    def stationary(self):
        # # empirical implementation
        # pi = [.5,.5]
        # for i in range(1000):
        #     pi= pi@self.trans

        w, v = np.linalg.eig(self.trans.T)
        one_eigenvalue = np.abs(w - 1) < 1e-7
        if np.count_nonzero(one_eigenvalue) != 1:
            ValueError('could not calculate expectation, not exactly one "1"-valued eigenvalue')
        stationary = v[:, one_eigenvalue].sum(axis=1)
        stationary /= stationary.sum()
        return stationary

    def reset(self):
        self.state = np.random.choice(self.n, p=self.stationary)

    def step(self):
        self.state = np.random.choice(self.n, p=self.trans[self.state])
        return self.values[self.state]

    def expectation(self):
        return np.array(self.values) @ self.stationary

    def __str__(self):
        return f'{self.__class__}:\n' \
               f'v = {self.values}\n' \
               f'tran = {self.trans}\n' \
               f'exp = {self.expectation()}'


if __name__ == '__main__':
    mc = MarkovChain([1, 3], [[.99, .01], [0.5, .5]])
    print(mc.expectation())
    print([mc.step() for i in range(10)])
    print([mc.step() for i in range(10)])
    print([mc.step() for i in range(10)])
    print([mc.step() for i in range(10)])
