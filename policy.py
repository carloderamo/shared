from copy import deepcopy

import numpy as np

from mushroom.policy import TDPolicy
from mushroom.utils.parameters import Parameter


class EpsGreedyMultiple(TDPolicy):
    def __init__(self, epsilon, n_actions_per_head):
        super().__init__()

        assert isinstance(epsilon, Parameter) and isinstance(n_actions_per_head,
                                                             list)
        self._n_actions_per_head = n_actions_per_head
        self._explorative_epsilons = [
            deepcopy(epsilon)] * len(n_actions_per_head)
        self._epsilons = [None] * len(n_actions_per_head)

    def __call__(self, *args):
        idx = args[0]
        state = args[1]
        q = self._approximator.predict(
            np.expand_dims(state, axis=0),
            idx=idx).ravel()[:self._n_actions_per_head[idx][0]]
        max_a = np.argwhere(q == np.max(q)).ravel()

        p = self._epsilon.get_value(state) / self._n_actions_per_head[idx][0]

        if len(args) == 2:
            action = args[1]
            if action in max_a:
                return p + (1. - self._epsilon.get_value(state)) / len(max_a)
            else:
                return p
        else:
            probs = np.ones(self._n_actions_per_head[idx][0]) * p
            probs[max_a] += (1. - self._epsilon.get_value(state)) / len(max_a)

            return probs

    def draw_action(self, state):
        idx = np.asscalar(state[0])
        state = state[1]
        if not np.random.uniform() < self._epsilons[idx](state):
            q = self._approximator.predict(
                state, idx=np.array([idx]))[:self._n_actions_per_head[idx][0]]
            max_a = np.argwhere(q == np.max(q)).ravel()

            if len(max_a) > 1:
                max_a = np.array([np.random.choice(max_a)])

            return max_a

        return np.array([np.random.choice(self._n_actions_per_head[idx][0])])

    def set_epsilon(self, epsilon):
        assert isinstance(epsilon, Parameter) or epsilon is None

        if epsilon is None:
            self._epsilons = self._explorative_epsilons
        else:
            for i in range(len(self._epsilons)):
                self._epsilons[i] = epsilon

    def update(self, state):
        idx = np.asscalar(state[0])
        self._epsilons[idx].update(state)
