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
        self._explorative_epsilons = [deepcopy(epsilon)
                                      for _ in range(len(n_actions_per_head))]
        self._epsilons = [None] * len(n_actions_per_head)

    def __call__(self, *args):
        idx = args[0]
        state = np.array(args[1])
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
        idx = state[0]
        state = np.array(state[1])
        if not np.random.uniform() < self._epsilons[idx](state):
            q = self._approximator.predict(
                state, idx=np.array([idx]))[:self._n_actions_per_head[idx][0]]
            max_a = np.argwhere(q == np.max(q)).ravel()

            if len(max_a) > 1:
                max_a = np.array([np.random.choice(
                    max_a[max_a < self._n_actions_per_head[idx][0]]
                )])

            return max_a

        return np.array([np.random.choice(self._n_actions_per_head[idx][0])])

    def set_epsilon(self, epsilon):
        assert isinstance(epsilon, Parameter) or epsilon is None

        if epsilon is None:
            for i in range(len(self._epsilons)):
                self._epsilons[i] = self._explorative_epsilons[i]
        else:
            for i in range(len(self._epsilons)):
                self._epsilons[i] = epsilon

    def update(self, state):
        idx = state[0]
        self._epsilons[idx].update(state)


class EpsGreedyEnsemble(TDPolicy):
    def __init__(self, epsilon, n):
        """
        Constructor.

        Args:
            epsilon (Parameter): the exploration coefficient. It indicates
                the probability of performing a random actions in the current
                step.

        """
        super().__init__()

        self._explorative_epsilons = [deepcopy(epsilon) for _ in range(n)]
        self._epsilons = [None] * n

    def __call__(self, *args):
        state = args[0]
        idx = np.asscalar(state[0])
        state = np.array(state[1])

        q = self._approximator[idx].predict(np.expand_dims(state, axis=0)).ravel()
        max_a = np.argwhere(q == np.max(q)).ravel()

        p = self._epsilons[idx].get_value(state) / self._approximator[idx].n_actions

        if len(args) == 2:
            action = args[1]
            if action in max_a:
                return p + (1. - self._epsilons[idx].get_value(state)) / len(max_a)
            else:
                return p
        else:
            probs = np.ones(self._approximator[idx].n_actions) * p
            probs[max_a] += (1. - self._epsilons[idx].get_value(state)) / len(max_a)

            return probs

    def draw_action(self, state):
        idx = state[0]
        state = np.array(state[1])
        if not np.random.uniform() < self._epsilons[idx](state):
            q = self._approximator[idx].predict(state)
            max_a = np.argwhere(q == np.max(q)).ravel()

            if len(max_a) > 1:
                max_a = np.array([np.random.choice(max_a)])

            return max_a

        return np.array([np.random.choice(self._approximator[idx].n_actions)])

    def set_epsilon(self, epsilon):
        assert isinstance(epsilon, Parameter) or epsilon is None

        if epsilon is None:
            for i in range(len(self._epsilons)):
                self._epsilons[i] = self._explorative_epsilons[i]
        else:
            for i in range(len(self._epsilons)):
                self._epsilons[i] = epsilon

    def update(self, state):
        idx = state[0]
        self._epsilons[idx].update(state)
