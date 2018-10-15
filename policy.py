from copy import deepcopy

import numpy as np

from mushroom.policy import ParametricPolicy, TDPolicy
from mushroom.utils.parameters import Parameter


class EpsGreedyMultiple(TDPolicy):
    def __init__(self, epsilon, n_actions_per_head):
        super().__init__()

        assert isinstance(epsilon, Parameter) and isinstance(n_actions_per_head,
                                                             list)
        self._n_actions_per_head = n_actions_per_head

        n_heads = len(n_actions_per_head)

        if isinstance(epsilon, list):
            self._explorative_epsilons = deepcopy(epsilon)
        else:
            self._explorative_epsilons = [deepcopy(epsilon)
                                          for _ in range(n_heads)]
        self._epsilons = [None] * n_heads

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

        if isinstance(epsilon, list):
            self._explorative_epsilons = deepcopy(epsilon)
        else:
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


class OrnsteinUhlenbeckPolicy(ParametricPolicy):
    def __init__(self, mu, sigma, theta, dt, n_actions_per_head,
                 max_action_value, x0=None):

        self._approximator = mu
        self._sigma = sigma
        self._theta = theta
        self._dt = dt
        self._max_action_value = max_action_value
        self._x0 = x0

        self._n_games = len(n_actions_per_head)

        self._n_actions_per_head = n_actions_per_head

    def __call__(self, state, action):
        raise NotImplementedError

    def draw_action(self, state):
        idx = state[0]
        state = state[1]
        mu = self._approximator.predict(state, idx=np.array([idx])) / self._max_action_value[idx]

        x = self._x_prev[idx] - self._theta * self._x_prev[idx] * self._dt + self._sigma *\
            np.sqrt(self._dt) * np.random.normal(size=self._approximator.output_shape)
        self._x_prev[idx] = x

        return mu[:self._n_actions_per_head[idx][0]] + x[:self._n_actions_per_head[idx][0]]

    def set_weights(self, weights):
        self._approximator.set_weights(weights)

    def get_weights(self):
        return self._approximator.get_weights()

    @property
    def weights_size(self):
        return self._approximator.weights_size

    def reset(self):
        self._x_prev = list()
        for i in range(self._n_games):
            self._x_prev.append(self._x0 if self._x0 is not None else np.zeros(self._approximator.output_shape))
