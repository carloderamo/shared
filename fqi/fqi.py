from copy import deepcopy

import numpy as np
from tqdm import trange

from mushroom.algorithms.agent import Agent
from mushroom.approximators.regressor import Regressor


def parse_multi_dataset(dataset, n_input_per_mdp, max_n_state):
    assert len(dataset) > 0

    max_state_shape = (max_n_state,)

    idxs = np.ones(len(dataset), dtype=int)
    state = np.ones((len(dataset),) + max_state_shape)
    action = np.ones((len(dataset),) + (1,))
    reward = np.ones(len(dataset))
    next_state = np.ones((len(dataset),) + max_state_shape)
    absorbing = np.ones(len(dataset))
    last = np.ones(len(dataset))

    for i in range(len(dataset)):
        idxs[i] = dataset[i][0][0]
        state[i, :n_input_per_mdp[idxs[i]][0]] = dataset[i][0][1]
        action[i, :] = dataset[i][1]
        reward[i] = dataset[i][2]
        next_state[i, :n_input_per_mdp[idxs[i]][0]] = dataset[i][3][1]
        absorbing[i] = dataset[i][4]
        last[i] = dataset[i][5]

    return np.array(idxs), np.array(state), np.array(action), np.array(reward), \
           np.array(next_state), np.array(absorbing), np.array(last)


class FQI(Agent):
    def __init__(self, approximator, policy, mdp_info, n_iterations,
                 n_actions_per_head, n_input_per_mdp=None, n_games=1,
                 reg_type=None, fit_params=None, approximator_params=None,
                 quiet=False):

        if reg_type == 'l1-weights' or reg_type == 'gl1-weights':
            self._get_features = False
            self._get_weights = True
        else:
            self._get_features = True
            self._get_weights = False

        self._n_iterations = n_iterations
        self._quiet = quiet

        self._fit_params = dict() if fit_params is None else fit_params
        self._approximator_params = dict() if approximator_params is None else \
            deepcopy(approximator_params)

        self._n_games = n_games
        if n_input_per_mdp is None:
            self._n_input_per_mdp = [mdp_info.observation_space.shape
                                     for _ in range(self._n_games)]
        else:
            self._n_input_per_mdp = n_input_per_mdp
        self._n_action_per_head = n_actions_per_head

        self._max_n_state = 0
        for s in self._n_input_per_mdp:
            print('s: ', s[0])
            print('max: ', self._max_n_state)
            self._max_n_state = np.maximum(s[0], self._max_n_state)
            print('new max: ', self._max_n_state)
            print('-----------------------------------------------------------')



        self.approximator = Regressor(approximator,
                                      **self._approximator_params)
        policy.set_q(self.approximator)

        self._target = None

        super().__init__(policy, mdp_info)

    def fit(self, dataset):
        for _ in trange(self._n_iterations, dynamic_ncols=True,
                        disable=self._quiet, leave=False):
            idxs, state, action, reward, next_state, absorbing, _ = \
                parse_multi_dataset(dataset, self._n_input_per_mdp,
                                    self._max_n_state)

            print('shape of state', state.shape)

            if self._target is None:
                self._target = reward
            else:
                q = self.approximator.predict(next_state, idx=idxs)
                if np.any(absorbing):
                    q *= 1 - absorbing.reshape(-1, 1)

                gamma_max_q = np.ones(len(q))

                for i, q_i in enumerate(q):
                    n_actions = self._n_action_per_head[idxs[i]][0]
                    gamma_max_q[i] = np.max(q_i[:n_actions])
                    gamma_max_q[i] *= self.mdp_info.gamma[idxs[i]]

                self._target = reward + gamma_max_q

            self.approximator.fit(state, action, self._target, idx=idxs,
                                  get_features=self._get_features,
                                  get_weights=self._get_weights,
                                  **self._fit_params)
