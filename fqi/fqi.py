import numpy as np
from tqdm import trange

from mushroom.algorithms.value.batch_td import BatchTD
from mushroom.utils.dataset import parse_dataset


class FQI(BatchTD):
    """
    Fitted Q-Iteration algorithm.
    "Tree-Based Batch Mode Reinforcement Learning", Ernst D. et al.. 2005.

    """
    def __init__(self, approximator, policy, mdp_info, n_iterations,
                 n_actions_per_head, test_states, test_actions, test_idxs,
                 len_datasets, fit_params=None, approximator_params=None,
                 quiet=False):
        """
        Constructor.

        Args:
            n_iterations (int): number of iterations to perform for training;
            quiet (bool, False): whether to show the progress bar or not.

        """
        self._n_iterations = n_iterations
        self._n_actions_per_head = n_actions_per_head
        self._n_games = len(self._n_actions_per_head)
        self._test_states = test_states
        self._test_actions = test_actions
        self._test_idxs = test_idxs
        self._len_datasets = len_datasets
        self._quiet = quiet

        self._qs = list()

        super().__init__(approximator, policy, mdp_info, fit_params,
                         approximator_params)

        self._target = None

    def fit(self, dataset):
        """
        Fit loop.

        """
        idxs = list()
        for i, l in enumerate(self._len_datasets):
            idxs += (np.ones(l, dtype=np.int) * i).tolist()
        idxs = np.array(idxs)

        state, action, reward, next_state, absorbing, _ = parse_dataset(dataset)

        for _ in trange(self._n_iterations, dynamic_ncols=True,
                        disable=self._quiet, leave=False):
            self._fit(state, action, reward, next_state, absorbing, idxs)

    def _fit(self, state, action, reward, next_state, absorbing, idxs):
        """
        Single fit iteration.

        Args:
            x (list): the dataset.

        """
        if self._target is None:
            self._target = reward.copy()
        else:
            q = self.approximator.predict(next_state, idx=idxs)
            if np.any(absorbing):
                q *= 1 - absorbing.reshape(-1, 1)

            max_q = np.max(q, axis=1)
            self._target = reward + self.mdp_info.gamma * max_q

        self.approximator.fit(state, action, self._target, idx=idxs,
                              **self._fit_params)

        self._qs.append(self.approximator.predict(self._test_states,
                                                  self._test_actions,
                                                  idx=self._test_idxs))
