import numpy as np

from mushroom.algorithms.value.batch_td import BatchTD



class FQI(BatchTD):
    """
    Fitted Q-Iteration algorithm.
    "Tree-Based Batch Mode Reinforcement Learning", Ernst D. et al.. 2005.

    """
    def __init__(self, approximator, policy, mdp_info, n_iterations,
                 n_actions_per_head, fit_params=None,
                 approximator_params=None, quiet=False):
        """
        Constructor.

        Args:
            n_iterations (int): number of iterations to perform for training;
            quiet (bool, False): whether to show the progress bar or not.

        """
        self._n_iterations = n_iterations
        self._n_actions_per_head = n_actions_per_head
        self._n_games = len(self._n_actions_per_head)
        self._quiet = quiet

        super().__init__(mdp_info, policy, approximator, approximator_params,
                         fit_params)

        self._target = None

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
