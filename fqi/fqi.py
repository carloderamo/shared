import numpy as np
from tqdm import trange

from mushroom.algorithms.value.batch_td import BatchTD


class FQI(BatchTD):
    """
    Fitted Q-Iteration algorithm.
    "Tree-Based Batch Mode Reinforcement Learning", Ernst D. et al.. 2005.

    """
    def __init__(self, approximator, policy, mdp_info, n_iterations,
                 n_actions_per_head, fit_params=None, approximator_params=None,
                 quiet=False):
        """
        Constructor.

        Args:
            n_iterations (int): number of iterations to perform for training;
            quiet (bool, False): whether to show the progress bar or not.

        """
        self._n_iterations = n_iterations
        self._n_actions_per_head = n_actions_per_head
        self._quiet = quiet

        self._qs = list()

        super().__init__(approximator, policy, mdp_info, fit_params,
                         approximator_params)

        self._target = None

    def fit(self, dataset):
        """
        Fit loop.

        """
        s = np.array([d[0][0] for d in dataset]).ravel()
        games = np.unique(s)
        d = list()
        idxs = list()
        for n, g in enumerate(games):
            idx = np.argwhere(s == g).ravel()
            for i in idx:
                d.append(dataset[i])
            idxs.append(idx % (n + 1))

        idxs = np.array(idxs)

        state, action, reward, next_state, absorbing, _ = self.parse_dataset(d)

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
                q *= 1 - absorbing

            max_q = np.max(q, axis=1)
            self._target = reward + self.mdp_info.gamma * max_q

        self.approximator.fit(state, action, self._target, idx=idxs,
                              **self._fit_params)

        self._qs.append(self.approximator.predict(state, action, idx=idxs))

    def parse_dataset(self, dataset):
        assert len(dataset) > 0

        shape = dataset[0][0][1].shape

        state = np.ones((len(dataset),) + shape)
        action = np.ones((len(dataset),) + dataset[0][1].shape)
        reward = np.ones(len(dataset))
        next_state = np.ones((len(dataset),) + shape)
        absorbing = np.ones(len(dataset))
        last = np.ones(len(dataset))

        for i in range(len(dataset)):
            state[i, ...] = dataset[i][0][1]
            action[i, ...] = dataset[i][1]
            reward[i] = dataset[i][2]
            next_state[i, ...] = dataset[i][3][1]
            absorbing[i] = dataset[i][4]
            last[i] = dataset[i][5]

        return np.array(state), np.array(action), np.array(reward), np.array(
            next_state), np.array(absorbing), np.array(last)
