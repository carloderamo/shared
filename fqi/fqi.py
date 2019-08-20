import numpy as np
from tqdm import trange

from mushroom.algorithms.value.batch_td import BatchTD


class FQI(BatchTD):
    """
    Fitted Q-Iteration algorithm.
    "Tree-Based Batch Mode Reinforcement Learning", Ernst D. et al.. 2005.

    """
    def __init__(self, approximator, policy, mdp_info, n_iterations,
                 discrete_actions, fit_params=None, approximator_params=None,
                 quiet=False):
        """
        Constructor.

        Args:
            n_iterations (int): number of iterations to perform for training;
            quiet (bool, False): whether to show the progress bar or not.

        """
        self._n_iterations = n_iterations
        self._discrete_actions = discrete_actions
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

        state_idxs = np.array(idxs).ravel()

        state, action, reward, next_state, absorbing, _ = self.parse_dataset(d)
        state_action = np.append(state, action, 1)
        next_state_repeat = np.repeat(next_state, len(self._discrete_actions))
        discrete_action_repeat = np.expand_dims(
            self._discrete_actions, 0).repeat(len(reward), 0)
        next_state_action = np.append(next_state_repeat.reshape(-1, 1),
                                      discrete_action_repeat.reshape(-1, 1), 1)
        next_state_idxs = np.repeat(idxs, len(self._discrete_actions))
        absorbing = np.repeat(absorbing, len(self._discrete_actions))

        for _ in trange(self._n_iterations, dynamic_ncols=True,
                        disable=self._quiet, leave=False):
            self._fit(state_action, reward, next_state_action, absorbing, state_idxs,
                      next_state_idxs)

    def _fit(self, state_action, reward, next_state_action, absorbing, state_idxs,
             next_state_idxs):
        """
        Single fit iteration.

        Args:
            x (list): the dataset.

        """
        if self._target is None:
            self._target = reward.copy()
        else:
            q = self.approximator.predict(next_state_action,
                                          idx=next_state_idxs)
            if np.any(absorbing):
                q *= 1 - absorbing

            q = q.reshape(len(reward), len(self._discrete_actions))
            max_q = np.max(q, axis=1)
            self._target = reward + self.mdp_info.gamma * max_q

        self.approximator.fit(state_action, self._target, idx=state_idxs,
                              **self._fit_params)

        self._qs.append(self.approximator.predict(state_action, idx=state_idxs))

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
