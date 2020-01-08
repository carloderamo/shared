import numpy as np

from mushroom_rl.utils.replay_memory import PrioritizedReplayMemory, ReplayMemory, SumTree


class ReplayMemory(ReplayMemory):
    def add(self, dataset):
        for i in range(len(dataset)):
            self._states[self._idx] = dataset[i][0][1]
            self._actions[self._idx] = dataset[i][1]
            self._rewards[self._idx] = dataset[i][2]
            self._next_states[self._idx] = dataset[i][3][1]
            self._absorbing[self._idx] = dataset[i][4]
            self._last[self._idx] = dataset[i][5]

            self._idx += 1
            if self._idx == self._max_size:
                self._full = True
                self._idx = 0


class PrioritizedReplayMemory(PrioritizedReplayMemory):
    def __init__(self, initial_size, max_size, alpha, beta,
                 epsilon=.01):
        self._initial_size = initial_size
        self._max_size = max_size
        self._alpha = alpha
        self._beta = beta
        self._epsilon = epsilon

        self._tree = SumTree(max_size)

    def get(self, n_samples):
        states = [None for _ in range(n_samples)]
        actions = [None for _ in range(n_samples)]
        rewards = [None for _ in range(n_samples)]
        next_states = [None for _ in range(n_samples)]
        absorbing = [None for _ in range(n_samples)]
        last = [None for _ in range(n_samples)]

        idxs = np.zeros(n_samples, dtype=np.int)
        priorities = np.zeros(n_samples)

        total_p = self._tree.total_p
        segment = total_p / n_samples

        a = np.arange(n_samples) * segment
        b = np.arange(1, n_samples + 1) * segment
        samples = np.random.uniform(a, b)
        for i, s in enumerate(samples):
            idx, p, data = self._tree.get(s)

            idxs[i] = idx
            priorities[i] = p
            states[i], actions[i], rewards[i], next_states[i], absorbing[i],\
                last[i] = data
            states[i] = np.array(states[i][1])
            next_states[i] = np.array(next_states[i][1])

        sampling_probabilities = priorities / self._tree.total_p
        is_weight = (self._tree.size * sampling_probabilities) ** -self._beta()
        is_weight /= is_weight.max()

        return np.array(states), np.array(actions), np.array(rewards),\
            np.array(next_states), np.array(absorbing), np.array(last),\
            idxs, is_weight
