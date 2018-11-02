from copy import deepcopy

import numpy as np
from scipy.special import logsumexp

from mushroom.algorithms.agent import Agent
from mushroom.approximators.regressor import Ensemble, Regressor

from replay_memory import ReplayMemory


class DQN(Agent):
    """
    Deep Q-Network algorithm.
    "Human-Level Control Through Deep Reinforcement Learning".
    Mnih V. et al.. 2015.

    """
    def __init__(self, approximator, policy, mdp_info, batch_size,
                 initial_replay_size, max_replay_size, n_actions_per_head,
                 history_length=4, n_input_per_mdp=None,
                 target_update_frequency=2500, fit_params=None,
                 approximator_params=None, n_games=1, clip_reward=True,
                 reg_type=None, dtype=np.uint8):
        if reg_type == 'l1-weights' or 'gl1-weights':
            self._get_features = False
            self._get_weights = True
        else:
            self._get_features = True
            self._get_weights = False

        self._fit_params = dict() if fit_params is None else fit_params

        self._batch_size = batch_size
        self._n_games = n_games
        self._clip_reward = clip_reward
        if n_input_per_mdp is None:
            self._n_input_per_mdp = [mdp_info.observation_space.shape
                                     for _ in range(self._n_games)]
        else:
            self._n_input_per_mdp = n_input_per_mdp
        self._n_action_per_head = n_actions_per_head
        self._history_length = history_length
        self._max_actions = max(n_actions_per_head)[0]
        self._target_update_frequency = target_update_frequency

        self._replay_memory = [
            ReplayMemory(initial_replay_size,
                         max_replay_size) for _ in range(self._n_games)
        ]

        self._n_updates = 0

        apprx_params_train = deepcopy(approximator_params)
        apprx_params_target = deepcopy(approximator_params)
        self.approximator = Regressor(approximator, **apprx_params_train)
        self.target_approximator = Regressor(approximator,
                                             **apprx_params_target)
        policy.set_q(self.approximator)

        self.target_approximator.model.set_weights(
            self.approximator.model.get_weights())

        super().__init__(policy, mdp_info)

        n_samples = self._batch_size * self._n_games
        self._state_idxs = np.zeros(n_samples, dtype=np.int)
        self._state = np.zeros(
            ((n_samples,
             self._history_length) + self.mdp_info.observation_space.shape),
            dtype=dtype
        ).squeeze()
        self._action = np.zeros((n_samples, 1))
        self._reward = np.zeros(n_samples)
        self._next_state_idxs = np.zeros(n_samples, dtype=np.int)
        self._next_state = np.zeros(
            ((n_samples,
             self._history_length) + self.mdp_info.observation_space.shape),
            dtype=dtype
        ).squeeze()
        self._absorbing = np.zeros(n_samples)

        self.v_list = list()

    def fit(self, dataset):
        s = np.array([d[0][0] for d in dataset]).ravel()
        games = np.unique(s)
        for g in games:
            idxs = np.argwhere(s == g).ravel()
            d = list()
            for idx in idxs:
                d.append(dataset[idx])

            self._replay_memory[g].add(d)

        fit_condition = np.all([rm.initialized for rm in self._replay_memory])

        if fit_condition:
            for i in range(len(self._replay_memory)):
                game_state, game_action, game_reward, game_next_state,\
                    game_absorbing, _ = self._replay_memory[i].get(
                        self._batch_size)

                start = self._batch_size * i
                stop = start + self._batch_size

                self._state_idxs[start:stop] = np.ones(self._batch_size) * i
                self._state[start:stop, :self._n_input_per_mdp[i][0]] = game_state
                self._action[start:stop] = game_action
                self._reward[start:stop] = game_reward
                self._next_state_idxs[start:stop] = np.ones(self._batch_size) * i
                self._next_state[start:stop, :self._n_input_per_mdp[i][0]] = game_next_state
                self._absorbing[start:stop] = game_absorbing

            if self._clip_reward:
                reward = np.clip(self._reward, -1, 1)
            else:
                reward = self._reward

            q_next = self._next_q()
            q = reward + q_next

            self.approximator.fit(self._state, self._action, q,
                                  idx=self._state_idxs,
                                  get_features=self._get_features,
                                  get_weights=self._get_weights,
                                  **self._fit_params)

            self._n_updates += 1

            if self._n_updates % self._target_update_frequency == 0:
                self._update_target()

    def get_shared_weights(self):
        return self.approximator.model.network.get_shared_weights()

    def set_shared_weights(self, weights):
        self.approximator.model.network.set_shared_weights(weights)

    def freeze_shared_weights(self):
        return self.approximator.model.network.freeze_shared_weights()

    def unfreeze_shared_weights(self):
        self.approximator.model.network.unfreeze_shared_weights()

    def _update_target(self):
        """
        Update the target network.

        """
        self.target_approximator.model.set_weights(
            self.approximator.model.get_weights())

    def _next_q(self):
        q = self.target_approximator.predict(self._next_state,
                                             idx=self._next_state_idxs)

        out_q = np.zeros(self._batch_size * self._n_games)

        v_list = list()
        for i in range(self._n_games):
            start = self._batch_size * i
            stop = start + self._batch_size
            if np.any(self._absorbing[start:stop]):
                q[start:stop] *= 1 - self._absorbing[start:stop].reshape(-1, 1)

            n_actions = self._n_action_per_head[i][0]
            out_q[start:stop] = np.max(q[start:stop, :n_actions], axis=1)
            out_q[start:stop] *= self.mdp_info.gamma[i]

            v_list.append(out_q[start:stop].mean())

        self.v_list.append(np.array(v_list))

        return out_q


class DoubleDQN(DQN):
    def _next_q(self):
        q = self.approximator.predict(self._next_state,
                                      idx=self._next_state_idxs)
        double_q = self.target_approximator.predict(self._next_state,
                                                    idx=self._next_state_idxs)
        if np.any(self._absorbing):
            double_q *= 1 - self._absorbing.reshape(-1, 1)

        out_q = np.zeros(self._batch_size * self._n_games)

        v_list = list()
        for i in range(self._n_games):
            start = self._batch_size * i
            stop = start + self._batch_size

            n_actions = self._n_action_per_head[i][0]
            idxs = np.argmax(q[start:stop, :n_actions], axis=1)
            out_q[start:stop] = double_q[np.arange(start, stop),
                                         idxs] * self.mdp_info.gamma[i]

            v_list.append(out_q[start:stop].mean())

        self.v_list.append(np.array(v_list))

        return out_q
