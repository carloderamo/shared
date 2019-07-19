from copy import deepcopy

import numpy as np

from mushroom.algorithms.agent import Agent
from mushroom.approximators.regressor import Regressor

from replay_memory import PrioritizedReplayMemory, ReplayMemory


class DQN(Agent):
    """
    Deep Q-Network algorithm.
    "Human-Level Control Through Deep Reinforcement Learning".
    Mnih V. et al.. 2015.

    """
    def __init__(self, approximator, policy, mdp_info, batch_size,
                 initial_replay_size, max_replay_size, n_actions_per_head,
                 history_length=4, n_input_per_mdp=None, replay_memory=None,
                 target_update_frequency=2500, fit_params=None,
                 approximator_params=None, n_games=1, clip_reward=True,
                 sampling_prior=False, dtype=np.uint8):
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

        if replay_memory is not None:
            self._replay_memory = replay_memory
            if isinstance(replay_memory[0], PrioritizedReplayMemory):
                self._fit = self._fit_prioritized
            else:
                self._fit = self._fit_standard
        else:
            self._replay_memory = [ReplayMemory(
                initial_replay_size, max_replay_size) for _ in range(self._n_games)
            ]
            self._fit = self._fit_standard

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
        self._action = np.zeros((n_samples, 1), dtype=np.int)
        self._reward = np.zeros(n_samples)
        self._next_state_idxs = np.zeros(n_samples, dtype=np.int)
        self._next_state = np.zeros(
            ((n_samples,
             self._history_length) + self.mdp_info.observation_space.shape),
            dtype=dtype
        ).squeeze()
        self._absorbing = np.zeros(n_samples)
        self._idxs = np.zeros(n_samples, dtype=np.int)
        self._is_weight = np.zeros(n_samples)
        self.replay_memory_priorities = np.ones(self._n_games) / self._n_games

        self._sampling_prior = sampling_prior

    def fit(self, dataset):
        self._fit(dataset)

        self._n_updates += 1
        if self._n_updates % self._target_update_frequency == 0:
            self._update_target()

    def _fit_standard(self, dataset):
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

            if self._sampling_prior:
                self._update_er_priorities()

            self.approximator.fit(self._state, self._action, q,
                                  idx=self._state_idxs, **self._fit_params)

    def _fit_prioritized(self, dataset):
        s = np.array([d[0][0] for d in dataset]).ravel()
        games = np.unique(s)
        for g in games:
            idxs = np.argwhere(s == g).ravel()
            d = list()
            for idx in idxs:
                d.append(dataset[idx])

            self._replay_memory[g].add(
                d, np.ones(len(d)) * self._replay_memory[g].max_priority
            )

        fit_condition = np.all([rm.initialized for rm in self._replay_memory])

        if fit_condition:
            for i in range(len(self._replay_memory)):
                game_state, game_action, game_reward, game_next_state,\
                    game_absorbing, _, game_idxs, game_is_weight =\
                    self._replay_memory[i].get(self._batch_size)

                start = self._batch_size * i
                stop = start + self._batch_size

                self._state_idxs[start:stop] = np.ones(self._batch_size) * i
                self._state[start:stop, :self._n_input_per_mdp[i][0]] = game_state
                self._action[start:stop] = game_action
                self._reward[start:stop] = game_reward
                self._next_state_idxs[start:stop] = np.ones(self._batch_size) * i
                self._next_state[start:stop, :self._n_input_per_mdp[i][0]] = game_next_state
                self._absorbing[start:stop] = game_absorbing
                self._idxs[start:stop] = game_idxs
                self._is_weight[start:stop] = game_is_weight

            if self._clip_reward:
                reward = np.clip(self._reward, -1, 1)
            else:
                reward = self._reward

            q_next = self._next_q()
            q = reward + q_next
            q_current = self.approximator.predict(self._state, self._action,
                                                  idx=self._state_idxs)
            td_error = q - q_current

            if self._sampling_prior:
                self._update_er_priorities()

            for er in self._replay_memory:
                er.update(td_error, self._idxs)

            self.approximator.fit(self._state, self._action, q,
                                  weights=self._is_weight,
                                  idx=self._state_idxs,
                                  **self._fit_params)

    def _update_er_priorities(self):
        grads = np.zeros(self._n_games)
        for i in range(self._n_games):
            idxs = np.argwhere(self._state_idxs == i).ravel()
            self.approximator.fit(
                self._state[idxs], self._action[idxs], q[idxs],
                weights=self._is_weight[idxs],
                idx=self._state_idxs[idxs],
                er_idx=i,
                params=self.approximator.model.network.get_shared_weights_tensor(),
                **self._fit_params
            )
            grads[i] = self.approximator.model.grad
        self.replay_memory_priorities = grads / grads.sum()

    def get_shared_weights(self):
        return self.approximator.model.network.get_shared_weights()

    def set_shared_weights(self, weights):
        self.approximator.model.network.set_shared_weights(weights)

    def freeze_shared_weights(self):
        self.approximator.model.network.freeze_shared_weights()

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

        for i in range(self._n_games):
            start = self._batch_size * i
            stop = start + self._batch_size
            if np.any(self._absorbing[start:stop]):
                q[start:stop] *= 1 - self._absorbing[start:stop].reshape(-1, 1)

            n_actions = self._n_action_per_head[i][0]
            out_q[start:stop] = np.max(q[start:stop, :n_actions], axis=1)
            out_q[start:stop] *= self.mdp_info.gamma[i]

        return out_q
