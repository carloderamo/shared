from copy import deepcopy

import numpy as np
from scipy.special import logsumexp

from mushroom.algorithms.agent import Agent
from mushroom.approximators.regressor import Ensemble, Regressor
from mushroom.utils.replay_memory import Buffer

from replay_memory import ReplayMemory


class DQN(Agent):
    """
    Deep Q-Network algorithm.
    "Human-Level Control Through Deep Reinforcement Learning".
    Mnih V. et al.. 2015.

    """
    def __init__(self, approximator, policy, mdp_info, batch_size,
                 initial_replay_size, max_replay_size, n_actions_per_head,
                 target_update_frequency=2500, fit_params=None,
                 approximator_params=None, n_games=1, history_length=1,
                 clip_reward=True, max_no_op_actions=0, no_op_action_value=0,
                 dtype=np.float32, distill=False, entropy_coeff=np.inf):
        self._fit_params = dict() if fit_params is None else fit_params

        self._batch_size = batch_size
        self._n_games = n_games
        self._clip_reward = clip_reward
        self._n_action_per_head = n_actions_per_head
        self._max_actions = max(n_actions_per_head)[0]
        self._target_update_frequency = target_update_frequency
        self._history_length = history_length
        self._max_no_op_actions = max_no_op_actions
        self._no_op_action_value = no_op_action_value
        self._distill = distill
        self._freeze_shared_weights = False
        self._entropy_coeff = entropy_coeff

        self._replay_memory = [
            ReplayMemory(mdp_info[i], initial_replay_size, max_replay_size,
                         history_length, dtype) for i in range(self._n_games)
        ]
        self._buffer = [
            Buffer(history_length, dtype) for _ in range(self._n_games)
        ]

        self._n_updates = 0
        self._episode_steps = [0 for _ in range(self._n_games)]
        self._no_op_actions = [None for _ in range(self._n_games)]

        apprx_params_train = deepcopy(approximator_params)
        apprx_params_target = deepcopy(approximator_params)
        self.approximator = Regressor(approximator, **apprx_params_train)
        self.target_approximator = Regressor(approximator,
                                             **apprx_params_target)
        policy.set_q(self.approximator)

        self.target_approximator.model.set_weights(
            self.approximator.model.get_weights())

        super().__init__(policy, mdp_info[np.argmax(self._n_action_per_head)])

        n_samples = self._batch_size * self._n_games
        self._state_idxs = np.zeros(n_samples, dtype=np.int)
        self._state = np.zeros(
            ((n_samples,
             self._history_length) + self.mdp_info.observation_space.shape),
            dtype=dtype
        )
        self._action = np.zeros((n_samples, 1))
        self._reward = np.zeros(n_samples)
        self._next_state_idxs = np.zeros(n_samples, dtype=np.int)
        self._next_state = np.zeros(
            ((n_samples,
             self._history_length) + self.mdp_info.observation_space.shape),
            dtype=dtype
        )
        self._absorbing = np.zeros(n_samples)

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
                self._state[start:stop] = game_state
                self._action[start:stop] = game_action
                self._reward[start:stop] = game_reward
                self._next_state_idxs[start:stop] = np.ones(self._batch_size) * i
                self._next_state[start:stop] = game_next_state
                self._absorbing[start:stop] = game_absorbing

            if self._clip_reward:
                reward = np.clip(self._reward, -1, 1)
            else:
                reward = self._reward

            q_next = self._next_q()
            q = reward + self.mdp_info.gamma * q_next

            self.approximator.fit(self._state, self._action, q,
                                  idx=self._state_idxs,
                                  get_features=True, **self._fit_params)

            if self._distill:
                self.approximator.fit(self._state, self._action, q,
                                      idx=self._state_idxs,
                                      get_features=True, **self._fit_params)
                self._switch_freezed_weights()

            self._n_updates += 1

            if self._n_updates % self._target_update_frequency == 0:
                self._update_target()

    def _update_target(self):
        """
        Update the target network.

        """
        self.target_approximator.model.set_weights(
            self.approximator.model.get_weights())

    def _switch_freezed_weights(self):
        n_shared = self.approximator.model._network._n_shared
        if self._freeze_shared_weights:
            for i, p in enumerate(self.approximator.model._network.parameters()):
                if i < n_shared:
                    p.requires_grad = False
                else:
                    p.requires_grad = True
        else:
            for i, p in enumerate(self.approximator.model._network.parameters()):
                if i < n_shared:
                    p.requires_grad = True
                else:
                    p.requires_grad = False

        self._freeze_shared_weights = not self._freeze_shared_weights

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
            if self._entropy_coeff == np.inf:
                out_q[start:stop] = np.max(q[start:stop, :n_actions], axis=1)
            elif self._entropy_coeff == 0:
                out_q[start:stop] = np.mean(q[start:stop, :n_actions], axis=1)
            elif self._entropy_coeff == -np.inf:
                out_q[start:stop] = np.min(q[start:stop, :n_actions], axis=1)
            else:
                out_q[start:stop] = logsumexp(
                    self._entropy_coeff * q[start:stop, :n_actions], axis=1
                ) / self._entropy_coeff

        return out_q

    def draw_action(self, state):
        idx = state[0]
        self._buffer[idx].add(state[1])

        if self._episode_steps[idx] < self._no_op_actions[idx]:
            action = np.array([self._no_op_action_value])
            self.policy.update(state)
        else:
            extended_state = self._buffer[idx].get()

            extended_state = [idx, np.array([extended_state])]
            action = super(DQN, self).draw_action(extended_state)

        self._episode_steps[idx] += 1

        return action

    def episode_start(self, idx):
        if self._max_no_op_actions == 0:
            self._no_op_actions[idx] = 0
        else:
            self._no_op_actions[idx] = np.random.randint(
                self._history_length, self._max_no_op_actions + 1)
        self._episode_steps[idx] = 0
