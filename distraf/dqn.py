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
                 distilled_params, initial_replay_size, max_replay_size,
                 target_update_frequency=2500, fit_params=None,
                 approximator_params=None, n_games=1, history_length=1,
                 clip_reward=True, max_no_op_actions=0, no_op_action_value=0,
                 dtype=np.float32, entropy_coeff=np.inf):
        self._fit_params = dict() if fit_params is None else fit_params

        self._batch_size = batch_size
        self._n_games = n_games
        self._clip_reward = clip_reward
        self._target_update_frequency = target_update_frequency
        self._history_length = history_length
        self._max_no_op_actions = max_no_op_actions
        self._no_op_action_value = no_op_action_value
        self._entropy_coeff = entropy_coeff

        self._replay_memory = [
            ReplayMemory(mdp_info, initial_replay_size, max_replay_size,
                         history_length, dtype) for _ in range(self._n_games)
        ]
        self._buffer = [
            Buffer(history_length, dtype) for _ in range(self._n_games)
        ]

        self._n_updates = 0
        self._episode_steps = 0
        self._no_op_actions = None

        self._distilled = Regressor(approximator, **distilled_params)
        self.approximator = list()
        self.target_approximator = list()
        for i in range(self._n_games):
            apprx_params_train = deepcopy(approximator_params[i])
            apprx_params_target = deepcopy(approximator_params[i])
            self.approximator.append(
                Regressor(approximator, **apprx_params_train)
            )
            self.target_approximator.append(
                Regressor(approximator, **apprx_params_target)
            )
        policy.set_q(self.approximator)

        for i in range(self._n_games):
            self.target_approximator[i].model.set_weights(
                self.approximator[i].model.get_weights())

        super().__init__(policy, mdp_info)

        self._all_states = np.zeros(
            ((self._batch_size * self._n_games,
             self._history_length) + self.mdp_info.observation_space.shape),
            dtype=dtype
        )
        self._features = np.zeros((self._batch_size * self._n_games,
                                   self._distilled.model._network.n_features))

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
                state, action, reward, next_state, absorbing, _ =\
                    self._replay_memory[i].get(self._batch_size)

                start = self._batch_size * i
                stop = start + self._batch_size
                self._all_states[start:stop] = state

                if self._clip_reward:
                    reward = np.clip(reward, -1, 1)

                q_next = self._next_q(next_state, absorbing, idx=i)
                q = reward + self.mdp_info.gamma * q_next

                features = self._distilled.predict(state)
                self.approximator[i].model.fit(state, action, q, features,
                                               get_type=2, **self._fit_params)

                self._n_updates += 1

                if self._n_updates % self._target_update_frequency == 0:
                    self._update_target(i)

                self._features[start:stop] = self.target_approximator[
                    i].predict(state, get_type=1)

            self._distilled.fit(self._all_states, self._features)

    def _update_target(self, idx):
        """
        Update the target network.

        """
        self.target_approximator[idx].model.set_weights(
            self.approximator[idx].model.get_weights())

    def _next_q(self, next_state, absorbing, idx):
        q = self.target_approximator[idx].predict(next_state)

        if np.any(absorbing):
            q *= 1 - absorbing.reshape(-1, 1)

        if self._entropy_coeff == np.inf:
            out_q = np.max(q, axis=1)
        elif self._entropy_coeff == 0:
            out_q = np.mean(q, axis=1)
        elif self._entropy_coeff == -np.inf:
            out_q = np.min(q, axis=1)
        else:
            out_q = logsumexp(self._entropy_coeff * q,
                              axis=1) / self._entropy_coeff

        return out_q

    def draw_action(self, state):
        self._buffer[np.asscalar(state[0])].add(state[1])

        if self._episode_steps < self._no_op_actions:
            action = np.array([self._no_op_action_value])
            self.policy.update(state)
        else:
            extended_state = self._buffer[np.asscalar(state[0])].get()

            extended_state = np.array([state[0], extended_state])
            action = super(DQN, self).draw_action(extended_state)

        self._episode_steps += 1

        return action

    def episode_start(self):
        if self._max_no_op_actions == 0:
            self._no_op_actions = 0
        else:
            self._no_op_actions = np.random.randint(
                self._history_length, self._max_no_op_actions + 1)
        self._episode_steps = 0
