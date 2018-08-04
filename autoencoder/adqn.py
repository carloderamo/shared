from copy import deepcopy

import numpy as np

from mushroom.algorithms.agent import Agent
from mushroom.approximators.regressor import Ensemble, Regressor

from replay_memory import ReplayMemory

import scipy


class DQN(Agent):
    """
    Deep Q-Network algorithm.
    "Human-Level Control Through Deep Reinforcement Learning".
    Mnih V. et al.. 2015.

    """
    def __init__(self, approximator, autoencoder, policy, mdp_info, batch_size,
                 initial_replay_size, max_replay_size, n_actions_per_head,
                 target_update_frequency=2500, fit_params=None,
                 approximator_params=None, autoencoder_params=None,
                 n_games=1, history_length=1, clip_reward=True,
                 dtype=np.float32):
        self._fit_params = dict() if fit_params is None else fit_params

        self._batch_size = batch_size
        self._n_games = n_games
        self._clip_reward = clip_reward
        self._target_update_frequency = target_update_frequency
        self._history_length = history_length
        self._n_action_per_head = n_actions_per_head

        self._replay_memory = [
            ReplayMemory(initial_replay_size,
                         max_replay_size) for _ in range(self._n_games)]

        self._n_updates = 0

        self.autoencoder = Regressor(autoencoder, **autoencoder_params)
        self.approximator = list()
        self.target_approximator = list()

        for i in range(self._n_games):
            apprx_params_train = deepcopy(approximator_params[i])
            apprx_params_target = deepcopy(approximator_params[i])
            apprx = Regressor(approximator, **apprx_params_train)
            target_approximator = Regressor(approximator, **apprx_params_target)

            self.approximator.append(apprx)
            self.target_approximator.append(target_approximator)

        policy.set_q(self.approximator)

        for i in range(self._n_games):
            self.target_approximator[i].model.set_weights(
                self.approximator[i].model.get_weights())

        super().__init__(policy, mdp_info[np.argmax(self._n_action_per_head)])

        self._state = np.zeros((self._batch_size * self._n_games,
                                self._history_length)
                               + self.mdp_info.observation_space.shape,
                               dtype=dtype)
        self.tmp_var = 0

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
            # Fit autoencoder
            for i in range(len(self._replay_memory)):
                game_state, _, _, _, _, _ = self._replay_memory[i].get(
                        self._batch_size)

                start = self._batch_size * i
                stop = self._batch_size * i + self._batch_size

                self._state[start:stop] = game_state

            self.autoencoder.fit(self._state, self._state / 255.,
                                 get_features=True)

            # Fit DQN
            for i in range(len(self._replay_memory)):
                state, action, reward, next_state, absorbing, _ =  \
                     self._replay_memory[i].get(self._batch_size)

                state = self.autoencoder(state, encode=True)
                next_state = self.autoencoder(next_state, encode=True)

                if self._clip_reward:
                    reward = np.clip(reward, -1, 1)

                q_next = self._next_q(next_state, i, absorbing)
                q = reward + self.mdp_info.gamma * q_next

                self.approximator[i].fit(state, action, q, **self._fit_params)

            self._n_updates += 1

            if self._n_updates % self._target_update_frequency == 0:
                self._update_target()
                image = self._replay_memory[i].get(1)[0]
                print('saving image...')
                print(image.shape)

                encoded_image = self.autoencoder(image, encode=False)
                scipy.misc.imsave('original_' + str(self.tmp_var) + '.png',
                                  image[0, 0, :, :])
                scipy.misc.imsave('encoded_' + str(self.tmp_var) + '.png',
                                  encoded_image[0, 0, :, :])
                self.tmp_var += 1

    def _update_target(self):
        """
        Update the target network.

        """
        for i in range(len(self.target_approximator)):
            self.target_approximator[i].model.set_weights(
                self.approximator[i].model.get_weights())

    def _next_q(self, next_state, i, absorbing):
        q = self.target_approximator[i].predict(next_state)
        if np.any(absorbing):
            q *= 1 - absorbing.reshape(-1, 1)

        return np.max(q, axis=1)

    def draw_action(self, state):
        encoded_state = self.autoencoder(np.array(state[1]), encode=True)
        extended_state = [state[0], encoded_state]
        action = super(DQN, self).draw_action(extended_state)

        return action
