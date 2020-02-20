from copy import deepcopy

import numpy as np

import torch.nn as nn
from mushroom_rl.algorithms import Agent
from mushroom_rl.approximators import Regressor

from replay_memory import ReplayMemory


class ActorLoss(nn.Module):
    def __init__(self, critic):
        super().__init__()

        self._critic = critic

    def forward(self, arg, state, idxs):
        action = arg

        q = self._critic.model.network(state, action, idx=idxs)

        return -q.mean()


class DDPG(Agent):
    def __init__(self, actor_approximator, critic_approximator, policy_class,
                 mdp_info, batch_size, initial_replay_size, max_replay_size,
                 tau, actor_params, critic_params, policy_params,
                 n_actions_per_head, history_length=1, n_input_per_mdp=None,
                 n_games=1, lps_update_frequency=100, lps_samples=1000,
                 dtype=np.uint8):
        self._batch_size = batch_size
        self._n_games = n_games
        if n_input_per_mdp is None:
            self._n_input_per_mdp = [mdp_info.observation_space.shape
                                     for _ in range(self._n_games)]
        else:
            self._n_input_per_mdp = n_input_per_mdp
        self._n_actions_per_head = n_actions_per_head
        self._max_actions = max(n_actions_per_head)[0]
        self._history_length = history_length
        self._tau = tau
        self._lps_update_frequency = lps_update_frequency
        self._lps_samples = lps_samples

        self._replay_memory = [
            ReplayMemory(initial_replay_size,
                         max_replay_size) for _ in range(self._n_games)
        ]

        self._n_updates = 0

        target_critic_params = deepcopy(critic_params)
        self._critic_approximator = Regressor(critic_approximator,
                                              **critic_params)
        self._target_critic_approximator = Regressor(critic_approximator,
                                                     **target_critic_params)

        if 'loss' not in actor_params:
            actor_params['loss'] = ActorLoss(self._critic_approximator)

        target_actor_params = deepcopy(actor_params)
        self._actor_approximator = Regressor(actor_approximator,
                                             n_fit_targets=2, **actor_params)
        self._target_actor_approximator = Regressor(actor_approximator,
                                                    n_fit_targets=2,
                                                    **target_actor_params)

        self._target_actor_approximator.model.set_weights(
            self._actor_approximator.model.get_weights())
        self._target_critic_approximator.model.set_weights(
            self._critic_approximator.model.get_weights())

        policy = policy_class(self._actor_approximator, **policy_params)

        super().__init__(mdp_info, policy)

        n_samples = self._batch_size * self._n_games
        self._state_idxs = np.zeros(n_samples, dtype=np.int)
        self._state = np.zeros(
            ((n_samples,
             self._history_length) + self.mdp_info.observation_space.shape),
            dtype=dtype
        ).squeeze()
        self._action = np.zeros((n_samples, self._max_actions))
        self._reward = np.zeros(n_samples)
        self._next_state_idxs = np.zeros(n_samples, dtype=np.int)
        self._next_state = np.zeros(
            ((n_samples,
             self._history_length) + self.mdp_info.observation_space.shape),
            dtype=dtype
        ).squeeze()
        self._absorbing = np.zeros(n_samples)

        self.norm_lps = np.ones(self._n_games) / self._n_games
        self.all_norm_lps = list()

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
                self._action[start:stop, :self._n_actions_per_head[i][0]] = game_action
                self._reward[start:stop] = game_reward
                self._next_state_idxs[start:stop] = np.ones(self._batch_size) * i
                self._next_state[start:stop, :self._n_input_per_mdp[i][0]] = game_next_state
                self._absorbing[start:stop] = game_absorbing

            q_next = self._next_q()
            q = self._reward + q_next

            self._critic_approximator.fit(self._state, self._action, q,
                                          idx=self._state_idxs)
            self._actor_approximator.fit(self._state, self._state,
                                         self._state_idxs,
                                         idx=self._state_idxs)

            self._n_updates += 1

            self._update_target()

            if self._n_updates % self._lps_update_frequency == 0:
                self._update_lps()

    def get_shared_weights(self):
        cw = self._critic_approximator.model.network.get_shared_weights()
        aw = self._actor_approximator.model.network.get_shared_weights()

        return [cw, aw]

    def set_shared_weights(self, weights):
        self._critic_approximator.model.network.set_shared_weights(weights[0])
        self._actor_approximator.model.network.set_shared_weights(weights[1])

    def freeze_shared_weights(self):
        self._critic_approximator.model.network.freeze_shared_weights()
        self._actor_approximator.model.network.freeze_shared_weights()

    def unfreeze_shared_weights(self):
        self._critic_approximator.model.network.unfreeze_shared_weights()
        self._actor_approximator.model.network.unfreeze_shared_weights()

    def _update_target(self):
        """
        Update the target networks.

        """
        critic_weights = self._tau * self._critic_approximator.model.get_weights()
        critic_weights += (1 - self._tau) * self._target_critic_approximator.get_weights()
        self._target_critic_approximator.set_weights(critic_weights)

        actor_weights = self._tau * self._actor_approximator.model.get_weights()
        actor_weights += (1 - self._tau) * self._target_actor_approximator.get_weights()
        self._target_actor_approximator.set_weights(actor_weights)

    def _update_lps(self):
        n_samples = self._lps_samples * self._n_games
        state = np.zeros(
            ((n_samples,
             self._history_length) + self.mdp_info.observation_space.shape)
        ).squeeze()
        action = np.zeros((n_samples, self._max_actions))
        reward = np.zeros(n_samples)
        next_state = np.zeros(
            ((n_samples,
             self._history_length) + self.mdp_info.observation_space.shape)
        ).squeeze()
        absorbing = np.zeros(n_samples)
        idxs = np.zeros(n_samples, dtype=np.int)
        for i in range(self._n_games):
            start = self._lps_samples * i
            stop = start + self._lps_samples
            state[start:stop, :self._n_input_per_mdp[i][0]],\
                action[start:stop, :self._n_actions_per_head[i][0]], reward[start:stop],\
                next_state[start:stop, :self._n_input_per_mdp[i][0]],\
                absorbing[start:stop], _ = self._replay_memory[i].get(self._lps_samples)
            idxs[start:stop] = np.ones(self._lps_samples, dtype=np.int) * i

        next_action = self._target_actor_approximator(next_state,
                                                      idx=idxs)
        next_q = self._target_critic_approximator.predict(next_state,
                                                          next_action,
                                                          idx=idxs)
        q = self._target_critic_approximator.predict(state, action, idx=idxs)

        norm_lps = np.zeros((self._n_games, self._lps_samples))
        for i in range(self._n_games):
            start = self._lps_samples * i
            stop = start + self._lps_samples
            if np.any(absorbing[start:stop]):
                next_q[start:stop] *= 1 - absorbing[start:stop]

            td_errors = self.mdp_info.gamma[i] * next_q[start:stop]
            td_errors += reward[start:stop] - q[start:stop]

            norm_lps[i] = np.abs(td_errors)

        self.norm_lps = norm_lps.mean(1)

        self.all_norm_lps.append(self.norm_lps.copy())

    def _next_q(self):
        a = self._target_actor_approximator(self._next_state,
                                            idx=self._next_state_idxs)
        q = self._target_critic_approximator(self._next_state, a,
                                             idx=self._next_state_idxs).ravel()

        out_q = np.zeros(self._batch_size * self._n_games)
        for i in range(self._n_games):
            start = self._batch_size * i
            stop = start + self._batch_size

            out_q[start:stop] = q[start:stop] * self.mdp_info.gamma[i]
            if np.any(self._absorbing[start:stop]):
                out_q[start:stop] = out_q[start:stop] * (
                    1 - self._absorbing[start:stop]
                )

        return out_q
