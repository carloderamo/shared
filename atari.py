import cv2
cv2.ocl.setUseOpenCL(False)
import gym
import numpy as np

from mushroom.environments import Atari, Environment, MDPInfo
from mushroom.utils.spaces import Box, Discrete


class AtariMultiple(Environment):
    def __init__(self, name, width=84, height=84, ends_at_life=False,
                 max_pooling=True):
        # MPD creation
        self.envs = list()
        for n in name:
            self.envs.append(Atari(n, width, height, ends_at_life, max_pooling))

        self._env_idxs = np.arange(len(self.envs))
        self._reset_envs_list()
        self._freezed_env = False

        # MDP properties
        action_space = Discrete(len(gym.envs.atari.atari_env.ACTION_MEANING))
        observation_space = Box(low=0., high=255., shape=(width, height))
        horizon = np.inf  # the gym time limit is used.
        gamma = .99
        mdp_info = MDPInfo(observation_space, action_space, gamma, horizon)

        super().__init__(mdp_info)

    def reset(self, state=None):
        self._state = self.envs[self._current_idx].reset(state)
        self._state = self._augment_state(self._state)

        return self._state

    def step(self, action):
        self._state, reward, absorbing, info = self.envs[
            self._current_idx].step(action)
        self._state = self._augment_state(self._state)

        if absorbing and not self._freezed_env:
            self._current_idx += 1
            if self._current_idx == len(self.envs):
                self._reset_envs_list()

        return self._state, reward, absorbing, info

    def render(self, mode='human'):
        self.envs[self._current_idx].render(mode=mode)

    def stop(self):
        self.envs[self._current_idx].stop()

    def set_env(self, idx):
        self._current_idx = idx

    def set_episode_end(self, ends_at_life):
        self.envs[self._current_idx].set_episode_end(ends_at_life)

    def freeze_env(self, freeze):
        self._freezed_env = freeze

    @property
    def n_games(self):
        return len(self.envs)

    def _reset_envs_list(self):
        np.random.shuffle(self._env_idxs)
        self._current_idx = 0

    def _augment_state(self, state):
        return np.array([np.array([self._current_idx]), state])
