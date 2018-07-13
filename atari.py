import cv2
cv2.ocl.setUseOpenCL(False)
import numpy as np

from mushroom.environments import Atari, Environment, MDPInfo
from mushroom.utils.spaces import Box, Discrete


class AtariMultiple(Environment):
    def __init__(self, name, width=84, height=84, ends_at_life=False,
                 max_pooling=True, n_steps_per_game=32):
        # MPD creation
        self.envs = list()
        for n in name:
            self.envs.append(Atari(n, width, height, ends_at_life, max_pooling))

        max_actions = np.array([e.info.action_space.n for e in self.envs]).max()

        self._current_idx = 0
        self._current_step = 0
        self._freezed_env = False
        self._learn_idx = None
        self._n_steps_per_game = n_steps_per_game
        self._state = [None] * len(self.envs)

        # MDP properties
        action_space = Discrete(max_actions)
        observation_space = Box(low=0., high=255., shape=(width, height))
        horizon = np.inf  # the gym time limit is used.
        gamma = .99
        mdp_info = MDPInfo(observation_space, action_space, gamma, horizon)

        super().__init__(mdp_info)

    def reset(self, state=None):
        state = self.envs[self._current_idx].reset(state)
        self._state[self._current_idx] = self._augment_state(state)

        return self._state[self._current_idx]

    def step(self, action):
        if not self._freezed_env:
            self._current_step += 1
            if self._current_step == self._n_steps_per_game:
                self._current_idx += 1
                if self._current_idx == len(self.envs):
                    self._current_idx = 0
                self._current_step = 0

                return self.reset(), 0, 0, {}
        state, reward, absorbing, info = self.envs[
            self._current_idx].step(action)
        self._state[self._current_idx] = self._augment_state(state)

        return self._state[self._current_idx], reward, absorbing, info

    def render(self, mode='human'):
        self.envs[self._current_idx].render(mode=mode)

    def stop(self):
        self.envs[self._current_idx].stop()

    def set_env(self, idx=None):
        if idx is None:
            self._current_idx = self._learn_idx
            self._learn_idx = None
        else:
            if self._learn_idx is None:
                self._learn_idx = self._current_idx
            self._current_idx = idx

    def set_episode_end(self, ends_at_life):
        self.envs[self._current_idx].set_episode_end(ends_at_life)

    def freeze_env(self, freeze):
        self._freezed_env = freeze

    def _augment_state(self, state):
        return np.array([np.array([self._current_idx]), state])
