import cv2
cv2.ocl.setUseOpenCL(False)
import numpy as np

from mushroom.environments import Atari, Environment, MDPInfo


class AtariMultiple:
    def __init__(self, name, width=84, height=84, ends_at_life=False,
                 max_pooling=True):
        # MPD creation
        self.envs = list()
        for n in name:
            self.envs.append(Atari(n, width, height, ends_at_life, max_pooling))

        self._env_idxs = np.arange(len(self.envs))
        self._reset_envs_list()

    def reset(self, state=None):
        self._state = self._env_idxs[self._current_idx].reset(state)

        return self._state

    def step(self, action):
        self._state, reward, absorbing, info = self._env_idxs[
            self._current_idx].step(action)

        if absorbing:
            self._current_idx += 1
            if self._current_idx == len(self.envs):
                self._reset_envs_list()

        return self._state, reward, absorbing, info

    def render(self, mode='human'):
        self.envs[self._current_idx].render(mode=mode)

    def stop(self):
        self.envs[self._current_idx].stop()

    def set_episode_end(self, ends_at_life):
        self.envs[self._current_idx].set_episode_end(ends_at_life)

    def _reset_envs_list(self):
        np.random.shuffle(self._env_idxs)
        self._current_idx = 0
