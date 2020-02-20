from copy import copy

import numpy as np
from tqdm import tqdm


class Core(object):
    def __init__(self, agent, mdp, epsilon, callbacks=None):
        self.agent = agent
        self.mdp = mdp
        self._epsilon = epsilon
        self._n_mdp = len(self.mdp)
        self.callbacks = callbacks if callbacks is not None else list()

        self._state = [None for _ in range(self._n_mdp)]

        self._total_steps_counter = 0
        self._current_steps_counter = 0
        self._episode_steps = [None for _ in range(self._n_mdp)]
        self._n_steps_per_fit = None

        self.task_sampling = False
        self.n_samples_per_task = list()

    def learn(self, n_steps=None, n_steps_per_fit=None, render=False,
              quiet=False):
        self._n_steps_per_fit = n_steps_per_fit

        fit_condition = \
            lambda: self._current_steps_counter >= self._n_steps_per_fit

        self._run(n_steps, fit_condition, render, quiet)

    def evaluate(self, n_steps=None, render=False,
                 quiet=False):
        fit_condition = lambda: False

        return self._run(n_steps, fit_condition, render, quiet)

    def _run(self, n_steps, fit_condition, render, quiet):
        move_condition = lambda: self._total_steps_counter < n_steps

        steps_progress_bar = tqdm(total=n_steps,
                                  dynamic_ncols=True, disable=quiet,
                                  leave=False)

        return self._run_impl(move_condition, fit_condition, steps_progress_bar,
                              render)

    def _run_impl(self, move_condition, fit_condition, steps_progress_bar,
                  render):
        self._total_steps_counter = 0
        self._current_steps_counter = 0

        dataset = list()
        last = [True] * self._n_mdp
        while move_condition():
            if self.task_sampling is False:
                mdps = np.arange(self._n_mdp)
            else:
                eps = self._epsilon()
                p = eps / self._n_mdp + (1 - eps) * (self.agent.norm_lps) / self.agent.norm_lps.sum()
                print(p)
                mdps = np.random.choice(self._n_mdp, size=self._n_mdp, p=p)

                sampled_tasks = np.zeros(self._n_mdp) if not self.n_samples_per_task else copy(self.n_samples_per_task[-1])
                tasks, counts = np.unique(mdps, return_counts=True)
                for i, t in enumerate(tasks):
                    sampled_tasks[t] += counts[i]
                self.n_samples_per_task.append(sampled_tasks)

            for i in mdps:
                if last[i]:
                    self.reset(i)

                sample = self._step(i, render)
                dataset.append(sample)

                last[i] = sample[-1]

            self._total_steps_counter += 1
            self._current_steps_counter += 1
            steps_progress_bar.update(1)

            if fit_condition():
                self.agent.fit(dataset)
                self._current_steps_counter = 0

                for c in self.callbacks:
                    callback_pars = dict(dataset=dataset)
                    c(**callback_pars)

                dataset = list()

        self.agent.stop()
        for i in range(self._n_mdp):
            self.mdp[i].stop()

        return dataset

    def _step(self, i, render):
        action = self.agent.draw_action([i, self._state[i]])
        next_state, reward, absorbing, _ = self.mdp[i].step(action)

        self._episode_steps[i] += 1

        if render:
            self.mdp[i].render()

        last = not(
            self._episode_steps[i] < self.mdp[i].info.horizon and not absorbing)

        state = self._state[i]
        self._state[i] = next_state.copy()

        return [i, state], action, reward, [i, next_state], absorbing, last

    def reset(self, i):
        self._state[i] = self.mdp[i].reset().copy()
        self.agent.episode_start()
        self.agent.next_action = None
        self._episode_steps[i] = 0
