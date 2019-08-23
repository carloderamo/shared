import datetime
import pathlib
import pickle
import sys

import numpy as np
from joblib import Parallel, delayed
import torch.optim as optim

sys.path.append('..')

from mushroom.approximators.parametric.torch_approximator import TorchApproximator
from mushroom.environments import CarOnHill
from mushroom.utils.callbacks import CollectDataset
from mushroom.utils.dataset import compute_J
from mushroom.utils.parameters import Parameter

from core import Core
from fqi import FQI
from losses import LossFunction
from networks import Network
from policy import EpsGreedyMultiple
from solver import solve_car_on_hill

"""
This script aims to replicate the experiments on the Car on Hill MDP as
presented in:
"Tree-Based Batch Mode Reinforcement Learning", Ernst D. et al.. 2005. 

"""


def get_stats(dataset, gamma):
    J = np.mean(compute_J(dataset, gamma))

    return J


def experiment():
    np.random.seed()

    # MDP
    mdp = [CarOnHill()]
    n_games = len(mdp)
    input_shape = [(m.info.observation_space.shape[0],) for m in mdp]
    n_actions_per_head = [(m.info.action_space.n,) for m in mdp]

    test_states_0 = np.linspace(mdp[0].info.observation_space.low[0],
                                mdp[0].info.observation_space.high[0], 10)
    test_states_1 = np.linspace(mdp[0].info.observation_space.low[1],
                                mdp[0].info.observation_space.high[1], 10)
    test_states = list()
    for s0 in test_states_0:
        for s1 in test_states_1:
            test_states += [s0, s1]
    test_states = np.array([test_states]).repeat(2, 0).reshape(-1, 2)
    test_actions = np.array(
        [np.zeros(len(test_states) // 2),
         np.ones(len(test_states) // 2)]).reshape(-1, 1).astype(np.int)

    load_test_q = True
    # Test Q
    if not load_test_q:
        test_q = list()
        for m in mdp:
            test_q.append(solve_car_on_hill(m, test_states, test_actions,
                                            m.info.gamma))
        np.save('test_q.npy', test_q)
    else:
        test_q = np.load('test_q.npy')

    test_states = np.array([test_states]).repeat(len(mdp), 0).reshape(-1, 2)
    test_actions = np.array([test_actions]).repeat(len(mdp), 0).reshape(-1, 1)
    test_idxs = np.ones(len(test_states), dtype=np.int) * np.arange(len(mdp)).repeat(
        len(test_states) // len(mdp), 0)

    # Policy
    epsilon = Parameter(value=1.)
    pi = EpsGreedyMultiple(parameter=epsilon,
                           n_actions_per_head=n_actions_per_head)

    # Approximator
    optimizer = {'class': optim.Adam, 'params': dict()}
    loss = LossFunction(n_games)

    approximator_params = dict(
        network=Network,
        input_shape=input_shape,
        output_shape=n_actions_per_head,
        optimizer=optimizer,
        loss=loss,
        features='relu',
        n_features=10,
        use_cuda=True,
        quiet=False,
        reinitialize=True
    )

    approximator = TorchApproximator

    # Agent
    algorithm_params = dict(n_iterations=20,
                            n_actions_per_head=n_actions_per_head,
                            test_states=test_states, test_actions=test_actions,
                            test_idxs=test_idxs,
                            fit_params=dict(patience=100, epsilon=1e-5))
    agent = FQI(approximator, pi, mdp[0].info,
                approximator_params=approximator_params, **algorithm_params)

    # Algorithm
    core = Core(agent, mdp)

    # Train
    core.learn(n_steps=60000, n_steps_per_fit=60000)

    qs_hat = np.array(agent._qs)
    avi_diff = list()
    for i in range(len(qs_hat)):
        avi_diff.append(np.linalg.norm(qs_hat[i] - test_q, ord=1) / len(test_q))
    print(avi_diff)

    return avi_diff


if __name__ == '__main__':
    folder_name = './logs/coh_' + datetime.datetime.now().strftime(
        '%Y-%m-%d_%H-%M-%S') + '/'
    pathlib.Path(folder_name).mkdir(parents=True)

    n_exp = 1
    out = Parallel(n_jobs=-1)(delayed(experiment)() for i in range(n_exp))

    avi_diff = np.array([out])

    np.save(folder_name + 'avi_diff.npy', avi_diff)
