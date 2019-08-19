import datetime
import pathlib
import sys

import numpy as np
from joblib import Parallel, delayed
import torch.optim as optim

sys.path.append('..')

from mushroom.approximators.parametric.torch_approximator import TorchApproximator
from mushroom.environments import *
from mushroom.utils.callbacks import CollectDataset
from mushroom.utils.dataset import compute_J
from mushroom.utils.parameters import Parameter

from core import Core
from fqi import FQI
from losses import LossFunction
from networks import LQRNetwork
from policy import EpsGreedyMultipleDiscretized
from utils import computeOptimalK, computeQFunction

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
    mdp = [LQR.generate(dimensions=1, max_action=2., random_init=True)
           ]
    n_games = len(mdp)
    discrete_actions = np.linspace(mdp[0].info.action_space.low[0],
                                   mdp[0].info.action_space.high[0],
                                   100)
    input_shape = [(m.info.observation_space.shape[0] +
                    m.info.action_space.shape[0],) for m in mdp]

    # Policy
    epsilon = Parameter(value=1.)
    pi = EpsGreedyMultipleDiscretized(parameter=epsilon,
                                      n_actions_per_head=discrete_actions)

    # Approximator
    optimizer = {'class': optim.Adam, 'params': dict()}
    loss = LossFunction(n_games)

    approximator_params = dict(
        network=LQRNetwork,
        input_shape=input_shape,
        optimizer=optimizer,
        loss=loss,
        features='relu',
        n_features=15,
        use_cuda=True,
        quiet=False
    )

    approximator = TorchApproximator

    # Agent
    algorithm_params = dict(n_iterations=5, discrete_actions=discrete_actions,
                            fit_params=dict(patience=100, epsilon=1e-5))
    agent = FQI(approximator, pi, mdp[0].info,
                approximator_params=approximator_params, **algorithm_params)

    # Algorithm
    collect_dataset = CollectDataset()
    core = Core(agent, mdp, callbacks=[collect_dataset])

    # Train
    pi.set_parameter(epsilon)

    core.learn(n_steps=500, n_steps_per_fit=500)

    temp_dataset = collect_dataset.get()
    dataset = list()
    for i in range(len(mdp)):
        dataset += temp_dataset[i::len(mdp)]

    K = computeOptimalK(mdp[0].A, mdp[0].B, mdp[0].Q, mdp[0].R, mdp[0].info.gamma)
    qs = list()
    for d in dataset:
        qs.append(computeQFunction(
            d[0][1], d[1], K, mdp[0].A, mdp[0].B, mdp[0].Q, mdp[0].R,
            np.array([[0]]), mdp[0].info.gamma, n_random_xn=100)
        )
    qs = np.array(qs)
    qs_hat = np.array(agent._qs)

    avi_diff = list()
    for i in range(len(qs_hat)):
        avi_diff.append(np.linalg.norm(qs_hat[i] - qs, ord=1) / len(qs))
    print(avi_diff)

    # Test
    test_epsilon = Parameter(0.)
    pi.set_parameter(test_epsilon)

    dataset = core.evaluate(n_steps=1000, render=False)

    scores = [list() for _ in range(len(mdp))]
    for i in range(len(mdp)):
        d = dataset[i::len(mdp)]
        current_score = get_stats(d, mdp[0].info.gamma)
        scores[i].append(current_score)

    return scores, avi_diff


if __name__ == '__main__':
    folder_name = './logs/gym_' + datetime.datetime.now().strftime(
        '%Y-%m-%d_%H-%M-%S') + '/'
    pathlib.Path(folder_name).mkdir(parents=True)

    n_exp = 1
    out = Parallel(n_jobs=-1)(delayed(experiment)() for i in range(n_exp))

    scores = np.array([o[0] for o in out])
    avi_diff = np.array([o[1] for o in out])

    np.save(folder_name + 'scores.npy', scores)
    np.save(folder_name + 'avi_diff.npy', avi_diff)
