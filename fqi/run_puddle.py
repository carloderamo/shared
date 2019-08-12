import datetime
import pathlib
import sys

import numpy as np
from joblib import Parallel, delayed
import torch.optim as optim

sys.path.append('..')

from mushroom.approximators.parametric.torch_approximator import TorchApproximator
from mushroom.environments import *
from mushroom.utils.dataset import compute_J
from mushroom.utils.parameters import Parameter

from core import Core
from fqi import FQI
from losses import LossFunction
from networks import PuddleNetwork
from policy import EpsGreedyMultiple

"""
This script aims to replicate the experiments on the Car on Hill MDP as
presented in:
"Tree-Based Batch Mode Reinforcement Learning", Ernst D. et al.. 2005. 

"""


def experiment():
    np.random.seed()

    # MDP
    mdp = [PuddleWorld(thrust=.2)]
    n_games = len(mdp)
    input_shape = [m.info.observation_space.shape for m in mdp]
    n_actions = mdp[0].info.action_space.n
    n_actions_per_head = [(m.info.action_space.n,) for m in mdp]

    # Policy
    epsilon = Parameter(value=1.)
    pi = EpsGreedyMultiple(parameter=epsilon,
                           n_actions_per_head=n_actions_per_head)

    # Approximator
    optimizer = {'class': optim.Adam, 'params': dict()}
    loss = LossFunction(n_games)

    approximator_params = dict(
        network=PuddleNetwork,
        input_shape=input_shape,
        output_shape=(max(n_actions_per_head)[0],),
        n_actions=n_actions,
        n_actions_per_head=n_actions_per_head,
        optimizer=optimizer,
        loss=loss,
        features='relu',
        use_cuda=False,
        quiet=False
    )

    approximator = TorchApproximator

    # Agent
    algorithm_params = dict(n_iterations=20,
                            fit_params=dict(patience=10, epsilon=1e-3))
    agent = FQI(approximator, pi, mdp[0].info,
                approximator_params=approximator_params, **algorithm_params)

    # Algorithm
    core = Core(agent, mdp)

    # Train
    pi.set_parameter(epsilon)
    core.learn(n_steps=1000, n_steps_per_fit=1000)

    # Test
    test_epsilon = Parameter(0.)
    pi.set_parameter(test_epsilon)

    dataset = core.evaluate(n_steps=1000, render=True)

    print(np.mean(compute_J(dataset, mdp[0].info.gamma)))

    return (np.mean(compute_J(dataset, mdp[0].info.gamma)),)


if __name__ == '__main__':
    folder_name = './logs/gym_' + datetime.datetime.now().strftime(
        '%Y-%m-%d_%H-%M-%S') + '/'
    pathlib.Path(folder_name).mkdir(parents=True)

    n_exp = 1
    out = Parallel(n_jobs=-1)(delayed(experiment)() for i in range(n_exp))

    scores = np.array([o[0] for o in out])
    # loss = np.array([o[1] for o in out])

    np.save(folder_name + 'scores.npy', scores)
    # np.save(folder_name + 'loss.npy', loss)
