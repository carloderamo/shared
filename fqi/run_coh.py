import pathlib
import pickle
import sys

import numpy as np
from joblib import Parallel, delayed
import torch.optim as optim
from tqdm import trange

sys.path.append('..')

from mushroom.approximators.parametric.torch_approximator import TorchApproximator
from mushroom.utils.dataset import compute_J, parse_dataset
from mushroom.utils.parameters import Parameter

from car_on_hill import CarOnHill
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


def experiment(mdp, test_states, test_actions, test_q, names):
    np.random.seed()

    n_games = len(mdp)
    input_shape = [(m.info.observation_space.shape[0],) for m in mdp]
    n_actions_per_head = [(m.info.action_space.n,) for m in mdp]

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
        features='sigmoid',
        n_features=30,
        use_cuda=True,
        quiet=False
    )

    approximator = TorchApproximator

    dataset = list()
    len_datasets = list()
    for i in range(len(mdp)):
        d = pickle.load(open('dataset_%s.pkl' % names[i], 'rb'))
        len_datasets.append(len(d))
        dataset += d

    # Agent
    algorithm_params = dict(n_iterations=1,
                            n_actions_per_head=n_actions_per_head,
                            fit_params=dict(patience=100, epsilon=1e-6))
    agent = FQI(approximator, pi, mdp[0].info,
                approximator_params=approximator_params, **algorithm_params)

    qs = list()
    scores = list()

    idxs = list()
    for i, l in enumerate(len_datasets):
        idxs += (np.ones(l, dtype=np.int) * i).tolist()
    idxs = np.array(idxs)

    state, action, reward, next_state, absorbing, _ = parse_dataset(dataset)
    for _ in trange(50, dynamic_ncols=True, disable=False, leave=False):
        agent._fit(state, action, reward, next_state, absorbing, idxs)
        # Algorithm
        core = Core(agent, mdp)
        test_epsilon = Parameter(0.)
        pi.set_parameter(test_epsilon)
        dataset = core.evaluate(n_steps=100)

        qs.append(agent.approximator.predict(test_states, test_actions,
                                             idx=test_idxs))
        scores.append(np.mean(compute_J(dataset, mdp[0].info.gamma)))

    qs_hat = np.array(qs)
    avi_diff = list()
    for i in range(len(qs_hat)):
        avi_diff.append(np.linalg.norm(qs_hat[i] - test_q, ord=1) / len(test_q))

    print(avi_diff, scores)

    return avi_diff, scores


if __name__ == '__main__':
    n_exp = 100

    use_mdp = np.array([0])  # , 4, 8, 13, 1, 2, 3, 5, 6, 7, 9, 10, 11, 12, 14, 15])

    load_test_q = True

    # MDP
    all_mdps = [
        CarOnHill(1, 9.81, 4), CarOnHill(1, 9.81, 4.125), CarOnHill(1, 9.81, 4.25), CarOnHill(1, 9.81, 4.375),
        CarOnHill(.8, 9.81, 4), CarOnHill(.85, 9.81, 4), CarOnHill(.9, 9.81, 4), CarOnHill(.95, 9.81, 4),
        CarOnHill(1, 9.81, 4.5), CarOnHill(1.05, 9.81, 4.5), CarOnHill(1.1, 9.81, 4.5), CarOnHill(1.15, 9.81, 4.5),
        CarOnHill(1.2, 9.81, 4.375), CarOnHill(1.2, 9.81, 4.5), CarOnHill(1.2, 9.81, 4.625), CarOnHill(1.2, 9.81, 4.75)
    ]

    mdp = list()
    for i in use_mdp:
        mdp.append(all_mdps[i])

    names = ['%1.3f_%1.3f' % (m._m, m._discrete_actions[-1]) for m in mdp]

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

    # Test Q
    test_q = list()
    if not load_test_q:
        for i, j in enumerate(use_mdp):
            current_test_q = solve_car_on_hill(all_mdps[j], test_states,
                                               test_actions,
                                               all_mdps[j].info.gamma)
            np.save('test_q_%s.npy' % names[i], current_test_q)

            test_q += current_test_q
    else:
        for i in range(len(mdp)):
            test_q += np.load('test_q_%s.npy' % names[i]).tolist()

    test_q = np.array(test_q)

    folder_name = './logs/%s/' % ''.join(names)
    pathlib.Path(folder_name).mkdir(parents=True, exist_ok=True)

    out = Parallel(n_jobs=8)(delayed(experiment)(
        mdp, test_states, test_actions, test_q, names) for i in range(n_exp))

    avi_diff = np.array([o[0] for o in out])
    scores = np.array([o[1] for o in out])

    np.save(folder_name + 'avi_diff.npy', avi_diff)
    np.save(folder_name + 'scores.npy', scores)
