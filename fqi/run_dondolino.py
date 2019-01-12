import argparse
import datetime
import pathlib
import sys
import tqdm

from joblib import delayed, Parallel
from tqdm import trange, tqdm
import numpy as np
import torch.optim as optim

import pickle

sys.path.append('..')

from mushroom.approximators.parametric import PyTorchApproximator
from mushroom.utils.dataset import compute_J
from mushroom.utils.parameters import Parameter
from mushroom.environments import *

from core import Core
from fqi import FQI
from policy import EpsGreedyMultiple
from networks import GymNetwork
from losses import *

from pendulum import InvertedPendulumDiscreteV2


def get_stats(dataset, gamma, idx, mass):
    J = np.mean(compute_J(dataset, gamma[idx]))
    tqdm.write('m - %f: J: %f' % (mass[idx], J))

    return J


def experiment(args, idx):
    np.random.seed()

    # MDP
    mdp = list()
    gamma_eval = list()

    for i, m in enumerate(args.pendulum_mass):
        mdp.append(InvertedPendulumDiscreteV2(m=m))
        gamma_eval.append(mdp[-1].info.gamma)

    n_input_per_mdp = [m.info.observation_space.shape for m in mdp]
    n_actions_per_head = [(m.info.action_space.n,) for m in mdp]

    max_obs_dim = 0
    max_act_n = 0
    for i in range(len(args.pendulum_mass)):
        n = mdp[i].info.observation_space.shape[0]
        m = mdp[i].info.action_space.n
        if n > max_obs_dim:
            max_obs_dim = n
            max_obs_idx = i
        if m > max_act_n:
            max_act_n = m
            max_act_idx = i
    gammas = [m.info.gamma for m in mdp]
    horizons = [m.info.horizon for m in mdp]
    mdp_info = MDPInfo(mdp[max_obs_idx].info.observation_space,
                       mdp[max_act_idx].info.action_space, gammas, horizons)

    assert args.reg_type != 'kl' or args.features == 'sigmoid'

    scores = list()
    for _ in range(len(args.pendulum_mass)):
        scores.append(list())

    optimizer = dict()
    if args.optimizer == 'adam':
        optimizer['class'] = optim.Adam
        optimizer['params'] = dict(lr=args.learning_rate)
    elif args.optimizer == 'adadelta':
        optimizer['class'] = optim.Adadelta
        optimizer['params'] = dict(lr=args.learning_rate)
    elif args.optimizer == 'rmsprop':
        optimizer['class'] = optim.RMSprop
        optimizer['params'] = dict(lr=args.learning_rate,
                                   alpha=args.decay,
                                   eps=args.epsilon)
    elif args.optimizer == 'rmspropcentered':
        optimizer['class'] = optim.RMSprop
        optimizer['params'] = dict(lr=args.learning_rate,
                                   alpha=args.decay,
                                   eps=args.epsilon,
                                   centered=True)
    else:
        raise ValueError

    # FQI learning run

    # Settings
    if args.debug:
        test_samples = 20
        max_steps = 1000
    else:
        test_samples = args.test_samples
        max_steps = args.max_steps

    # Policy
    epsilon = Parameter(value=args.exploration_rate)
    epsilon_test = Parameter(value=args.test_exploration_rate)
    pi = EpsGreedyMultiple(parameter=epsilon,
                           n_actions_per_head=n_actions_per_head)

    # Approximator
    input_shape = [m.info.observation_space.shape for m in mdp]
    n_games = len(args.pendulum_mass)
    if args.reg_type == 'l1':
        regularized_loss = FeaturesL1Loss(args.reg_coeff)
    else:
        regularized_loss = FeaturesKLLoss(args.k, args.reg_coeff)

    approximator_params = dict(
        network=GymNetwork,
        input_shape=input_shape,
        output_shape=(max(n_actions_per_head)[0],),
        n_actions=max(n_actions_per_head)[0],
        n_actions_per_head=n_actions_per_head,
        optimizer=optimizer,
        loss=regularized_loss,
        reinitialize=True,
        use_cuda=args.use_cuda,
        dropout=args.dropout,
        features=args.features,
        quiet=False,
        batch_size=args.batch_size * n_games
    )

    approximator = PyTorchApproximator

    # Agent
    if args.n_fit_epochs == np.inf:
        fit_params = dict(n_epochs=args.n_fit_epochs, epsilon=args.fit_epsilon,
                          patience=args.fit_patience)
    else:
        fit_params = dict(n_epochs=args.n_fit_epochs)

    algorithm_params = dict(
        n_iterations=1,
        n_actions_per_head=n_actions_per_head,
        n_input_per_mdp=n_input_per_mdp,
        n_games=len(args.pendulum_mass),
        reg_type=args.reg_type,
        fit_params=fit_params,
        quiet=True)

    agent = FQI(approximator, pi, mdp_info,
                approximator_params=approximator_params,
                **algorithm_params)

    # Algorithm
    core = Core(agent, mdp)

    # RUN
    if args.transfer:
        weights = pickle.load(open(args.transfer, 'rb'))
        agent.set_shared_weights(weights)

    if args.load:
        weights = np.load(args.load)
        agent.approximator.set_weights(weights)

    print('- Learning:')
    # learning step
    if args.dataset is not None:
        with open(args.dataset, 'rb') as f:
            dataset = pickle.load(f)
    else:
        pi.set_parameter(None)
        dataset = core.evaluate(n_steps=max_steps, quiet=args.quiet)
        pickle.dump(dataset, open(folder_name + 'dataset.pkl', 'wb'))

    for it in trange(args.n_iterations):
        agent.fit(dataset)

        # evaluation step
        pi.set_parameter(epsilon_test)
        eval_dataset = core.evaluate(n_steps=test_samples, render=args.render,
                                     quiet=args.quiet)

        current_score_sum = 0
        tqdm.write('-- Iteration %d' % it)
        for i in range(len(mdp)):
            d = eval_dataset[i::len(mdp)]
            current_score = get_stats(d, gamma_eval, i, args.pendulum_mass)
            scores[i].append(current_score)
            current_score_sum += current_score

        # Save shared weights
        if args.save_shared:
            best_weights = agent.get_shared_weights()
            pickle.dump(best_weights, open(args.save_shared, 'wb'))

        if args.save:
            np.save(folder_name + 'weights-exp-%d-%d.npy' % (idx, it),
                    agent.approximator.get_weights())
            np.save(folder_name + 'targets-exp-%d-%d.npy' % (idx, it),
                    agent._target)
            np.save(folder_name + 'min-exp-%d-%d.npy' % (idx, it),
                    agent._min)
            np.save(folder_name + 'delta-exp-%d-%d.npy' % (idx, it),
                    agent._delta)

    np.save(folder_name + 'scores-exp-%d.npy' % idx, scores)

    return scores


if __name__ == '__main__':
    # Argument parser
    parser = argparse.ArgumentParser()

    arg_game = parser.add_argument_group('Game')
    arg_game.add_argument("--pendulum-mass", type=float, nargs='+')
    arg_game.add_argument("--n-exp", type=int)

    arg_net = parser.add_argument_group('Network params')
    arg_net.add_argument("--optimizer",
                         choices=['adadelta',
                                  'adam',
                                  'rmsprop',
                                  'rmspropcentered'],
                         default='adam',
                         help='Name of the optimizer to use to learn.')
    arg_net.add_argument("--learning-rate", type=float, default=1e-4,
                         help='Learning rate value of the optimizer.')
    arg_net.add_argument("--decay", type=float, default=.95,
                         help='Discount factor for the history coming from the'
                              'gradient momentum in rmspropcentered')
    arg_net.add_argument("--epsilon", type=float, default=1e-8,
                         help='Epsilon term used in rmspropcentered')
    arg_net.add_argument("--reg-coeff", type=float, default=0)
    arg_net.add_argument("--reg-type", type=str,
                         choices=['l1', 'l1-weights', 'gl1-weights', 'kl'])
    arg_net.add_argument("--k", type=float, default=10)
    arg_net.add_argument("--n-fit-epochs", type=int, default=np.inf)
    arg_net.add_argument("--fit-epsilon", type=float, default=1e-6)
    arg_net.add_argument("--fit-patience", type=int, default=50)
    arg_net.add_argument("--batch-size", type=int, default=5000,
                         help='Batch size for each fit of the network.')

    arg_alg = parser.add_argument_group('Algorithm')
    arg_alg.add_argument("--n-iterations", type=int, default=100,
                         help="Number of iterations of the FQI algorithm for "
                              "each fit call")
    arg_alg.add_argument("--features", choices=['relu', 'sigmoid'])
    arg_alg.add_argument("--dropout", action='store_true')
    arg_alg.add_argument("--max-steps", type=int, default=10000,
                         help='Total number of learning steps.')
    arg_alg.add_argument("--exploration-rate", type=float, default=1.,
                         help='Initial value of the exploration rate.')
    arg_alg.add_argument("--test-exploration-rate", type=float, default=0.,
                         help='Exploration rate used during evaluation.')
    arg_alg.add_argument("--test-samples", type=int, default=6000,
                         help='Number of steps for each evaluation.')
    arg_alg.add_argument("--transfer", type=str, default='',
                         help='Path to  the file of the weights of the common '
                              'layers to be loaded')
    arg_alg.add_argument("--save-shared", type=str, default='',
                         help='filename where to save the shared weights')
    arg_alg.add_argument("--unfreeze-epoch", type=int, default=0,
                         help="Number of epoch where to unfreeze shared weights.")

    arg_utils = parser.add_argument_group('Utils')
    arg_utils.add_argument('--use-cuda', action='store_true',
                           help='Flag specifying whether to use the GPU.')
    arg_utils.add_argument('--dataset', default=None, type=str)
    arg_utils.add_argument('--load', type=str,
                           help='Path of the model to be loaded.')
    arg_utils.add_argument('--save', action='store_true',
                           help='Flag specifying whether to save the model.')
    arg_utils.add_argument('--render', action='store_true',
                           help='Flag specifying whether to render the game.')
    arg_utils.add_argument('--quiet', action='store_true',
                           help='Flag specifying whether to hide the progress'
                                'bar.')
    arg_utils.add_argument('--debug', action='store_true',
                           help='Flag specifying whether the script has to be'
                                'run in debug mode.')
    arg_utils.add_argument('--postfix', type=str, default='',
                           help='Flag used to add a postfix to the folder name')

    args = parser.parse_args()

    folder_name = './logs/batch_gym_' + datetime.datetime.now().strftime(
        '%Y-%m-%d_%H-%M-%S') + args.postfix + '/'
    pathlib.Path(folder_name).mkdir(parents=True)
    with open(folder_name + 'args.pkl', 'wb') as f:
        pickle.dump(args, f)

    out = Parallel(n_jobs=-1)(delayed(experiment)(args, i)
                              for i in range(args.n_exp))

    scores = np.array(out)

    np.save(folder_name + 'scores.npy', scores)
