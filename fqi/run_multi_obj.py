import argparse
import datetime
import pathlib
import sys

from joblib import delayed, Parallel
import numpy as np
import torch.optim as optim

import pickle

sys.path.append('..')

from mushroom.approximators.parametric import PyTorchApproximator
from mushroom.environments import *
from mushroom.utils.dataset import compute_J
from mushroom.utils.parameters import Parameter

from core import Core
from fqi import FQI
from policy import EpsGreedyMultiple
from networks import GymNetwork
from losses import *


def get_stats(dataset, gamma, idx):
    J = np.mean(compute_J(dataset, gamma))
    print(str(idx) + ': J: %f' % J)

    return J


def experiment(start, end, args):
    np.random.seed()

    n_games = end - start

    # MDP
    mdp = list()
    starts = np.load('puddle/start.npy')
    goals = np.load('puddle/goal.npy')
    for i in range(start, end):
        if args.game == 'puddleworld':
            mdp.append(PuddleWorld(start=starts[i], goal=goals[i],
                                   horizon=1000))
        else:
            raise ValueError

    n_input_per_mdp = [m.info.observation_space.shape for m in mdp]
    n_actions_per_head = [(m.info.action_space.n,) for m in mdp]

    gamma_eval = mdp[0].info.gamma
    mdp_info = MDPInfo(mdp[0].info.observation_space, mdp[0].info.action_space,
                       [mdp[0].info.gamma] * n_games,
                       [mdp[0].info.horizon] * n_games)

    assert args.reg_type != 'kl' or args.features == 'sigmoid'

    scores = list()
    for _ in range(n_games):
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
        use_cuda=args.use_cuda,
        dropout=args.dropout,
        features=args.features,
        quiet=False,
        batch_size=args.batch_size
    )

    approximator = PyTorchApproximator

    # Agent
    fit_params = dict(n_epochs=args.n_fit_epochs, epsilon=args.fit_epsilon,
                      patience=args.fit_patience)

    algorithm_params = dict(
        n_iterations=args.n_iterations,
        n_actions_per_head=n_actions_per_head,
        n_input_per_mdp=n_input_per_mdp,
        n_games=n_games,
        reg_type=args.reg_type,
        fit_params=fit_params,
        quiet=False)

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

    # Evaluate initial policy
    pi.set_parameter(epsilon_test)
    dataset = core.evaluate(n_steps=test_samples, render=args.render,
                            quiet=args.quiet)
    for i in range(len(mdp)):
        d = dataset[i::len(mdp)]
        scores[i].append(get_stats(d, gamma_eval, i))

    print('- Learning:')
    # learning step
    pi.set_parameter(None)
    core.learn(n_steps=max_steps,
               n_steps_per_fit=max_steps,
               quiet=args.quiet)

    print('- Evaluation:')
    # evaluation step
    pi.set_parameter(epsilon_test)
    dataset = core.evaluate(n_steps=test_samples,
                            render=args.render,
                            quiet=args.quiet)

    current_score_sum = 0
    for i in range(len(mdp)):
        d = dataset[i::len(mdp)]
        current_score = get_stats(d, gamma_eval, i)
        scores[i].append(current_score)
        current_score_sum += current_score

    # Save shared weights
    if args.save_shared:
        best_weights = agent.get_shared_weights()
        pickle.dump(best_weights, open(args.save_shared, 'wb'))

    if args.save:
        n_epoch = 1  # TODO: change this if more epochs are needed
        np.save(folder_name + 'weights-exp-%d.npy' % n_epoch,
                agent.approximator.get_weights())

    return scores


if __name__ == '__main__':
    # Argument parser
    parser = argparse.ArgumentParser()

    arg_game = parser.add_argument_group('Game')
    arg_game.add_argument("--game", type=str)
    arg_game.add_argument("--multi", action='store_true')
    arg_game.add_argument("--n-games", type=int, default=100,
                          help='number of games to be played')

    arg_net = parser.add_argument_group('Network params')
    arg_net.add_argument("--optimizer",
                         choices=['adadelta',
                                  'adam',
                                  'rmsprop',
                                  'rmspropcentered'],
                         default='adam',
                         help='Name of the optimizer to use to learn.')
    arg_net.add_argument("--learning-rate", type=float, default=.001,
                         help='Learning rate value of the optimizer. Only used'
                              'in rmspropcentered')
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
    arg_net.add_argument("--fit-epsilon", type=int, default=1e-6)
    arg_net.add_argument("--fit-patience", type=int, default=20)
    arg_net.add_argument("--batch-size", type=int, default=100,
                         help='Batch size for each fit of the network.')

    arg_alg = parser.add_argument_group('Algorithm')
    arg_alg.add_argument("--n-iterations", type=int, default=30,
                         help="Number of iterations of the FQI algorithm for "
                              "each fit call")
    arg_alg.add_argument("--features", choices=['relu', 'sigmoid'])
    arg_alg.add_argument("--dropout", action='store_true')
    arg_alg.add_argument("--max-steps", type=int, default=50000,
                         help='Total number of learning steps.')
    arg_alg.add_argument("--exploration-rate", type=float, default=1.,
                         help='Initial value of the exploration rate.')
    arg_alg.add_argument("--test-exploration-rate", type=float, default=0.,
                         help='Exploration rate used during evaluation.')
    arg_alg.add_argument("--test-samples", type=int, default=2000,
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

    folder_name = './logs/multitask_' + datetime.datetime.now().strftime(
        '%Y-%m-%d_%H-%M-%S') + args.postfix + '/'
    pathlib.Path(folder_name).mkdir(parents=True)

    with open(folder_name + 'args.pkl', 'wb') as f:
        pickle.dump(args, f)

    if args.multi:
        out = experiment(0, args.n_games, args)
    else:
        out = Parallel(n_jobs=-1)(
            delayed(experiment)(i, i + 1, args) for i in range(args.n_games))

    scores = np.array(out)

    np.save(folder_name + 'scores.npy', scores)
