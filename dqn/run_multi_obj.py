import argparse
import datetime
import pathlib
import sys

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from joblib import Parallel, delayed

import pickle

sys.path.append('..')

from mushroom.approximators.parametric import PyTorchApproximator
from mushroom.environments import *
from mushroom.utils.dataset import compute_J
from mushroom.utils.parameters import LinearDecayParameter, Parameter

from core import Core
from dqn import DQN, DoubleDQN
from policy import EpsGreedyMultiple
from networks import GymNetwork
from losses import *

"""
This script runs Atari experiments with DQN as presented in:
"Human-Level Control Through Deep Reinforcement Learning". Mnih V. et al.. 2015.

"""


def print_epoch(epoch):
    print('################################################################')
    print('Epoch: ', epoch)
    print('----------------------------------------------------------------')


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

    # DQN learning run

    # Settings
    if args.debug:
        initial_replay_size = args.batch_size
        max_replay_size = 500
        train_frequency = 5
        target_update_frequency = 10
        test_samples = 20
        evaluation_frequency = 50
        max_steps = 1000
    else:
        initial_replay_size = args.initial_replay_size
        max_replay_size = args.max_replay_size
        train_frequency = args.train_frequency
        target_update_frequency = args.target_update_frequency
        test_samples = args.test_samples
        evaluation_frequency = args.evaluation_frequency
        max_steps = args.max_steps

    # Policy
    epsilon = LinearDecayParameter(value=args.initial_exploration_rate,
                                   min_value=args.final_exploration_rate,
                                   n=args.final_exploration_frame)
    epsilon_test = Parameter(value=args.test_exploration_rate)
    epsilon_random = Parameter(value=1)
    pi = EpsGreedyMultiple(parameter=epsilon,
                           n_actions_per_head=n_actions_per_head)

    # Approximator
    input_shape = [m.info.observation_space.shape for m in mdp]
    n_games = len(args.games)
    if args.reg_type == 'l1':
        regularized_loss = FeaturesL1Loss(args.reg_coeff, n_games,
                                          args.batch_size,
                                          args.evaluation_frequency)
    elif args.reg_type == 'l1-weights':
        regularized_loss = WeightsL1Loss(n_actions_per_head, args.reg_coeff,
                                         n_games, args.batch_size,
                                         args.evaluation_frequency)
    elif args.reg_type == 'gl1-weights':
        regularized_loss = WeightsGLLoss(n_actions_per_head, args.reg_coeff,
                                         n_games, args.batch_size,
                                         args.evaluation_frequency)
    else:
        regularized_loss = FeaturesKLLoss(args.k, args.reg_coeff, n_games,
                                          args.batch_size,
                                          args.evaluation_frequency)
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
        n_features=args.n_features
    )

    approximator = PyTorchApproximator

    # Agent
    algorithm_params = dict(
        batch_size=args.batch_size,
        n_games=n_games,
        initial_replay_size=initial_replay_size,
        max_replay_size=max_replay_size,
        target_update_frequency=target_update_frequency // train_frequency,
        n_input_per_mdp=n_input_per_mdp,
        n_actions_per_head=n_actions_per_head,
        clip_reward=False,
        history_length=args.history_length,
        reg_type=args.reg_type,
        dtype=np.float32
    )

    if args.algorithm == 'dqn':
        agent = DQN(approximator, pi, mdp_info,
                    approximator_params=approximator_params,
                    **algorithm_params)
    elif args.algorithm == 'ddqn':
        agent = DoubleDQN(approximator, pi, mdp_info,
                          approximator_params=approximator_params,
                          **algorithm_params)
    else:
        raise ValueError

    # Algorithm
    core = Core(agent, mdp)

    # RUN

    # Fill replay memory with random dataset
    print_epoch(0)
    pi.set_parameter(epsilon_random)
    core.learn(n_steps=initial_replay_size,
               n_steps_per_fit=initial_replay_size, quiet=args.quiet)

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

    if args.unfreeze_epoch > 0:
        agent.freeze_shared_weights()

    best_score_sum = -np.inf
    best_weights = None

    for n_epoch in range(1, max_steps // evaluation_frequency + 1):
        if n_epoch >= args.unfreeze_epoch > 0:
            agent.unfreeze_shared_weights()

        print_epoch(n_epoch)
        print('- Learning:')
        # learning step
        pi.set_parameter(None)
        core.learn(n_steps=evaluation_frequency,
                   n_steps_per_fit=train_frequency, quiet=args.quiet)

        print('- Evaluation:')
        # evaluation step
        pi.set_parameter(epsilon_test)
        dataset = core.evaluate(n_steps=test_samples,
                                render=args.render, quiet=args.quiet)

        current_score_sum = 0
        for i in range(len(mdp)):
            d = dataset[i::len(mdp)]
            current_score = get_stats(d, gamma_eval, i)
            scores[i].append(current_score)
            current_score_sum += current_score

        # Save shared weights if best score
        if args.save_shared and current_score_sum >= best_score_sum:
            best_score_sum = current_score_sum
            best_weights = agent.get_shared_weights()

        if args.save:
            np.save(folder_name + 'weights-%d.npy' % n_epoch,
                    agent.approximator.get_weights())

    if args.save_shared:
        pickle.dump(best_weights, open(args.save_shared, 'wb'))

    return scores, regularized_loss.get_losses(), \
           regularized_loss.get_reg_losses(), agent.v_list


if __name__ == '__main__':
    # Argument parser
    parser = argparse.ArgumentParser()

    arg_game = parser.add_argument_group('Game')
    arg_game.add_argument("--game", type=str)
    arg_game.add_argument("--multi", action='store_true')
    arg_game.add_argument("--n-games", type=int, default=100,
                          help='number of games to be played')

    arg_mem = parser.add_argument_group('Replay Memory')
    arg_mem.add_argument("--initial-replay-size", type=int, default=100,
                         help='Initial size of the replay memory.')
    arg_mem.add_argument("--max-replay-size", type=int, default=5000,
                         help='Max size of the replay memory.')


    arg_net = parser.add_argument_group('Deep Q-Network')
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
    arg_net.add_argument("--n-features", type=int, default=80,
                         help='number of features of the network')

    arg_alg = parser.add_argument_group('Algorithm')
    arg_alg.add_argument("--algorithm", default='dqn', choices=['dqn', 'ddqn'])
    arg_alg.add_argument("--features", choices=['relu', 'sigmoid'])
    arg_alg.add_argument("--dropout", action='store_true')
    arg_alg.add_argument("--batch-size", type=int, default=100,
                         help='Batch size for each fit of the network.')
    arg_alg.add_argument("--history-length", type=int, default=1,
                         help='Number of frames composing a state.')
    arg_alg.add_argument("--target-update-frequency", type=int, default=100,
                         help='Number of collected samples before each update'
                              'of the target network.')
    arg_alg.add_argument("--evaluation-frequency", type=int, default=2000,
                         help='Number of learning step before each evaluation.'
                              'This number represents an epoch.')
    arg_alg.add_argument("--train-frequency", type=int, default=1,
                         help='Number of learning steps before each fit of the'
                              'neural network.')
    arg_alg.add_argument("--max-steps", type=int, default=50000,
                         help='Total number of learning steps.')
    arg_alg.add_argument("--final-exploration-frame", type=int, default=50000,
                         help='Number of steps until the exploration rate stops'
                              'decreasing.')
    arg_alg.add_argument("--initial-exploration-rate", type=float, default=1.,
                         help='Initial value of the exploration rate.')
    arg_alg.add_argument("--final-exploration-rate", type=float, default=.1,
                         help='Final value of the exploration rate. When it'
                              'reaches this values, it stays constant.')
    arg_alg.add_argument("--test-exploration-rate", type=float, default=0.,
                         help='Exploration rate used during evaluation.')
    arg_alg.add_argument("--test-samples", type=int, default=1000,
                         help='Number of steps for each evaluation.')
    arg_alg.add_argument("--max-no-op-actions", type=int, default=0,
                         help='Maximum number of no-op action performed at the'
                              'beginning of the episodes. The minimum number is'
                              'history_length.')
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
        scores, loss, l1_loss, v = experiment(0, args.n_games, args)
    else:
        out = Parallel(n_jobs=-1)(
            delayed(experiment)(i, i + 1, args) for i in range(args.n_games))
        scores = [x[0] for x in out]
        loss = [x[1] for x in out]
        l1_loss = [x[2] for x in out]
        v = [x[3] for x in out]

    np.save(folder_name + 'scores.npy', scores)
    np.save(folder_name + 'loss.npy', loss)
    np.save(folder_name + 'reg_loss.npy', l1_loss)
    np.save(folder_name + 'v_raw.npy', v)