import argparse
import datetime
import pathlib
import sys

from joblib import delayed, Parallel
import numpy as np
import torch.optim as optim

import pickle

sys.path.append('..')

from mushroom_rl.approximators.parametric.torch_approximator import TorchApproximator
from mushroom_rl.environments import *
from mushroom_rl.utils.dataset import compute_J
from mushroom_rl.utils.parameters import LinearParameter, Parameter

from core import Core
from dqn import DQN, DoubleDQN
from policy import EpsGreedyMultiple
from networks import GymNetwork
from losses import LossFunction
from replay_memory import PrioritizedReplayMemory

"""
This script runs Atari experiments with DQN as presented in:
"Human-Level Control Through Deep Reinforcement Learning". Mnih V. et al.. 2015.

"""


def print_epoch(epoch):
    print('################################################################')
    print('Epoch: ', epoch)
    print('----------------------------------------------------------------')


def get_stats(dataset, gamma, idx, games):
    J = np.mean(compute_J(dataset, gamma[idx]))
    print(games[idx] + ': J: %f' % J)

    return J


def experiment(args, idx):
    np.random.seed()

    args.games = [''.join(g) for g in args.games]

    # MDP
    mdp = list()
    gamma_eval = list()
    for i, g in enumerate(args.games):
        if g == 'pendulum':
            mdp.append(CartPole(horizon=args.horizon[i], gamma=args.gamma[i]))
        elif g == 'caronhill':
            mdp.append(CarOnHill(horizon=args.horizon[i], gamma=args.gamma[i]))
        else:
            mdp.append(Gym(g, args.horizon[i], args.gamma[i]))

        gamma_eval.append(args.gamma[i])

    n_input_per_mdp = [m.info.observation_space.shape for m in mdp]
    n_actions_per_head = [(m.info.action_space.n,) for m in mdp]

    max_obs_dim = 0
    max_act_n = 0
    for i in range(len(args.games)):
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

    scores = list()
    for _ in range(len(args.games)):
        scores.append(list())

    optimizer = dict()
    if args.optimizer == 'adam':
        optimizer['class'] = optim.Adam
        optimizer['params'] = dict(lr=args.learning_rate,
                                   eps=args.epsilon)
    elif args.optimizer == 'adadelta':
        optimizer['class'] = optim.Adadelta
        optimizer['params'] = dict(lr=args.learning_rate,
                                   eps=args.epsilon)
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
    epsilon = LinearParameter(value=args.initial_exploration_rate,
                              threshold_value=args.final_exploration_rate,
                              n=args.final_exploration_frame)
    epsilon_test = Parameter(value=args.test_exploration_rate)
    epsilon_random = Parameter(value=1)
    pi = EpsGreedyMultiple(parameter=epsilon,
                           n_actions_per_head=n_actions_per_head)

    # Approximator
    input_shape = [m.info.observation_space.shape for m in mdp]
    n_games = len(args.games)
    loss = LossFunction(n_games, args.batch_size,
                        args.evaluation_frequency)

    approximator_params = dict(
        network=GymNetwork,
        input_shape=input_shape,
        output_shape=(max(n_actions_per_head)[0],),
        n_actions=max(n_actions_per_head)[0],
        n_actions_per_head=n_actions_per_head,
        optimizer=optimizer,
        loss=loss,
        use_cuda=args.use_cuda,
        features=args.features
    )

    approximator = TorchApproximator

    if args.prioritized:
        replay_memory = [PrioritizedReplayMemory(
            initial_replay_size, max_replay_size, alpha=.6,
            beta=LinearParameter(.4, threshold_value=1,
                                 n=max_steps // train_frequency)
        ) for _ in range(n_games)]
    else:
        replay_memory = None

    # Agent
    algorithm_params = dict(
        batch_size=args.batch_size,
        n_games=len(args.games),
        initial_replay_size=initial_replay_size,
        max_replay_size=max_replay_size,
        target_update_frequency=target_update_frequency // train_frequency,
        replay_memory=replay_memory,
        n_input_per_mdp=n_input_per_mdp,
        n_actions_per_head=n_actions_per_head,
        clip_reward=False,
        history_length=args.history_length,
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
        scores[i].append(get_stats(d, gamma_eval, i, args.games))

    if args.unfreeze_epoch > 0:
        agent.freeze_shared_weights()

    best_score_sum = -np.inf
    best_weights = None

    np.save(folder_name + 'scores-exp-%d.npy' % idx, scores)
    np.save(folder_name + 'loss-exp-%d.npy' % idx,
            agent.approximator.model._loss.get_losses())

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
            current_score = get_stats(d, gamma_eval, i, args.games)
            scores[i].append(current_score)
            current_score_sum += current_score

        # Save shared weights if best score
        if args.save_shared and current_score_sum >= best_score_sum:
            best_score_sum = current_score_sum
            best_weights = agent.get_shared_weights()

        if args.save:
            np.save(folder_name + 'weights-exp-%d-%d.npy' % (idx, n_epoch),
                    agent.approximator.get_weights())

        np.save(folder_name + 'scores-exp-%d.npy' % idx, scores)
        np.save(folder_name + 'loss-exp-%d.npy' % idx,
                agent.approximator.model._loss.get_losses())

    if args.save_shared:
        pickle.dump(best_weights, open(args.save_shared, 'wb'))

    return scores, agent.approximator.model._loss.get_losses()


if __name__ == '__main__':
    # Argument parser
    parser = argparse.ArgumentParser()

    arg_game = parser.add_argument_group('Game')
    arg_game.add_argument("--games",
                          type=list,
                          nargs='+',
                          default=['Acrobot-v1'],
                          help='Gym ID of the problem.')
    arg_game.add_argument("--horizon", type=int, nargs='+')
    arg_game.add_argument("--gamma", type=float, nargs='+')
    arg_game.add_argument("--n-exp", type=int)

    arg_mem = parser.add_argument_group('Replay Memory')
    arg_mem.add_argument("--initial-replay-size", type=int, default=100,
                         help='Initial size of the replay memory.')
    arg_mem.add_argument("--max-replay-size", type=int, default=5000,
                         help='Max size of the replay memory.')
    arg_mem.add_argument("--prioritized", action='store_true',
                         help='Whether to use prioritized memory or not.')

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

    arg_alg = parser.add_argument_group('Algorithm')
    arg_alg.add_argument("--algorithm", choices=['dqn', 'ddqn'],
                         default='dqn',
                         help='Name of the algorithm. dqn is for standard'
                              'DQN, ddqn is for Double DQN and adqn is for'
                              'Averaged DQN.')
    arg_alg.add_argument("--features", choices=['relu', 'sigmoid'])
    arg_alg.add_argument("--batch-size", type=int, default=100,
                         help='Batch size for each fit of the network.')
    arg_alg.add_argument("--history-length", type=int, default=1,
                         help='Number of frames composing a state.')
    arg_alg.add_argument("--target-update-frequency", type=int, default=100,
                         help='Number of collected samples before each update'
                              'of the target network.')
    arg_alg.add_argument("--evaluation-frequency", type=int, default=1000,
                         help='Number of learning step before each evaluation.'
                              'This number represents an epoch.')
    arg_alg.add_argument("--train-frequency", type=int, default=1,
                         help='Number of learning steps before each fit of the'
                              'neural network.')
    arg_alg.add_argument("--max-steps", type=int, default=50000,
                         help='Total number of learning steps.')
    arg_alg.add_argument("--final-exploration-frame", type=int, default=5000,
                         help='Number of steps until the exploration rate stops'
                              'decreasing.')
    arg_alg.add_argument("--initial-exploration-rate", type=float, default=1.,
                         help='Initial value of the exploration rate.')
    arg_alg.add_argument("--final-exploration-rate", type=float, default=.01,
                         help='Final value of the exploration rate. When it'
                              'reaches this values, it stays constant.')
    arg_alg.add_argument("--test-exploration-rate", type=float, default=0.,
                         help='Exploration rate used during evaluation.')
    arg_alg.add_argument("--test-samples", type=int, default=2000,
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

    folder_name = './logs/gym_' + datetime.datetime.now().strftime(
        '%Y-%m-%d_%H-%M-%S') + args.postfix + '/'
    pathlib.Path(folder_name).mkdir(parents=True)
    with open(folder_name + 'args.pkl', 'wb') as f:
        pickle.dump(args, f)

    out = Parallel(n_jobs=-1)(delayed(experiment)(args, i)
                              for i in range(args.n_exp))

    scores = np.array([o[0] for o in out])
    loss = np.array([o[1] for o in out])

    np.save(folder_name + 'scores.npy', scores)
    np.save(folder_name + 'loss.npy', loss)
