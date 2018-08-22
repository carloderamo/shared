import argparse
import datetime
import pathlib
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

sys.path.append('..')

from mushroom.approximators.parametric import PyTorchApproximator

from mushroom.environments import *
from mushroom.utils.dataset import compute_scores
from mushroom.utils.parameters import LinearDecayParameter, Parameter

from core import Core
from dqn import DQN
from policy import EpsGreedyEnsemble

"""
This script runs Atari experiments with DQN as presented in:
"Human-Level Control Through Deep Reinforcement Learning". Mnih V. et al.. 2015.

"""


class Network(nn.Module):
    def __init__(self, input_shape, output_shape, **kwargs):
        super(Network, self).__init__()

        n_input = input_shape[0]
        n_output = int(output_shape[0])

        self.n_features = 3136

        self._h1 = nn.Conv2d(n_input, 32, kernel_size=8, stride=4)
        self._h2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self._h3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self._h4 = nn.Linear(self.n_features, 512)
        self._h5 = nn.Linear(512, n_output)

        nn.init.xavier_uniform_(self._h1.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h2.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h3.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h4.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h5.weight,
                                gain=nn.init.calculate_gain('linear'))

    def forward(self, state, action=None, get_type=0):
        h = F.relu(self._h1(state.float() / 255.))
        h = F.relu(self._h2(h))
        h = F.relu(self._h3(h))
        h_f = h.view(-1, self.n_features)
        h = F.relu(self._h4(h_f))
        q = self._h5(h)

        if action is None:
            if get_type == 0:
                return q
            elif get_type == 1:
                return h_f
            elif get_type == 2:
                return q, h_f
        else:
            q_acted = torch.squeeze(q.gather(1, action.long()))

            if get_type == 0:
                return q_acted
            elif get_type == 1:
                return h_f
            elif get_type == 2:
                return q_acted, h_f


class DistilledNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, **kwargs):
        super(DistilledNetwork, self).__init__()

        n_input = input_shape[0]

        self.n_features = 3136

        self._h1 = nn.Conv2d(n_input, 32, kernel_size=8, stride=4)
        self._h2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self._h3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        nn.init.xavier_uniform_(self._h1.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h2.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h3.weight,
                                gain=nn.init.calculate_gain('relu'))

    def forward(self, state, action=None):
        h = F.relu(self._h1(state.float() / 255.))
        h = F.relu(self._h2(h))
        features = F.relu(self._h3(h))

        return features.view(-1, self.n_features)


def print_epoch(epoch):
    print('################################################################')
    print('Epoch: ', epoch)
    print('----------------------------------------------------------------')


def get_stats(dataset, idx, games):
    score = compute_scores(dataset)
    print((games[idx] + ': min_reward: %f, max_reward: %f, mean_reward: %f,'
          ' games_completed: %d' % score))

    return score


def experiment():
    np.random.seed()

    # Argument parser
    parser = argparse.ArgumentParser()

    arg_game = parser.add_argument_group('Game')
    arg_game.add_argument("--games",
                          type=list,
                          nargs='+',
                          default=['BreakoutNoFrameskip-v4'],
                          help='Gym ID of the Atari game.')
    arg_game.add_argument("--screen-width", type=int, default=84,
                          help='Width of the game screen.')
    arg_game.add_argument("--screen-height", type=int, default=84,
                          help='Height of the game screen.')

    arg_mem = parser.add_argument_group('Replay Memory')
    arg_mem.add_argument("--initial-replay-size", type=int, default=50000,
                         help='Initial size of the replay memory.')
    arg_mem.add_argument("--max-replay-size", type=int, default=500000,
                         help='Max size of the replay memory.')

    arg_net = parser.add_argument_group('Deep Q-Network')
    arg_net.add_argument("--optimizer",
                         choices=['adadelta',
                                  'adam',
                                  'rmsprop',
                                  'rmspropcentered'],
                         default='rmspropcentered',
                         help='Name of the optimizer to use.')
    arg_net.add_argument("--learning-rate", type=float, default=.00025,
                         help='Learning rate value of the optimizer.')
    arg_net.add_argument("--decay", type=float, default=.95,
                         help='Discount factor for the history coming from the'
                              'gradient momentum in rmspropcentered and'
                              'rmsprop')
    arg_net.add_argument("--epsilon", type=float, default=1e-8,
                         help='Epsilon term used in rmspropcentered and'
                              'rmsprop')
    arg_net.add_argument("--reg-coeff", type=float, default=.5)

    arg_alg = parser.add_argument_group('Algorithm')
    arg_alg.add_argument("--entropy-coeff", type=float, default=np.inf)
    arg_alg.add_argument("--batch-size", type=int, default=32,
                         help='Batch size for each fit of the network.')
    arg_alg.add_argument("--history-length", type=int, default=4,
                         help='Number of frames composing a state.')
    arg_alg.add_argument("--target-update-frequency", type=int, default=10000,
                         help='Number of collected samples before each update'
                              'of the target network.')
    arg_alg.add_argument("--evaluation-frequency", type=int, default=250000,
                         help='Number of collected samples before each'
                              'evaluation. An epoch ends after this number of'
                              'steps')
    arg_alg.add_argument("--train-frequency", type=int, default=4,
                         help='Number of collected samples before each fit of'
                              'the neural network.')
    arg_alg.add_argument("--max-steps", type=int, default=50000000,
                         help='Total number of collected samples.')
    arg_alg.add_argument("--final-exploration-frame", type=int, default=1000000,
                         help='Number of collected samples until the exploration'
                              'rate stops decreasing.')
    arg_alg.add_argument("--initial-exploration-rate", type=float, default=1.,
                         help='Initial value of the exploration rate.')
    arg_alg.add_argument("--final-exploration-rate", type=float, default=.1,
                         help='Final value of the exploration rate. When it'
                              'reaches this values, it stays constant.')
    arg_alg.add_argument("--test-exploration-rate", type=float, default=.05,
                         help='Exploration rate used during evaluation.')
    arg_alg.add_argument("--test-samples", type=int, default=125000,
                         help='Number of collected samples for each'
                              'evaluation.')
    arg_alg.add_argument("--max-no-op-actions", type=int, default=30,
                         help='Maximum number of no-op actions performed at the'
                              'beginning of the episodes.')

    arg_utils = parser.add_argument_group('Utils')
    arg_utils.add_argument('--use-cuda', action='store_true',
                           help='Flag specifying whether to use the GPU.')
    arg_utils.add_argument('--load-path', type=str,
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

    args = parser.parse_args()

    args.games = [''.join(g) for g in args.games]

    def regularized_loss(loss_args, y, f):
        y_hat, f_hat = loss_args

        return F.smooth_l1_loss(y_hat, y) + args.reg_coeff * F.mse_loss(f_hat,
                                                                        f)

    scores = list()
    for _ in range(len(args.games)):
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

    mdp = list()
    for g in args.games:
        mdp.append(Atari(g, args.screen_width, args.screen_height,
                         ends_at_life=True, history_length=args.history_length,
                         max_no_op_actions=args.max_no_op_actions)
                   )
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
    mdp_info = MDPInfo(mdp[max_obs_idx].info.observation_space,
                       mdp[max_act_idx].info.action_space, mdp[0].info.gamma,
                       mdp[0].info.horizon)

    # Evaluation of the model provided by the user.
    if args.load_path:
        # Policy
        epsilon_test = Parameter(value=args.test_exploration_rate)
        pi = EpsGreedyEnsemble(epsilon=epsilon_test, n=len(mdp))

        # Approximator
        input_shape = (args.history_length, args.screen_height,
                       args.screen_width)
        approximator_params = [dict(
            network=Network,
            input_shape=input_shape,
            output_shape=(mdp[i].info.action_space.n,),
            n_actions=mdp[i].info.action_space.n,
            n_fit_targets=2,
            optimizer=optimizer,
            loss=regularized_loss,
            use_cuda=args.use_cuda
        ) for i in range(len(args.games))]

        distilled_params = dict(
            network=DistilledNetwork,
            input_shape=input_shape,
            output_shape=3136,
            optimizer=optimizer,
            loss=F.mse_loss,
            use_cuda=args.use_cuda)

        approximator = PyTorchApproximator

        # Agent
        algorithm_params = dict(
            distilled_params=distilled_params,
            batch_size=1,
            train_frequency=1,
            target_update_frequency=1,
            initial_replay_size=0,
            max_replay_size=0,
            n_actions_per_head=n_actions_per_head,
            history_length=args.history_length,
            dtype=np.uint8,
            entropy_coeff=args.entropy_coeff
        )
        agent = DQN(approximator, pi, mdp_info,
                    approximator_params=approximator_params, **algorithm_params)

        # Algorithm
        core_test = Core(agent, mdp)

        # Evaluate model
        pi.set_epsilon(epsilon_test)
        dataset = core_test.evaluate(n_steps=args.test_samples,
                                     render=args.render,
                                     quiet=args.quiet)
        #get_stats(dataset)
    else:
        # DQN learning run

        # Summary folder
        folder_name = './logs/atari_' + datetime.datetime.now().strftime(
            '%Y-%m-%d_%H-%M-%S')
        pathlib.Path(folder_name).mkdir(parents=True)

        # Settings
        if args.debug:
            initial_replay_size = 50
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
        pi = EpsGreedyEnsemble(epsilon=epsilon, n=len(mdp))

        # Approximator
        input_shape = (args.history_length, args.screen_height,
                       args.screen_width)
        approximator_params = [dict(
            network=Network,
            input_shape=input_shape,
            output_shape=(mdp[i].info.action_space.n,),
            n_fit_targets=2,
            n_actions=mdp[i].info.action_space.n,
            optimizer=optimizer,
            loss=regularized_loss,
            use_cuda=args.use_cuda
        ) for i in range(len(args.games))]

        distilled_params = dict(
            network=DistilledNetwork,
            input_shape=input_shape,
            output_shape=3136,
            optimizer=optimizer,
            loss=F.mse_loss,
            use_cuda=args.use_cuda)

        approximator = PyTorchApproximator

        # Agent
        algorithm_params = dict(
            distilled_params=distilled_params,
            batch_size=args.batch_size,
            n_games=len(args.games),
            initial_replay_size=initial_replay_size,
            max_replay_size=max_replay_size,
            n_actions_per_head=n_actions_per_head,
            history_length=args.history_length,
            target_update_frequency=target_update_frequency // train_frequency,
            dtype=np.uint8,
            entropy_coeff=args.entropy_coeff
        )

        agent = DQN(approximator, pi, mdp_info,
                    approximator_params=approximator_params,
                    **algorithm_params)

        # Algorithm
        core = Core(agent, mdp)

        # RUN

        # Fill replay memory with random dataset
        print_epoch(0)
        pi.set_epsilon(epsilon_random)
        core.learn(n_steps=initial_replay_size,
                   n_steps_per_fit=initial_replay_size, quiet=args.quiet)

        if args.save:
            agent.approximator.model.save()

        for m in mdp:
            m.set_episode_end(False)
        # Evaluate initial policy
        pi.set_epsilon(epsilon_test)
        dataset = core.evaluate(n_steps=test_samples, render=args.render,
                                quiet=args.quiet)
        for i in range(len(mdp)):
            d = dataset[i::len(mdp)]
            scores[i].append(get_stats(d, i, args.games))

        np.save(folder_name + '/scores.npy', scores)
        for n_epoch in range(1, max_steps // evaluation_frequency + 1):
            print_epoch(n_epoch)
            print('- Learning:')
            # learning step
            for m in mdp:
                m.set_episode_end(True)
            pi.set_epsilon(None)
            core.learn(n_steps=evaluation_frequency,
                       n_steps_per_fit=train_frequency, quiet=args.quiet)

            if args.save:
                agent.approximator.model.save()

            print('- Evaluation:')
            # evaluation step
            for m in mdp:
                m.set_episode_end(False)
            pi.set_epsilon(epsilon_test)
            dataset = core.evaluate(n_steps=test_samples,
                                    render=args.render, quiet=args.quiet)
            for i in range(len(mdp)):
                d = dataset[i::len(mdp)]
                scores[i].append(get_stats(d, i, args.games))

            np.save(folder_name + '/scores.npy', scores)

    return scores


if __name__ == '__main__':
    experiment()
