import argparse
import datetime
import pathlib

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from mushroom.approximators.parametric import PyTorchApproximator
from mushroom.core import Core
from mushroom.utils.dataset import compute_scores
from mushroom.utils.parameters import LinearDecayParameter, Parameter

from atari import AtariMultiple
from dqn import DQN
from policy import EpsGreedyMultiple

"""
This script runs Atari experiments with DQN as presented in:
"Human-Level Control Through Deep Reinforcement Learning". Mnih V. et al.. 2015.

"""


class Network(nn.Module):
    def __init__(self, input_shape, _, n_actions_per_head, use_cuda, dropout):
        super(Network, self).__init__()

        n_input = input_shape[0]
        self._n_games = len(n_actions_per_head)
        self._max_actions = max(n_actions_per_head)[0]
        self._use_cuda = use_cuda
        self._dropout = dropout
        self._n_shared = 6

        self._h1 = nn.Conv2d(n_input, 32, kernel_size=8, stride=4)
        self._h2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self._h3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self._h4 = nn.ModuleList([nn.Linear(3136, 512) for _ in range(
            self._n_games)])
        self._h5 = nn.ModuleList(
            [nn.Linear(512, self._max_actions) for _ in range(
                self._n_games)])

        if self._dropout:
            self._h1_dropout = nn.Dropout2d()
            self._h2_dropout = nn.Dropout2d()
            self._h3_dropout = nn.Dropout2d()

        nn.init.xavier_uniform_(self._h1.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h2.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h3.weight,
                                gain=nn.init.calculate_gain('relu'))
        for i in range(self._n_games):
            nn.init.xavier_uniform_(self._h4[i].weight,
                                    gain=nn.init.calculate_gain('relu'))
            nn.init.xavier_uniform_(self._h5[i].weight,
                                    gain=nn.init.calculate_gain('linear'))

    def forward(self, state, action=None, idx=None, get_features=False):
        h = F.relu(self._h1(state.float() / 255.))
        if self._dropout:
            h = F.relu(self._h2(self._h1_dropout(h)))
            h = F.relu(self._h3(self._h2_dropout(h)))
            h_f = self._h3_dropout(h).view(-1, 3136)
        else:
            h = F.relu(self._h2(h))
            h = F.relu(self._h3(h))
            h_f = h.view(-1, 3136)

        features = list()
        q = list()

        for i in range(self._n_games):
            features.append(F.relu(self._h4[i](h_f)))
            q.append(self._h5[i](features[i]))

        q = torch.stack(q, dim=1)

        if action is not None:
            action = action.long()
            q_acted = torch.squeeze(
                q.gather(2, action.repeat(1, self._n_games).unsqueeze(-1)), -1)

            q = q_acted

        if idx is not None:
            idx = torch.from_numpy(idx)
            if self._use_cuda:
                idx = idx.cuda()
            if q.dim() == 2:
                q_idx = q.gather(1, idx.unsqueeze(-1))
            else:
                q_idx = q.gather(1, idx.view(-1, 1).repeat(
                    1, self._max_actions).unsqueeze(1))

            q = torch.squeeze(q_idx, 1)

        if get_features:
            return q, h_f
        else:
            return q


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
    arg_net.add_argument("--reg-coeff", type=float, default=0.)

    arg_alg = parser.add_argument_group('Algorithm')
    arg_alg.add_argument("--dropout", action='store_true')
    arg_alg.add_argument("--distill", action='store_true')
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
    arg_alg.add_argument("--max-no-op-actions", type=int, default=8,
                         help='Maximum number of no-op actions performed at the'
                              'beginning of the episodes. The minimum number is'
                              'history_length. This number is reported to be 30'
                              'in the DQN Deepmind paper but, since they'
                              'consider the first 30 frames without frame'
                              'skipping and that the number of skipped frames'
                              'is generally 4, we set it to 8.')
    arg_alg.add_argument("--no-op-action-value", type=int, default=0,
                         help='Value of the no-op action.')

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

    def regularized_loss(arg, y):
        yhat, h_f = arg
        loss = F.smooth_l1_loss(yhat, y) * len(args.games)

        return loss + args.reg_coeff * torch.norm(h_f, 1)

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

    # Evaluation of the model provided by the user.
    if args.load_path:
        # MDP
        mdp = AtariMultiple(args.games, args.screen_width, args.screen_height,
                            ends_at_life=False,
                            n_steps_per_game=args.batch_size)
        n_actions_per_head = [(m.info.action_space.n,) for m in mdp.envs]

        # Policy
        epsilon_test = Parameter(value=args.test_exploration_rate)
        pi = EpsGreedyMultiple(epsilon=epsilon_test,
                               n_actions_per_head=n_actions_per_head)

        # Approximator
        input_shape = (args.history_length, args.screen_height,
                       args.screen_width)
        approximator_params = dict(
            network=Network,
            input_shape=input_shape,
            output_shape=(mdp.info.action_space.n,),
            n_actions=mdp.info.action_space.n,
            n_actions_per_head=n_actions_per_head,
            load_path=args.load_path,
            optimizer=optimizer,
            loss=regularized_loss,
            use_cuda=args.use_cuda,
            dropout=args.dropout
        )

        approximator = PyTorchApproximator

        # Agent
        algorithm_params = dict(
            batch_size=1,
            train_frequency=1,
            target_update_frequency=1,
            n_actions_per_head=n_actions_per_head,
            initial_replay_size=0,
            max_replay_size=0,
            history_length=args.history_length,
            max_no_op_actions=args.max_no_op_actions,
            no_op_action_value=args.no_op_action_value,
            dtype=np.uint8,
            distill=args.distill,
            entropy_coeff=args.entropy_coeff
        )
        agent = DQN(approximator, pi, mdp.info,
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

        # MDP
        mdp = AtariMultiple(args.games, args.screen_width, args.screen_height,
                            ends_at_life=True, n_steps_per_game=args.batch_size)
        n_actions_per_head = [(m.info.action_space.n,) for m in mdp.envs]

        # Policy
        epsilon = LinearDecayParameter(value=args.initial_exploration_rate,
                                       min_value=args.final_exploration_rate,
                                       n=args.final_exploration_frame)
        epsilon_test = Parameter(value=args.test_exploration_rate)
        epsilon_random = Parameter(value=1)
        pi = EpsGreedyMultiple(epsilon=epsilon,
                               n_actions_per_head=n_actions_per_head)

        # Approximator
        input_shape = (args.history_length, args.screen_height,
                       args.screen_width)
        approximator_params = dict(
            network=Network,
            input_shape=input_shape,
            output_shape=(mdp.info.action_space.n,),
            n_actions=mdp.info.action_space.n,
            n_actions_per_head=n_actions_per_head,
            optimizer=optimizer,
            loss=regularized_loss,
            use_cuda=args.use_cuda,
            dropout=args.dropout
        )

        approximator = PyTorchApproximator

        # Agent
        algorithm_params = dict(
            batch_size=args.batch_size,
            n_games=len(args.games),
            initial_replay_size=initial_replay_size,
            max_replay_size=max_replay_size,
            history_length=args.history_length,
            target_update_frequency=target_update_frequency // train_frequency,
            n_actions_per_head=n_actions_per_head,
            max_no_op_actions=args.max_no_op_actions,
            no_op_action_value=args.no_op_action_value,
            dtype=np.uint8,
            distill=args.distill,
            entropy_coeff=args.entropy_coeff
        )

        agent = DQN(approximator, pi, mdp.info,
                    approximator_params=approximator_params,
                    **algorithm_params)

        # Algorithm
        core = Core(agent, mdp)

        # RUN

        # Fill replay memory with random dataset
        print_epoch(0)
        mdp.freeze_env(True)
        for idx in range(len(args.games)):
            mdp.set_env(idx)
            pi.set_epsilon(epsilon_random)
            core.learn(n_steps=initial_replay_size,
                       n_steps_per_fit=initial_replay_size, quiet=args.quiet)

        if args.save:
            agent.approximator.model.save()

        # Evaluate initial policy
        for idx in range(len(args.games)):
            mdp.set_episode_end(False)
            mdp.set_env(idx)
            pi.set_epsilon(epsilon_test)
            dataset = core.evaluate(n_steps=test_samples, render=args.render,
                                    quiet=args.quiet)
            scores[idx].append(get_stats(dataset, idx, args.games))

        np.save(folder_name + '/scores.npy', scores)
        for n_epoch in range(1, max_steps // evaluation_frequency + 1):
            print_epoch(n_epoch)
            print('- Learning:')
            # learning step
            mdp.freeze_env(False)
            mdp.set_episode_end(True)
            mdp.set_env(None)
            pi.set_epsilon(None)
            core.learn(n_steps=evaluation_frequency,
                       n_steps_per_fit=train_frequency, quiet=args.quiet)

            if args.save:
                agent.approximator.model.save()

            print('- Evaluation:')
            # evaluation step
            mdp.freeze_env(True)
            for idx in range(len(args.games)):
                mdp.set_episode_end(False)
                mdp.set_env(idx)
                pi.set_epsilon(epsilon_test)
                dataset = core.evaluate(n_steps=test_samples,
                                        render=args.render, quiet=args.quiet)
                scores[idx].append(get_stats(dataset, idx, args.games))

            np.save(folder_name + '/scores.npy', scores)

    return scores


if __name__ == '__main__':
    experiment()
