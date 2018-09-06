import argparse
import datetime
import pathlib
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import pickle

sys.path.append('..')

from mushroom.approximators.parametric import PyTorchApproximator
from mushroom.environments import *
from mushroom.utils.dataset import compute_scores
from mushroom.utils.parameters import LinearDecayParameter, Parameter

from core import Core
from shared.dqn import DQN, DoubleDQN
from policy import EpsGreedyMultiple

"""
This script runs Atari experiments with DQN as presented in:
"Human-Level Control Through Deep Reinforcement Learning". Mnih V. et al.. 2015.

"""


class Network(nn.Module):
    def __init__(self, input_shape, _, n_actions_per_head, use_cuda, dropout,
                 features):
        super(Network, self).__init__()

        n_input = input_shape[0]
        self._n_games = len(n_actions_per_head)
        self._max_actions = max(n_actions_per_head)[0]
        self._use_cuda = use_cuda
        self._dropout = dropout
        self._n_shared = 6
        self._features = features

        self._h1 = nn.ModuleList([nn.Conv2d(
            n_input, 32, kernel_size=8, stride=4) for _ in range(self._n_games)]
        )
        self._h2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self._h3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self._h4 = nn.Linear(3136, 512)
        self._h5 = nn.ModuleList(
            [nn.Linear(512, self._max_actions) for _ in range(
                self._n_games)])

        if self._dropout:
            raise NotImplementedError

        nn.init.xavier_uniform_(self._h2.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h3.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h4.weight,
                                gain=nn.init.calculate_gain('relu'))
        for i in range(self._n_games):
            nn.init.xavier_uniform_(self._h1[i].weight,
                                    gain=nn.init.calculate_gain('relu'))
            nn.init.xavier_uniform_(self._h5[i].weight,
                                    gain=nn.init.calculate_gain('linear'))

    def forward(self, state, action=None, idx=None, get_features=False):
        state = state.float() / 255.

        h1 = list()
        for i in np.unique(idx):
            idxs = np.argwhere(idx == i).ravel()
            h1.append(F.relu(self._h1[i](state[idxs])))
        cat_h1 = torch.cat(h1)

        h = F.relu(self._h2(cat_h1))
        h = F.relu(self._h3(h))
        h_f = h.view(-1, 3136)

        if self._features == 'relu':
            h_f = F.relu(self._h4(h_f))
        elif self._features == 'sigmoid':
            h_f = F.sigmoid(self._h4(h_f))
        else:
            raise ValueError

        q = [self._h5[i](h_f) for i in range(self._n_games)]
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

    def get_shared_weights(self):
        p2 = list()
        p3 = list()
        p4 = list()

        for p in self._h2.parameters():
            p2.append(p.data.detach().cpu().numpy())

        for p in self._h3.parameters():
            p3.append(p.data.detach().cpu().numpy())

        for p in self._h4.parameters():
            p4.append(p.data.detach().cpu().numpy())

        return p2, p3, p4

    def set_shared_weights(self, weights):
        w2, w3, w4 = weights

        for p, w in zip(self._h2.parameters(), w2):
            w_tensor = torch.from_numpy(w).type(p.data.dtype)
            p.data = w_tensor

        for p, w in zip(self._h3.parameters(), w3):
            w_tensor = torch.from_numpy(w).type(p.data.dtype)
            p.data = w_tensor

        for p, w in zip(self._h4.parameters(), w4):
            w_tensor = torch.from_numpy(w).type(p.data.dtype)
            p.data = w_tensor

    def freeze_shared_weights(self):
        for p in self._h2.parameters():
            p.requires_grad = False
        for p in self._h3.parameters():
            p.requires_grad = False
        for p in self._h4.parameters():
            p.requires_grad = False

    def unfreeze_shared_weights(self):
        for p in self._h2.parameters():
            p.requires_grad = True
        for p in self._h3.parameters():
            p.requires_grad = True
        for p in self._h4.parameters():
            p.requires_grad = True


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
    arg_net.add_argument("--reg-coeff", type=float, default=1e-4.)

    arg_alg = parser.add_argument_group('Algorithm')
    arg_alg.add_argument("--algorithm", default='dqn', choices=['dqn', 'ddqn'])
    arg_alg.add_argument("--features", choices=['relu', 'sigmoid'],
                         default='sigmoid')
    arg_alg.add_argument("--dropout", action='store_true')
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

    losses = list()
    l1_losses = list()
    def regularized_loss(arg, y):
        yhat, h_f = arg

        loss = F.smooth_l1_loss(yhat, y, reduce=False)
        temp_losses = list()
        for i in range(len(args.games)):
            start = i * args.batch_size
            stop = start + args.batch_size
            temp_losses.append(torch.mean(loss[start:stop]).item())
        losses.append(temp_losses)
        loss = torch.mean(loss)
        l1_loss = torch.norm(h_f, 1) / h_f.shape[0]
        l1_losses.append(l1_loss.item())

        return loss + args.reg_coeff * l1_loss

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

    # MDP
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
    gammas = [m.info.gamma for m in mdp]
    horizons = [m.info.horizon for m in mdp]
    mdp_info = MDPInfo(mdp[max_obs_idx].info.observation_space,
                       mdp[max_act_idx].info.action_space, gammas, horizons)

    # Evaluation of the model provided by the user.
    if args.load_path:
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
            output_shape=(max(n_actions_per_head)[0],),
            n_actions=max(n_actions_per_head)[0],
            n_actions_per_head=n_actions_per_head,
            load_path=args.load_path,
            optimizer=optimizer,
            loss=regularized_loss,
            use_cuda=args.use_cuda,
            dropout=args.dropout,
            features=args.features
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
            dtype=np.uint8
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
        pi = EpsGreedyMultiple(epsilon=epsilon,
                               n_actions_per_head=n_actions_per_head)

        # Approximator
        input_shape = (args.history_length, args.screen_height,
                       args.screen_width)
        approximator_params = dict(
            network=Network,
            input_shape=input_shape,
            output_shape=(max(n_actions_per_head)[0],),
            n_actions=max(n_actions_per_head)[0],
            n_actions_per_head=n_actions_per_head,
            optimizer=optimizer,
            loss=regularized_loss,
            use_cuda=args.use_cuda,
            dropout=args.dropout,
            features=args.features
        )

        approximator = PyTorchApproximator

        # Agent
        algorithm_params = dict(
            batch_size=args.batch_size,
            n_games=len(args.games),
            initial_replay_size=initial_replay_size,
            max_replay_size=max_replay_size,
            target_update_frequency=target_update_frequency // train_frequency,
            n_actions_per_head=n_actions_per_head,
            history_length=args.history_length,
            dtype=np.uint8
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
        pi.set_epsilon(epsilon_random)
        core.learn(n_steps=initial_replay_size,
                   n_steps_per_fit=initial_replay_size, quiet=args.quiet)

        if args.save:
            agent.approximator.model.save()

        if args.transfer:
            weights = pickle.load(open(args.transfer, 'rb'))
            agent.set_shared_weights(weights)

        for m in mdp:
            m.set_episode_end(False)
        # Evaluate initial policy
        pi.set_epsilon(epsilon_test)
        dataset = core.evaluate(n_steps=test_samples, render=args.render,
                                quiet=args.quiet)
        for i in range(len(mdp)):
            d = dataset[i::len(mdp)]
            scores[i].append(get_stats(d, i, args.games))

        if args.unfreeze_epoch > 0:
            agent.freeze_shared_weights()

        best_score_sum = -np.inf
        best_weights = None

        np.save(folder_name + '/scores.npy', scores)
        for n_epoch in range(1, max_steps // evaluation_frequency + 1):
            if n_epoch >= args.unfreeze_epoch > 0:
                agent.unfreeze_shared_weights()

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

            current_score_sum = 0
            for i in range(len(mdp)):
                d = dataset[i::len(mdp)]
                current_score = get_stats(d, i, args.games)
                scores[i].append(current_score)
                current_score_sum += current_score[2]

            # Save shared weights if best score
            if args.save_shared and current_score_sum >= best_score_sum:
                best_score_sum = current_score_sum
                best_weights = agent.get_shared_weights()

            np.save(folder_name + '/scores.npy', scores)

    if args.save_shared:
        pickle.dump(best_weights, open(args.save_shared, 'wb'))

    return scores


if __name__ == '__main__':
    experiment()
