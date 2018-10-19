import argparse
import datetime
import pathlib
import sys

from joblib import delayed, Parallel
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import pickle

sys.path.append('..')

from mushroom.approximators.parametric import PyTorchApproximator
from mushroom.environments import *
from mushroom.utils.dataset import compute_J

from core import Core
from ddpg import DDPG
from policy import OrnsteinUhlenbeckPolicy


class ActorNetwork(nn.Module):
    def __init__(self, input_shape, _, n_actions_per_head,
                 use_cuda, dropout, features):
        super().__init__()

        self._n_input = input_shape
        self._n_games = len(n_actions_per_head)
        self._max_actions = max(n_actions_per_head)[0]
        self._use_cuda = use_cuda
        self._dropout = dropout
        self._features = features
        self._n_shared = 2

        self._h1 = nn.ModuleList(
            [nn.Linear(self._n_input[i][0], 400) for i in range(
                len(input_shape))]
        )
        self._h2 = nn.Linear(400, 300)
        self._h3 = nn.ModuleList(
            [nn.Linear(300, self._max_actions) for _ in range(
                self._n_games)]
        )

        if self._dropout:
            self._h2_dropout = nn.Dropout2d()

        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self._h2.weight)
        nn.init.uniform_(self._h2.weight, a=-1 / np.sqrt(fan_in),
                         b=1 / np.sqrt(fan_in))
        for i in range(self._n_games):
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self._h1[i].weight)
            nn.init.uniform_(self._h1[i].weight, a=-1 / np.sqrt(fan_in),
                             b=1 / np.sqrt(fan_in))
            nn.init.uniform_(self._h3[i].weight, a=-3e-3, b=3e-3)
            nn.init.uniform_(self._h3[i].bias, a=-3e-3, b=3e-3)

    def forward(self, state, idx=None, get_features=False):
        state = state.float()

        h1 = list()
        for i in np.unique(idx):
            idxs = np.argwhere(idx == i).ravel()
            h1.append(F.relu(self._h1[i](state[idxs, :self._n_input[i][0]])))
        cat_h1 = torch.cat(h1)

        if self._features == 'relu':
            h_f = F.relu(self._h2(cat_h1))
        elif self._features == 'sigmoid':
            h_f = F.sigmoid(self._h2(cat_h1))
        else:
            raise ValueError
        if self._dropout:
            h_f = self._h2_dropout(h_f)

        a = [F.tanh(self._h3[i](h_f)) for i in range(self._n_games)]
        a = torch.stack(a, dim=1)

        if idx is not None:
            idx = torch.from_numpy(idx)
            if self._use_cuda:
                idx = idx.cuda()
            a_idx = a.gather(1, idx.view(-1, 1).repeat(
                1, self._max_actions).unsqueeze(1)
                )

            a = torch.squeeze(a_idx, 1)

        if get_features:
            return a, h_f
        else:
            return a

    def get_shared_weights(self):
        p2 = list()

        for p in self._h2.parameters():
            p2.append(p.data.detach().cpu().numpy())

        return p2

    def set_shared_weights(self, weights):
        w2 = weights

        for p, w in zip(self._h2.parameters(), w2):
            w_tensor = torch.from_numpy(w).type(p.data.dtype)
            p.data = w_tensor

    def freeze_shared_weights(self):
        for p in self._h2.parameters():
            p.requires_grad = False

    def unfreeze_shared_weights(self):
        for p in self._h2.parameters():
            p.requires_grad = True


class CriticNetwork(nn.Module):
    def __init__(self, input_shape, _, n_actions_per_head, use_cuda, dropout,
                 features):
        super().__init__()

        self._n_input = input_shape
        self._n_games = len(n_actions_per_head)
        self._max_actions = max(n_actions_per_head)[0]
        self._n_actions_per_head = n_actions_per_head
        self._use_cuda = use_cuda
        self._dropout = dropout
        self._features = features
        self._n_shared = 2
        n_hidden_1 = 400
        n_hidden_2 = 300

        self._h1 = nn.ModuleList(
            [nn.Linear(self._n_input[i][0], n_hidden_1) for i in range(
                len(input_shape))]
        )
        self._h2_s = nn.Linear(n_hidden_1, n_hidden_2)
        self._h3 = nn.ModuleList(
            [nn.Linear(n_hidden_2, 1) for _ in range(
                self._n_games)]
        )
        self._h2_a = nn.ModuleList(
            [nn.Linear(n_actions_per_head[i][0], n_hidden_2, bias=False) for i in range(
                len(n_actions_per_head))]
        )

        if self._dropout:
            self._h2_dropout = nn.Dropout2d()

        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self._h2_s.weight)
        nn.init.uniform_(self._h2_s.weight, a=-1 / np.sqrt(fan_in),
                         b=1 / np.sqrt(fan_in))
        for i in range(self._n_games):
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(
                self._h2_a[i].weight)
            nn.init.uniform_(self._h2_a[i].weight, a=-1 / np.sqrt(fan_in),
                             b=1 / np.sqrt(fan_in))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self._h1[i].weight)
            nn.init.uniform_(self._h1[i].weight, a=-1 / np.sqrt(fan_in),
                             b=1 / np.sqrt(fan_in))
            nn.init.uniform_(self._h3[i].weight, a=-3e-3, b=3e-3)
            nn.init.uniform_(self._h3[i].bias, a=-3e-3, b=3e-3)

    def forward(self, state, action, idx=None, get_features=False):
        state = state.float()
        action = action.float()
        if not isinstance(idx, np.ndarray):
            idx = idx.cpu().numpy().astype(np.int)

        h2 = list()
        for i in np.unique(idx):
            idxs = np.argwhere(idx == i).ravel()
            h1 = F.relu(self._h1[i](state[idxs, :self._n_input[i][0]]))
            a = action[idxs, :self._n_actions_per_head[i][0]]
            h2.append(self._h2_s(h1) + self._h2_a[i](a))

        cat_h2 = torch.cat(h2)

        if self._features == 'relu':
            h_f = F.relu(cat_h2)
        elif self._features == 'sigmoid':
            h_f = F.sigmoid(cat_h2)
        else:
            raise ValueError

        if self._dropout:
            h_f = self._h2_dropout(h_f)

        q = [self._h3[i](h_f) for i in range(self._n_games)]
        q = torch.stack(q, dim=1).squeeze(-1)

        if idx is not None:
            idx = torch.from_numpy(idx)
            if self._use_cuda:
                idx = idx.cuda()

            q_idx = q.gather(1, idx.unsqueeze(-1))
            q = torch.squeeze(q_idx, 1)

        if get_features:
            return q, h_f
        else:
            return q

    def get_shared_weights(self):
        p2 = list()

        for p in self._h2_s.parameters():
            p2.append(p.data.detach().cpu().numpy())
        for p in self._h2_a.parameters():
            p2.append(p.data.detach().cpu().numpy())

        return p2

    def set_shared_weights(self, weights):
        w2 = weights

        for p, w in zip(self._h2_s.parameters(), w2):
            w_tensor = torch.from_numpy(w).type(p.data.dtype)
            p.data = w_tensor
        for p, w in zip(self._h2_a.parameters(), w2):
            w_tensor = torch.from_numpy(w).type(p.data.dtype)
            p.data = w_tensor

    def freeze_shared_weights(self):
        for p in self._h2_s.parameters():
            p.requires_grad = False
        for p in self._h2_a.parameters():
            p.requires_grad = False

    def unfreeze_shared_weights(self):
        for p in self._h2_s.parameters():
            p.requires_grad = True
        for p in self._h2_a.parameters():
            p.requires_grad = True


def print_epoch(epoch):
    print('################################################################')
    print('Epoch: ', epoch)
    print('----------------------------------------------------------------')


def get_stats(dataset, gamma, idx, domains, tasks):
    J = np.mean(compute_J(dataset, gamma[idx]))
    print(domains[idx] + '-' + tasks[idx] + ': J: %f' % J)

    return J


def experiment(idx):
    np.random.seed()

    # Argument parser
    parser = argparse.ArgumentParser()

    arg_game = parser.add_argument_group('Game')
    arg_game.add_argument("--games", type=list, nargs='+',
                          default=['cartpole', 'swingup'])
    arg_game.add_argument("--horizon", type=int, nargs='+')
    arg_game.add_argument("--gamma", type=float, nargs='+')

    arg_mem = parser.add_argument_group('Replay Memory')
    arg_mem.add_argument("--initial-replay-size", type=int, default=64,
                         help='Initial size of the replay memory.')
    arg_mem.add_argument("--max-replay-size", type=int, default=1000000,
                         help='Max size of the replay memory.')

    arg_net = parser.add_argument_group('Deep Q-Network')
    arg_net.add_argument("--learning-rate-actor", type=float, default=1e-4,
                         help='Learning rate value of the optimizer. Only used'
                              'in rmspropcentered')
    arg_net.add_argument("--learning-rate-critic", type=float, default=1e-3,
                         help='Learning rate value of the optimizer. Only used'
                              'in rmspropcentered')
    arg_net.add_argument("--reg-coeff", type=float, default=0)

    arg_alg = parser.add_argument_group('Algorithm')
    arg_alg.add_argument("--features", choices=['relu', 'sigmoid'])
    arg_alg.add_argument("--dropout", action='store_true')
    arg_alg.add_argument("--batch-size", type=int, default=64,
                         help='Batch size for each fit of the network.')
    arg_alg.add_argument("--tau", type=float, default=1e-3)
    arg_alg.add_argument("--history-length", type=int, default=1,
                         help='Number of frames composing a state.')
    arg_alg.add_argument("--evaluation-frequency", type=int, default=10000,
                         help='Number of learning step before each evaluation.'
                              'This number represents an epoch.')
    arg_alg.add_argument("--max-steps", type=int, default=1000000,
                         help='Total number of learning steps.')
    arg_alg.add_argument("--test-samples", type=int, default=5000,
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

    args = parser.parse_args()

    args.games = [''.join(g) for g in args.games]

    domains = args.games[::2]
    tasks = args.games[1::2]

    critic_losses = list()
    critic_l1_losses = list()
    def critic_loss(arg, y):
        yhat, h_f = arg

        loss = F.mse_loss(yhat, y, reduce=False)
        temp_losses = list()
        for i in range(len(domains)):
            start = i * args.batch_size
            stop = start + args.batch_size
            temp_losses.append(torch.mean(loss[start:stop]).item())
        critic_losses.append(temp_losses)
        loss = torch.mean(loss)
        l1_loss = torch.norm(h_f, 1) / h_f.shape[0]
        critic_l1_losses.append(l1_loss.item())

        return loss + args.reg_coeff * l1_loss

    scores = list()
    for _ in range(len(domains)):
        scores.append(list())

    optimizer_actor = dict()
    optimizer_actor['class'] = optim.Adam
    optimizer_actor['params'] = dict(lr=args.learning_rate_actor)

    optimizer_critic = dict()
    optimizer_critic['class'] = optim.Adam
    optimizer_critic['params'] = dict(lr=args.learning_rate_critic,
                                      weight_decay=1e-2)

    # MDP
    mdp = list()
    gamma_eval = list()
    for i, g in enumerate(zip(domains, tasks)):
        mdp.append(Mujoco(g[0], g[1], args.horizon[i], args.gamma[i]))
        gamma_eval.append(args.gamma[i])

    n_input_per_mdp = [m.info.observation_space.shape for m in mdp]
    n_actions_per_head = [(m.info.action_space.shape[0],) for m in mdp]

    max_obs_dim = 0
    max_act_n = 0
    for i in range(len(domains)):
        n = mdp[i].info.observation_space.shape[0]
        m = len(mdp[i].info.action_space.shape)
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
    max_action_value = list()
    for m in mdp:
        assert len(np.unique(m.info.action_space.low)) == 1
        assert len(np.unique(m.info.action_space.high)) == 1
        assert abs(m.info.action_space.low[0]) == m.info.action_space.high[0]

        max_action_value.append(m.info.action_space.high[0])

    # DQN learning run

    # Settings
    if args.debug:
        initial_replay_size = args.batch_size
        max_replay_size = 500
        test_samples = 20
        evaluation_frequency = 50
        max_steps = 1000
    else:
        initial_replay_size = args.initial_replay_size
        max_replay_size = args.max_replay_size
        test_samples = args.test_samples
        evaluation_frequency = args.evaluation_frequency
        max_steps = args.max_steps

    # Policy
    policy_class = OrnsteinUhlenbeckPolicy
    policy_params = dict(sigma=np.ones(1) * .2, theta=.15, dt=1e-2,
                         n_actions_per_head=n_actions_per_head,
                         max_action_value=max_action_value)

    # Approximator
    actor_approximator = PyTorchApproximator
    actor_input_shape = [m.info.observation_space.shape for m in mdp]

    actor_approximator_params = dict(
        network=ActorNetwork,
        input_shape=actor_input_shape,
        output_shape=(max(n_actions_per_head)[0],),
        n_actions_per_head=n_actions_per_head,
        optimizer=optimizer_actor,
        use_cuda=args.use_cuda,
        dropout=args.dropout,
        features=args.features
    )

    critic_approximator = PyTorchApproximator
    critic_input_shape = [m.info.observation_space.shape for m in mdp]
    critic_approximator_params = dict(
        network=CriticNetwork,
        input_shape=critic_input_shape,
        output_shape=(1,),
        n_actions_per_head=n_actions_per_head,
        optimizer=optimizer_actor,
        loss=critic_loss,
        use_cuda=args.use_cuda,
        dropout=args.dropout,
        features=args.features
    )

    # Agent
    algorithm_params = dict(
        batch_size=args.batch_size,
        initial_replay_size=initial_replay_size,
        max_replay_size=max_replay_size,
        tau=args.tau,
        actor_params=actor_approximator_params,
        critic_params=critic_approximator_params,
        policy_params=policy_params,
        n_games=len(domains),
        n_input_per_mdp=n_input_per_mdp,
        n_actions_per_head=n_actions_per_head,
        dtype=np.float32
    )

    agent = DDPG(actor_approximator, critic_approximator, policy_class,
                 mdp_info, **algorithm_params)

    # Algorithm
    core = Core(agent, mdp)

    # RUN

    # Fill replay memory with random dataset
    print_epoch(0)
    core.learn(n_steps=initial_replay_size,
               n_steps_per_fit=initial_replay_size, quiet=args.quiet)

    if args.transfer:
        weights = pickle.load(open(args.transfer, 'rb'))
        agent.set_shared_weights(weights)

    if args.load:
        weights = np.load(args.load)
        agent.policy.set_weights(weights)

    # Evaluate initial policy
    dataset = core.evaluate(n_steps=test_samples, render=args.render,
                            quiet=args.quiet)
    for i in range(len(mdp)):
        d = dataset[i::len(mdp)]
        scores[i].append(get_stats(d, gamma_eval, i, domains, tasks))

    if args.unfreeze_epoch > 0:
        agent.freeze_shared_weights()

    best_score_sum = -np.inf
    best_weights = None

    np.save(folder_name + 'scores-exp-%d.npy' % idx, scores)
    np.save(folder_name + 'critic_loss-exp-%d.npy' % idx, critic_losses)
    np.save(folder_name + 'critic_l1_loss-exp-%d.npy' % idx, critic_l1_losses)
    np.save(folder_name + 'q-exp-%d.npy' % idx, agent.q_list)
    for n_epoch in range(1, max_steps // evaluation_frequency + 1):
        if n_epoch >= args.unfreeze_epoch > 0:
            agent.unfreeze_shared_weights()

        print_epoch(n_epoch)
        print('- Learning:')
        # learning step
        core.learn(n_steps=evaluation_frequency,
                   n_steps_per_fit=1, quiet=args.quiet)

        print('- Evaluation:')
        # evaluation step
        dataset = core.evaluate(n_steps=test_samples,
                                render=args.render, quiet=args.quiet)

        current_score_sum = 0
        for i in range(len(mdp)):
            d = dataset[i::len(mdp)]
            current_score = get_stats(d, gamma_eval, i, domains, tasks)
            scores[i].append(current_score)
            current_score_sum += current_score

        # Save shared weights if best score
        if args.save_shared and current_score_sum >= best_score_sum:
            best_score_sum = current_score_sum
            best_weights = agent.get_shared_weights()

        if args.save:
            np.save(folder_name + 'best_weights-exp-%d.npy' % idx, agent.policy.get_weights())

        np.save(folder_name + 'scores-exp-%d.npy' % idx, scores)
        np.save(folder_name + 'critic_loss-exp-%d.npy' % idx, critic_losses)
        np.save(folder_name + 'critic_l1_loss-exp-%d.npy' % idx, critic_l1_losses)
        np.save(folder_name + 'q-exp-%d.npy' % idx, agent.q_list)

    if args.save_shared:
        pickle.dump(best_weights, open(args.save_shared, 'wb'))

    return scores, critic_losses, critic_l1_losses, agent.q_list


if __name__ == '__main__':
    n_experiments = 1

    folder_name = './logs/gym_' + datetime.datetime.now().strftime(
        '%Y-%m-%d_%H-%M-%S/')
    pathlib.Path(folder_name).mkdir(parents=True)

    out = Parallel(n_jobs=-1)(delayed(experiment)(i) for i in range(n_experiments))

    scores = np.array([o[0] for o in out])
    critic_loss = np.array([o[1] for o in out])
    critic_l1_loss = np.array([o[2] for o in out])
    qs = np.array([o[3] for o in out])

    np.save(folder_name + 'scores.npy', scores)
    np.save(folder_name + 'critic_loss.npy', critic_loss)
    np.save(folder_name + 'critic_l1_loss.npy', critic_l1_loss)
    np.save(folder_name + 'qs.npy', qs)
