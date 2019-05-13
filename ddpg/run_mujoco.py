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

from networks import ActorNetwork, CriticNetwork
from losses import *


def print_epoch(epoch):
    print('################################################################')
    print('Epoch: ', epoch)
    print('----------------------------------------------------------------')


def get_stats(dataset, gamma, idx, domains, tasks):
    J = np.mean(compute_J(dataset, gamma[idx]))
    print(domains[idx] + '-' + tasks[idx] + ': J: %f' % J)

    return J


def experiment(idx, args):
    np.random.seed()

    args.games = [''.join(g) for g in args.games]

    domains = args.games[::2]
    tasks = args.games[1::2]

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
    n_games = len(args.games)
    if args.reg_type == 'l1':
        loss = FeaturesL1Loss(args.reg_coeff, n_games,
                              args.batch_size, args.evaluation_frequency)
    elif args.reg_type == 'l1-weights':
        loss = WeightsL1Loss(n_actions_per_head, args.reg_coeff, n_games,
                             args.batch_size, args.evaluation_frequency)
    elif args.reg_type == 'gl1-weights':
        loss = WeightsGLLoss(n_actions_per_head, args.reg_coeff, n_games,
                             args.batch_size, args.evaluation_frequency)
    elif args.reg_type == 'kl':
        loss = FeaturesKLLoss(args.k, args.reg_coeff, n_games,
                              args.batch_size, args.evaluation_frequency)
    else:
        loss = LossFunction(args.reg_coeff, n_games, args.batch_size,
                            args.evaluation_frequency)

    actor_approximator = PyTorchApproximator
    actor_input_shape = [m.info.observation_space.shape for m in mdp]

    actor_approximator_params = dict(
        network=ActorNetwork,
        input_shape=actor_input_shape,
        output_shape=(max(n_actions_per_head)[0],),
        n_actions_per_head=n_actions_per_head,
        n_hidden_1=args.hidden_neurons[0],
        n_hidden_2=args.hidden_neurons[1],
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
        n_hidden_1=args.hidden_neurons[0],
        n_hidden_2=args.hidden_neurons[1],
        optimizer=optimizer_actor,
        loss=loss,
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
    agent.policy.eval = True
    dataset = core.evaluate(n_steps=test_samples, render=args.render,
                            quiet=args.quiet)
    agent.policy.eval = False
    for i in range(len(mdp)):
        d = dataset[i::len(mdp)]
        scores[i].append(get_stats(d, gamma_eval, i, domains, tasks))

    if args.unfreeze_epoch > 0:
        agent.freeze_shared_weights()

    best_score_sum = -np.inf
    best_weights = None

    np.save(folder_name + 'scores-exp-%d.npy' % idx, scores)
    np.save(folder_name + 'critic_loss-exp-%d.npy' % idx,
            agent._critic_approximator.model._loss.get_losses())
    np.save(folder_name + 'critic_l1_loss-exp-%d.npy' % idx,
            agent._critic_approximator.model._loss.get_reg_losses())
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
        agent.policy.eval = True
        dataset = core.evaluate(n_steps=test_samples,
                                render=args.render, quiet=args.quiet)
        agent.policy.eval = False

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
            np.save(folder_name + 'best_weights-exp-%d.npy' % idx,
                    agent.policy.get_weights())

        np.save(folder_name + 'scores-exp-%d.npy' % idx, scores)
        np.save(folder_name + 'critic_loss-exp-%d.npy' % idx,
                agent._critic_approximator.model._loss.get_losses())
        np.save(folder_name + 'critic_l1_loss-exp-%d.npy' % idx,
                agent._critic_approximator.model._loss.get_reg_losses())
        np.save(folder_name + 'q-exp-%d.npy' % idx, agent.q_list)

    if args.save_shared:
        pickle.dump(best_weights, open(args.save_shared, 'wb'))

    return scores, agent._critic_approximator.model._loss.get_losses(), \
           agent._critic_approximator.model._loss.get_reg_losses(), agent.q_list


if __name__ == '__main__':
    # Argument parser
    parser = argparse.ArgumentParser()

    arg_game = parser.add_argument_group('Game')
    arg_game.add_argument("--games", type=list, nargs='+',
                          default=['cartpole', 'swingup'])
    arg_game.add_argument("--horizon", type=int, nargs='+')
    arg_game.add_argument("--gamma", type=float, nargs='+')
    arg_game.add_argument("--n-exp", type=int)

    arg_mem = parser.add_argument_group('Replay Memory')
    arg_mem.add_argument("--initial-replay-size", type=int, default=64,
                         help='Initial size of the replay memory.')
    arg_mem.add_argument("--max-replay-size", type=int, default=1000000,
                         help='Max size of the replay memory.')

    arg_net = parser.add_argument_group('Deep Q-Network')
    arg_net.add_argument("--hidden-neurons", type=int, nargs=2,
                         default=[600, 500])
    arg_net.add_argument("--learning-rate-actor", type=float, default=1e-4,
                         help='Learning rate value of the optimizer. Only used'
                              'in rmspropcentered')
    arg_net.add_argument("--learning-rate-critic", type=float, default=1e-3,
                         help='Learning rate value of the optimizer. Only used'
                              'in rmspropcentered')
    arg_net.add_argument("--reg-coeff", type=float, default=0)
    arg_net.add_argument("--reg-type", type=str,
                         choices=['l1', 'l1-weights', 'gl1-weights', 'kl'])
    arg_net.add_argument("--k", type=float, default=10)

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
    arg_utils.add_argument('--postfix', type=str, default='',
                           help='Flag used to add a postfix to the folder name')

    args = parser.parse_args()

    folder_name = './logs/mujoco_' + datetime.datetime.now().strftime(
        '%Y-%m-%d_%H-%M-%S') + args.postfix + '/'
    pathlib.Path(folder_name).mkdir(parents=True)

    with open(folder_name + 'args.pkl', 'wb') as f:
        pickle.dump(args, f)

    out = Parallel(n_jobs=-1)(delayed(experiment)(i, args)
                              for i in range(args.n_exp))

    scores = np.array([o[0] for o in out])
    critic_loss = np.array([o[1] for o in out])
    critic_l1_loss = np.array([o[2] for o in out])
    qs = np.array([o[3] for o in out])

    np.save(folder_name + 'scores.npy', scores)
    np.save(folder_name + 'critic_loss_raw.npy', critic_loss)
    np.save(folder_name + 'critic_l1_loss_raw.npy', critic_l1_loss)
    np.save(folder_name + 'qs_raw.npy', qs)
