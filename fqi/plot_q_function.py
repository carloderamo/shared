import pickle
import numpy as np
import matplotlib.pyplot as plt

from mushroom.approximators import Regressor
from mushroom.approximators.parametric import PyTorchApproximator
from mushroom.environments import *
from networks import GymNetwork


def v_plot(approximator, game_idx, observation_space, ax, n_actions,
           contours=False, n=25):
    x = np.linspace(observation_space.low[0], observation_space.high[0], n)
    y = np.linspace(observation_space.low[1], observation_space.high[1], n)
    xv, yv = np.meshgrid(x, y)

    inputs = list()
    for i, j in zip(xv.flatten(), yv.flatten()):
        inputs.append(np.array([i, j]))

    inputs = np.array(inputs)
    outputs = approximator.predict(inputs, idx=np.ones(len(inputs),
                                                       dtype=np.int)*game_idx)
    outputs = outputs[:, :n_actions].max(1).reshape(xv.shape)

    if contours:
        ax.contour(xv, yv, outputs)
    else:
        ax.plot_surface(xv, yv, outputs)


folder_name = 'folder/'
#game_idx = 0 # single
game_idx = 2 # multi

args = pickle.load(open(folder_name + 'args.pkl'))

args.games = [''.join(g) for g in args.games]

# MDP
mdp = list()
gamma_eval = list()
for i, g in enumerate(args.games):
    if g == 'pendulum':
        mdp.append(InvertedPendulumDiscrete(horizon=args.horizon[i],
                                            gamma=args.gamma[i]))
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

assert args.reg_type != 'kl' or args.features == 'sigmoid'

scores = list()
for _ in range(len(args.games)):
    scores.append(list())

optimizer = dict()

# FQI learning run

# Settings
if args.debug:
    test_samples = 20
    max_steps = 1000
else:
    test_samples = args.test_samples
    max_steps = args.max_steps

# Approximator
input_shape = [m.info.observation_space.shape for m in mdp]
n_games = len(args.games)

approximator_params = dict(
    network=GymNetwork,
    input_shape=input_shape,
    output_shape=(max(n_actions_per_head)[0],),
    n_actions=max(n_actions_per_head)[0],
    n_actions_per_head=n_actions_per_head,
    optimizer=optimizer,
    loss=None,
    use_cuda=args.use_cuda,
    dropout=args.dropout,
    features=args.features,
    quiet=False,
    batch_size=args.batch_size * n_games
)


q_funct = Regressor(PyTorchApproximator, **approximator_params)

step = 5
max_step = 20

n_subplot = max_step / step

fig, ax = plt.subplots(1, n_subplot)

for i in range(0, max_step, step):
    weights = np.load(folder_name + 'weights-exp-0-%d.npy' % i)
    q_funct.set_weights(weights)

    v_plot(q_funct, game_idx,
           mdp[game_idx].info.observation_space, ax[i],
           n_actions_per_head[game_idx],
           contours=True)

