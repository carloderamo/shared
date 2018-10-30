import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from networks import GymNetwork

from mushroom.approximators.regressor import Regressor
from mushroom.approximators.parametric import PyTorchApproximator
from mushroom.utils.spaces import Box
from mushroom.utils.dataset import parse_dataset
from mushroom.environments import *


def load_mdps():
    mdp = list()
    horizon = 1000
    gamma = 0.99
    for i, g in enumerate(games):
        if g == 'pendulum':
            mdp.append(InvertedPendulumDiscrete(horizon=horizon,
                                                gamma=gamma))
        elif g == 'caronhill':
            mdp.append(CarOnHill(horizon=horizon, gamma=gamma))
        else:
            mdp.append(Gym(g, horizon, gamma))

    return mdp


def annotate(label, *axis):
    for ax in axis:
        ax.annotate(label,
                    xy=(0, 0.5),
                    xytext=(0, 0.5),
                    xycoords=ax.yaxis.label,
                    textcoords='axes fraction',
                    size='small',
                    ha='right',
                    va='center')


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


def max_phi_plot(approximator, game_idx, observation_space, ax, n=25):
    x = np.linspace(observation_space.low[0], observation_space.high[0], n)
    y = np.linspace(observation_space.low[1], observation_space.high[1], n)
    xv, yv = np.meshgrid(x, y)

    inputs = list()
    for i, j in zip(xv.flatten(), yv.flatten()):
        inputs.append(np.array([i, j]))

    inputs = np.array(inputs)
    _, outputs = approximator.predict(inputs, get_features=True,
                                      idx=np.ones(len(inputs),
                                                  dtype=np.int) * game_idx)
    outputs = np.argmax(outputs, axis=1).reshape(xv.shape)

    cmap = plt.get_cmap('gist_yarg', 80)
    ax.matshow(outputs, cmap=cmap, vmin=0, vmax=80)


def n_phi_plot(approximator, game_idx, observation_space, ax, t=0.5, n=25):
    x = np.linspace(observation_space.low[0], observation_space.high[0], n)
    y = np.linspace(observation_space.low[1], observation_space.high[1], n)
    xv, yv = np.meshgrid(x, y)

    inputs = list()
    for i, j in zip(xv.flatten(), yv.flatten()):
        inputs.append(np.array([i, j]))

    inputs = np.array(inputs)
    _, outputs = approximator.predict(inputs, get_features=True,
                                      idx=np.ones(len(inputs),
                                                  dtype=np.int) * game_idx)

    outputs = np.sum(outputs > t, 1).reshape(xv.shape)

    #ax.contourf(xv, yv, outputs)
    cmap = plt.get_cmap('gist_yarg', 80)
    ax.matshow(outputs, cmap=cmap, vmin=0, vmax=80)


def plot_specific_features(approximator, game_idx, inputs, ax):
    _, outputs = approximator.predict(inputs, get_features=True,
                                      idx=np.ones(len(inputs),
                                                  dtype=np.int) * game_idx)
    # ax.contourf(xv, yv, outputs)
    cmap = plt.get_cmap('gist_yarg', 1000)
    ax.matshow(outputs.T, cmap=cmap, vmin=0, vmax=1)


def feature_sum(approximator, game_idx, observation_space, ax, t=0.5, n=25):
    x = np.linspace(observation_space.low[0], observation_space.high[0], n)
    y = np.linspace(observation_space.low[1], observation_space.high[1], n)
    xv, yv = np.meshgrid(x, y)

    inputs = list()
    for i, j in zip(xv.flatten(), yv.flatten()):
        inputs.append(np.array([i, j]))

    inputs = np.array(inputs)
    _, outputs = approximator.predict(inputs, get_features=True,
                                      idx=np.ones(len(inputs),
                                                  dtype=np.int) * game_idx)

    outputs = np.sum(outputs, 1).reshape(xv.shape)

    #ax.contourf(xv, yv, outputs)
    cmap = plt.get_cmap('gist_yarg', 80)
    ax.matshow(outputs, cmap=cmap, vmin=0, vmax=80)


# Parameters
alg = 'multidqn'

reg = ['noreg', 'l1', 'kl']
#reg = ['kl']
activation = ['relu', 'sigmoid']
#activation = ['sigmoid']

games = ['CartPole-v0', 'Acrobot-v1', 'MountainCar-v0', 'caronhill', 'pendulum']
games_labels = ['cart', 'acro', 'mc', 'coh', 'pend']
game_idx = 2

nets = np.array([15, 30, 40, 50])

file_prefix = 'weights-exp-0-'

observation_space = Box(np.array([-1.2, -0.07]), np.array([0.6, 0.07]))

# Create MDP to get parameters
mdps = load_mdps()
n_input_per_mdp = [m.info.observation_space.shape for m in mdps]
n_actions_per_head = [(m.info.action_space.n,) for m in mdps]
input_shape = [m.info.observation_space.shape for m in mdps]

# Create subplots
n_rows = len(reg) * len(activation)
n_cols = len(nets)

fig = plt.figure()
fig.suptitle(games_labels[game_idx])
ax_3d = fig.subplots(n_rows, n_cols, subplot_kw=dict(projection='3d'))
ax_3d = np.atleast_2d(ax_3d)

fig = plt.figure()
fig.suptitle(games_labels[game_idx])
ax_c = fig.subplots(n_rows, n_cols)
ax_c = np.atleast_2d(ax_c)

fig = plt.figure()
fig.suptitle(games_labels[game_idx] + '  max feature index')
ax_f = fig.subplots(n_rows, n_cols)
ax_f = np.atleast_2d(ax_f)

fig = plt.figure()
fig.suptitle(games_labels[game_idx] + '  n features > 0.5')
ax_nf = fig.subplots(n_rows, n_cols)
ax_nf = np.atleast_2d(ax_nf)

fig = plt.figure()
fig.suptitle(games_labels[game_idx] + '  feature sum')
ax_sf = fig.subplots(n_rows, n_cols)
ax_sf = np.atleast_2d(ax_sf)

fig = plt.figure()
fig.suptitle(games_labels[game_idx] + '  traj features')
ax_t = fig.subplots(len(reg), n_cols)
ax_t = np.atleast_2d(ax_t)

for i in range(n_cols):
    ax_3d[0, i].set_title(str(nets[i]))
    ax_c[0, i].set_title(str(nets[i]))
    ax_f[0, i].set_title(str(nets[i]))
    ax_nf[0, i].set_title(str(nets[i]))
    ax_t[0, i].set_title(str(nets[i]))

# Plot every value function
base_path = '../results/dqn/' + alg + '/'


# Load Trajectories
traj = np.load(base_path + '../' + games_labels[game_idx] + '_traj.npy')
states = list()

for step in traj:
    states.append(step[0][1])

states = np.array(states)

k = 0
for act in activation:
    k2 = 0
    for r in reg:
        conf = r + '-' + act
        path = base_path + conf + '/nets/' + file_prefix

        # Create network
        approximator_params = dict(
                network=GymNetwork,
                input_shape=input_shape,
                output_shape=(max(n_actions_per_head)[0],),
                n_actions=max(n_actions_per_head)[0],
                n_actions_per_head=n_actions_per_head,
                optimizer=None,
                loss=None,
                use_cuda=False,
                dropout=False,
                features=act
            )

        approximator = Regressor(PyTorchApproximator, **approximator_params)

        # Plot value functions
        for i, j in enumerate(nets):
            file_name = path + str(j) + '.npy'
            weights = np.load(file_name)
            approximator.set_weights(weights)
            v_plot(approximator, game_idx, observation_space, ax_3d[k, i],
                   n_actions_per_head[game_idx][0])
            v_plot(approximator, game_idx, observation_space, ax_c[k, i],
                   n_actions_per_head[game_idx][0], True, 1000)

            max_phi_plot(approximator, game_idx, observation_space, ax_f[k, i])

            n_phi_plot(approximator, game_idx, observation_space, ax_nf[k, i],
                       0.5)
                       
            feature_sum(approximator, game_idx, observation_space, ax_sf[k, i],
                       0.5)

            if act is 'sigmoid':
                plot_specific_features(approximator, game_idx, states,
                                       ax_t[k2, i])

        if act is 'sigmoid':
            annotate(r, ax_t[k2, 0])

        k2 += 1

        annotate(conf, ax_3d[k, 0], ax_c[k, 0], ax_f[k, 0], ax_nf[k, 0], ax_sf[k, 0])


        k += 1

plt.show()
