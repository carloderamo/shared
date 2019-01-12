import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from mushroom.approximators import Regressor
from mushroom.approximators.parametric import PyTorchApproximator
from mushroom.environments import *
from networks import GymNetwork
from pendulum import *


def v_plot(approximator, game_idx, observation_space, ax, n_actions,
           train_dataset, target,
           min_t, delta_t,
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

    outputs = outputs*delta_t+min_t

    if contours:
        ax.contour(xv, yv, outputs)
    else:
        ax.plot_surface(xv, yv, outputs)

    x = list()
    y = list()
    z = list()
    for i, step in enumerate(train_dataset):
        x_i = step[0][1][0]
        y_i = step[0][1][1]
        z_i = target[i]*delta_t+min_t
        x.append(x_i)
        y.append(y_i)
        z.append(z_i)

    ax.scatter(x, y, z, c='g')

def max_plot(approximator, game_idx, observation_space, ax, n_actions, n=25):
    x = np.linspace(observation_space.low[0], observation_space.high[0], n)
    y = np.linspace(observation_space.high[1], observation_space.low[1], n)
    xv, yv = np.meshgrid(x, y)

    inputs = list()
    for i, j in zip(xv.flatten(), yv.flatten()):
        inputs.append(np.array([i, j]))

    inputs = np.array(inputs)
    outputs = approximator.predict(inputs, idx=np.ones(len(inputs),
                                                       dtype=np.int)*game_idx)
    outputs = outputs[:, :n_actions].argmax(1).reshape(xv.shape)

    cmap = plt.get_cmap('gist_yarg', n_actions)
    ax.matshow(outputs, cmap=cmap, vmin=0, vmax=n_actions)



############################################################### PLOT PARAMETERS

plot_action_indx = True
plot_action_indx = False

step = 1
first_step = 0
max_step = 1

surface = True
#surface = False


single = False
#single = True

eps_dataset = '0.2'
game_idx = 1

############################################################### PLOT PARAMETERS

if single:
    fig_title = 'single'
    folder_name = 'logs/dondolino_single/'
    game_idx = 0
else:
    fig_title = 'multi'
    folder_name = 'logs/dondolino_multi/'

args = pickle.load(open(folder_name + 'args.pkl', 'rb'))


train_dataset = pickle.load(open(folder_name + 'dataset.pkl', 'rb'))

# MDP
mdp = list()
gamma_eval = list()
for m in args.pendulum_mass:
    mdp.append(InvertedPendulumDiscreteV2(m=m))

    gamma_eval.append(mdp[-1].info.gamma)

n_input_per_mdp = [m.info.observation_space.shape for m in mdp]
n_actions_per_head = [(m.info.action_space.n,) for m in mdp]

max_obs_dim = 0
max_act_n = 0
for i in range(len(args.pendulum_mass)):
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
for _ in range(len(args.pendulum_mass)):
    scores.append(list())

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
n_games = len(args.pendulum_mass)

approximator_params = dict(
    network=GymNetwork,
    input_shape=input_shape,
    output_shape=(max(n_actions_per_head)[0],),
    n_actions=max(n_actions_per_head)[0],
    n_actions_per_head=n_actions_per_head,
    use_cuda=False,
    dropout=args.dropout,
    features=args.features,
    quiet=False,
    batch_size=args.batch_size * n_games
)


q_funct = Regressor(PyTorchApproximator, **approximator_params)

n_subplot = (max_step - first_step) // step

if surface and not plot_action_indx:
    fig, ax = plt.subplots(1, n_subplot, subplot_kw=dict(projection='3d'))
else:
    fig, ax = plt.subplots(1, n_subplot)

ax = np.atleast_1d(ax)
fig.suptitle(fig_title)

for i in range(n_subplot):
    epoch = first_step+i*step
    weights = np.load(folder_name + 'weights-exp-0-%d.npy' % epoch)
    target = np.load(folder_name + 'targets-exp-0-%d.npy' % epoch)
    min_t = np.load(folder_name + 'min-exp-0-%d.npy' % epoch)[game_idx]
    delta_t = np.load(folder_name + 'delta-exp-0-%d.npy' % epoch)[game_idx]
    q_funct.set_weights(weights)

    print('min %f, delta %f, max %f' % (min_t, delta_t, min_t+delta_t))

    ax[i].set_title(epoch)

    if plot_action_indx:
        max_plot(q_funct, game_idx,
                 mdp[game_idx].info.observation_space, ax[i],
                 n_actions_per_head[game_idx][0])
    else:
        v_plot(q_funct, game_idx,
               mdp[game_idx].info.observation_space, ax[i],
               n_actions_per_head[game_idx][0],
               train_dataset, target,
               min_t, delta_t,
               contours=not surface)

        if surface:
            z_lim = [min_t, min_t+delta_t]
            ax[i].axes.set_zlim3d(z_lim)

plt.show()