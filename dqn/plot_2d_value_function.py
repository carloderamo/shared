import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from networks import GymNetwork

from mushroom.approximators.regressor import Regressor
from mushroom.utils.spaces import Box
from mushroom.environments import *


def v_plot(approximator, game_idx, observation_space, ax, n_actions, n=150):
    x = np.linspace(observation_space.low[0], observation_space.high[0], n)
    y = np.linspace(observation_space.low[1], observation_space.high[1], n)
    xv, yv = np.meshgrid(x, y)

    inputs = list()
    for i, j in zip(xv.flatten(), yv.flatten()):
        inputs.append(np.array([i, j]))

    inputs = np.array(inputs)
    outputs, _ = approximator.predict(inputs, get_features=True,
                                      idx=np.ones(len(inputs),
                                      dtype=np.int) * game_idx)
    outputs = outputs[:, :n_actions].max(1).reshape(xv.shape)

    ax.plot_surface(xv, yv, outputs)

# Parameters
games = ['CartPole-v0', 'Acrobot-v1', 'MountainCar-v0', 'caronhill', 'pendulum']
path = '../results/multidqn/'
nets = np.array([30, 50])


# Create MDP to get parameters
mdp = list()
gamma_eval = list()
for i, g in enumerate(games):
    if g == 'pendulum':
        mdp.append(InvertedPendulumDiscrete(horizon=args.horizon[i],
                                            gamma=args.gamma[i]))
    elif g == 'caronhill':
        mdp.append(CarOnHill(horizon=args.horizon[i], gamma=args.gamma[i]))
    else:
        mdp.append(Gym(g, args.horizon[i], args.gamma[i]))

n_input_per_mdp = [m.info.observation_space.shape for m in mdp]
n_actions_per_head = [(m.info.action_space.n,) for m in mdp]
input_shape = [m.info.observation_space.shape for m in mdp]
features = 'relu'
observation_space = Box(np.array([-1.2, -0.07]), np.array([0.6, 0.07]))

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
        features=features
    )

approximator = Regressor(GymNetwork, approximator_params)

# Plot value functions
for i, j in enumerate(nets):
    weights = np.load(path + str(j) + '.npy')
    approximator.set_weights(weights)
    fig = plt.figure()
    ax = Axes3D(fig)
    v_plot(approximator, 2, observation_space, ax, n_actions_per_head[2][0])
plt.show()