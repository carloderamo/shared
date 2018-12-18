import numpy as np
import scipy.stats as st
from matplotlib import pyplot as plt


def get_mean_and_confidence(data):
    mean = np.mean(data, axis=0)
    se = st.sem(data, axis=0)
    n = len(data)

    interval, _ = st.t.interval(0.95, n-1, scale=se)

    return mean, interval

show_pendulum = False

if show_pendulum:
    alg = 'multi_pendulum'
    games = ['InvertedPendulumBulletEnv-v0', 'InvertedDoublePendulumBulletEnv-v0',
             'InvertedPendulumSwingupBulletEnv-v0']
else:
    alg = 'multi_walkers'
    games = ['AntBulletEnv-v0', 'HopperBulletEnv-v0',
             'Walker2DBulletEnv-v0', 'HalfCheetahBulletEnv-v0']

reg = ['noreg', 'kl-100-1e-2']
activation = ['sigmoid']

n_games = len(games)

legend_items = list()

fig, ax = plt.subplots(1, n_games)
for i, g in enumerate(games):
    ax[i].set_title(g)
    ax[i].grid()

if alg != '':
    for r in reg:
        for act in activation:
            name = r + '-' + act
            path = alg + '/' + name + '/'
    
            legend_items.append(name)
            a = np.load(path + 'scores.npy')
            a_mean, a_err = get_mean_and_confidence(a)
            for i, g in enumerate(games):
                ax[i].plot(a_mean[i])
                ax[i].fill_between(np.arange(len(a_mean[i])), a_mean[i] - a_err[i], a_mean[i] + a_err[i], alpha=.5)

for r in reg:
    for act in activation:
        name = r + '-' + act
        legend_items.append('single ' + name)
        for i, g in enumerate(games):
            path = 'single/' + name + '/' + g + '/'
            a = np.load(path + 'scores.npy')
            a_mean, a_err = get_mean_and_confidence(a)
            ax[i].plot(a_mean[0])
            ax[i].fill_between(np.arange(len(a_mean[0])),
                               a_mean[0] - a_err[0],
                               a_mean[0] + a_err[0], alpha=.5)

plt.legend(legend_items)
plt.show()
