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
leg_idx = 0 if show_pendulum else -1

if show_pendulum:
    alg = 'multi_pendulum'
    games = ['InvertedPendulumBulletEnv-v0', 'InvertedDoublePendulumBulletEnv-v0',
             'InvertedPendulumSwingupBulletEnv-v0']
    titles = ['Inverted-Pendulum', 'Inverted-Double-Pendulum', 'Inverted-Pendulum-Swingup']
else:
    alg = 'multi_walker'
    games = ['hop_stand', 'walk_walk', 'chee_run']
    titles = ['Hopper', 'Walker', 'Half-Cheetah']

reg = ['noreg']
activation = ['sigmoid']

n_games = len(games)

legend_items = list()

fig, ax = plt.subplots(1, n_games)
for i, t in enumerate(titles):
    ax[i].set_title(t, fontsize=22)
    ax[i].grid()

for r in reg:
    for act in activation:
        name = r + '-' + act
        legend_items.append('single ' + name)
        for i, g in enumerate(games):
            path = 'single/' + name + '/' + g
            a = np.load(path + '.npy')
            a_mean, a_err = get_mean_and_confidence(a)
            ax[i].plot(a_mean[0], linewidth=3)
            ax[i].fill_between(np.arange(len(a_mean[0])),
                               a_mean[0] - a_err[0],
                               a_mean[0] + a_err[0], alpha=.5)

if alg != '':
    for r in reg:
        for act in activation:
            name = r + '-' + act
            path = alg + '/' + name + '/'
    
            legend_items.append(name)
            a = np.load(path + 'scores.npy')
            a_mean, a_err = get_mean_and_confidence(a)
            for i, g in enumerate(games):
                ax[i].plot(a_mean[i], linewidth=3)
                ax[i].fill_between(np.arange(len(a_mean[i])), a_mean[i] - a_err[i], a_mean[i] + a_err[i], alpha=.5)
                ax[i].set_xlabel('#Epochs', fontsize=22)
                if i == 0:
                    ax[i].set_ylabel('Performance', fontsize=22)
                for tick in ax[i].xaxis.get_major_ticks():
                    tick.label.set_fontsize(22)
                    tick.label
                for tick in ax[i].yaxis.get_major_ticks():
                    tick.label.set_fontsize(22)
                ax[i].set_xticks([0, 50, 100])

ax[leg_idx].legend(['DDPG', 'MULTI'], loc='lower right', fontsize=22)
plt.show()
