import numpy as np
import scipy.stats as st
from matplotlib import pyplot as plt


def get_mean_and_confidence(data):
    mean = np.mean(data, axis=0)
    se = st.sem(data, axis=0)
    n = len(data)

    interval, _ = st.t.interval(0.95, n-1, scale=se)

    return mean, interval

algs = ['dqn', 'multidqn']
games = ['acro', 'mc']
game_ids = [1, 2]
reg = ['l1-weights-1e-4', 'l1-weights-1e-3']
activation = ['sigmoid']
n_games = len(games)

legend_items = list()

fig, ax = plt.subplots(n_games, 1)
for i, g in enumerate(games):
    ax[i].set_title(g)
    ax[i].grid()

for act in activation:
    for r in reg:
        for alg in algs:
            legend_items.append(alg + '-' + r + '-' + act)
            path = alg + '/' + r + '-' + act + '/'
            legend_items.append(r + '-' + act)
            a = np.load(path + 'scores.npy')
            a_mean, a_err = get_mean_and_confidence(a)
            for i, idx in enumerate(game_ids):
                ax[i].plot(a_mean[idx])
                ax[i].fill_between(np.arange(51), a_mean[idx] - a_err[idx], a_mean[idx] + a_err[idx], alpha=.5)
                
        legend_items.append('transfer' + '-' + r + '-' + act)
        for i, g in zip(game_ids, games):
            file_path = 'transfer' + '/' + g + '/' + r + '-' + act + '.npy'

            a = np.load(file_path)
            a_mean, a_err = get_mean_and_confidence(a)
            ax[i].plot(a_mean[0])
            ax[i].fill_between(np.arange(51), a_mean[0] - a_err[0], a_mean[0] + a_err[0], alpha=.5)

plt.legend(legend_items)
plt.show()
    
