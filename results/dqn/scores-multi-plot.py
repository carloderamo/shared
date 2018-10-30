import numpy as np
import scipy.stats as st
from matplotlib import pyplot as plt


def get_mean_and_confidence(data):
    mean = np.mean(data, axis=0)
    se = st.sem(data, axis=0)
    n = len(data)

    interval, _ = st.t.interval(0.95, n-1, scale=se)

    return mean, interval

alg = 'multidqn'
games = ['cart', 'acro', 'mc', 'coh', 'pend']
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
        path = alg + '/' + r + '-' + act + '/'
        legend_items.append(r + '-' + act)
        a = np.load(path + 'scores.npy')
        a_mean, a_err = get_mean_and_confidence(a)
        for i, g in enumerate(games):
            if g == '':
                continue
            ax[i].plot(a_mean[i])
            ax[i].fill_between(np.arange(51), a_mean[i] - a_err[i], a_mean[i] + a_err[i], alpha=.5)

plt.legend(legend_items)
plt.show()
    
