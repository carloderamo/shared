import numpy as np
import scipy.stats as st
from matplotlib import pyplot as plt


def get_mean_and_confidence(data):
    mean = np.mean(data, axis=0)
    se = st.sem(data, axis=0)
    n = len(data)

    interval, _ = st.t.interval(0.95, n-1, scale=se)

    return mean, interval

games = ['acro-noreg']
game_ids = [1]
reg = ['noreg']
activation = ['sigmoid']
n_games = len(games)
unfreezes = [0, 10, 51]

legend_items = list()

fig, ax = plt.subplots(n_games, 1)
# for i, g in enumerate(games):
#     ax.set_title(g)
#     ax.grid()

for act in activation:
    for r in reg:
        legend_items.append('No initialization')
        path = 'dqn/' + r + '-' + act + '/'
        a = np.load(path + 'scores.npy')
        a_mean, a_err = get_mean_and_confidence(a)
        for i, idx in enumerate(game_ids):
            ax.plot(a_mean[idx], linewidth=3)
            ax.fill_between(np.arange(51), a_mean[idx] - a_err[idx], a_mean[idx] + a_err[idx], alpha=.5)
    
    for u in unfreezes:
        for i, g in zip(game_ids, games):
            if u == 51:
                legend_items.append('No unfreeze')
            else:
                legend_items.append('Unfreeze-' + str(u))
            file_path = 'transfer' + '/' + g + '/unfreeze' + str(u) + '-' + r + '-' + act + '.npy'

            a = np.load(file_path)
            a_mean, a_err = get_mean_and_confidence(a)
            ax.plot(a_mean[0], linewidth=3)
            ax.fill_between(np.arange(51), a_mean[0] - a_err[0], a_mean[0] + a_err[0], alpha=.5)

plt.xlabel('#Epochs', fontsize=35)
plt.ylabel('Performance', fontsize=35)
plt.xticks([0, 25, 50], fontsize=35)
plt.yticks(fontsize=35)
plt.title('Acrobot', fontsize=35)

plt.grid()

plt.legend(legend_items, fontsize=25, loc='lower right')

plt.show()
    
