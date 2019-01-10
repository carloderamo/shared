import numpy as np
import scipy.stats as st
from matplotlib import pyplot as plt


def get_mean_and_confidence(data):
    mean = np.mean(data, axis=0)
    se = st.sem(data, axis=0)
    n = len(data)

    interval, _ = st.t.interval(0.95, n-1, scale=se)

    return mean, interval

folders = ['dqn', 'multidqn']
games = ['cart', 'acro', 'mc', 'coh', 'pend']
reg = ['noreg']
activation = ['sigmoid']
n_games = len(games)
n_settings = len(reg) * len(activation)

# plt.suptitle('DQN VS MULTI')

for i, g in enumerate(games):
    j = 1
    for act in activation:
        for r in reg:
            s = r + '-' + act
            plt.subplot(n_settings, n_games, i * n_settings + j)
            # plt.title(g + '-' + s)
            
            single = np.load('dqn/' + s + '/scores.npy')[:, i]
            single_mean, single_err = get_mean_and_confidence(single)
            
            multi = np.load('multidqn/' + s + '/scores.npy')[:, i]
            multi_mean, multi_err = get_mean_and_confidence(multi)
            
            plt.plot(single_mean, linewidth=3)
            plt.fill_between(np.arange(51), single_mean - single_err, single_mean + single_err, alpha=.5)
            
            plt.plot(multi_mean, linewidth=3)
            plt.fill_between(np.arange(51), multi_mean - multi_err, multi_mean + multi_err, alpha=.5)

            plt.xlabel('#Epochs', fontsize='xx-large')

            plt.xticks(fontsize='xx-large')
            plt.yticks(fontsize='xx-large')

            if i == 0:
                plt.ylabel('Performance', fontsize='xx-large')
            
            plt.grid()
            
            # plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
            
            j += 1

plt.legend(['DQN', 'MULTI'], fontsize='xx-large', loc='lower right')

plt.show()

