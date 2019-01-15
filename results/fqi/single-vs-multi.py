import numpy as np
import scipy.stats as st
from matplotlib import pyplot as plt


def get_mean_and_confidence(data):
    mean = np.mean(data, axis=0)
    se = st.sem(data, axis=0)
    n = len(data)

    interval, _ = st.t.interval(0.95, n-1, scale=se)

    return mean, interval

folders = ['fqi', 'multifqi']
games = ['2', '5', '10']
n_games = len(games)

# plt.suptitle('FQI VS MULTI')

for i, g in enumerate(games):
    j = 1
    plt.subplot(1, n_games, i + 1)
    plt.title(g, fontsize='xx-large')
    
    single = np.load('single/' + g + '.npy')[:, 0]
    single_mean, single_err = get_mean_and_confidence(single)
    
    multi = np.load('multi/scores.npy')[:, i]
    multi_mean, multi_err = get_mean_and_confidence(multi)
    
    plt.plot(single_mean, linewidth=3)
    plt.fill_between(np.arange(20), single_mean - single_err, single_mean + single_err, alpha=.5)
    
    plt.plot(multi_mean, linewidth=3)
    plt.fill_between(np.arange(20), multi_mean - multi_err, multi_mean + multi_err, alpha=.5)

    plt.xlabel('#Epochs', fontsize='xx-large')

    plt.xticks([0, 10, 20], fontsize='xx-large')
    plt.yticks(fontsize='xx-large')

    if i == 0:
        plt.ylabel('Performance', fontsize='xx-large')
    
    plt.grid()
    
    # plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    
    j += 1

plt.legend(['FQI', 'MULTI'], fontsize='xx-large', loc='lower right')

plt.show()

