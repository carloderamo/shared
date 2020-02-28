import numpy as np
import scipy.stats as st
from matplotlib import pyplot as plt


def get_mean_and_confidence(data):
    mean = np.mean(data, axis=0)
    se = st.sem(data, axis=0)
    n = len(data)

    interval, _ = st.t.interval(0.95, n-1, scale=se)

    return mean, interval

games = ['Cart-Pole', 'Acrobot', 'Mountain-Car', 'Car-On-Hill', 'Inverted-Pendulum']
xlabels = ['#Target updates', '#Epochs']
ylabels = ['Learning progresses', '#Samples']
files = ['all_norm_lps.npy', 'n_samples_per_task.npy']
s = 'noreg-sigmoid/epsilon_40000_.2'

for i in range(2):
    plt.subplot(1, 2, i + 1)
    prism = np.load('prism/' + s + '/' + files[i])
    
    for j, g in enumerate(games):
        prism_mean, prism_err = get_mean_and_confidence(prism[..., j])

        plt.plot(prism_mean, linewidth=3)
        plt.fill_between(np.arange(prism_mean.shape[0]), prism_mean - prism_err, prism_mean + prism_err, alpha=.5)

        plt.xlabel(xlabels[i], fontsize=20)
        plt.ylabel(ylabels[i], fontsize=20)
        
        if i == 1:
            plt.xticks([0, 2500, 5000], [0, 25, 50], fontsize=20)
        else:
            plt.xticks(fontsize=20)

        plt.yticks(fontsize=20)
        
        plt.grid()
        
        # plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

plt.legend(games, fontsize=20, loc='upper left')
plt.plot(np.arange(5000), 10 * np.arange(5000), linewidth=3, color='k')

plt.show()

