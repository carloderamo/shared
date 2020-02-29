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
ylabels = ['Learning progress', '#Samples']
files = ['all_norm_lps.npy', 'n_samples_per_task.npy']
s = ['noreg-sigmoid/epsilon_40000_.2', 'noreg-sigmoid/epsilon_1_.1']
colors = ['darkred', 'orangered', 'gold', 'mediumseagreen', 'royalblue']
titles = [r'$\varepsilon=0.2$', r'$\varepsilon=0.1$']
sp = [1, 3, 2, 4]

for i in range(4):
    plt.subplot(2, 2, sp[i])
    prism = np.load('prism/' + s[i // 2] + '/' + files[i % 2])

    for j, g in enumerate(games):
        prism_mean, prism_err = get_mean_and_confidence(prism[..., j])

        plt.plot(prism_mean, linewidth=3, color=colors[j])
        plt.fill_between(np.arange(prism_mean.shape[0]), prism_mean - prism_err, prism_mean + prism_err, alpha=.5, color=colors[j])

        plt.xlabel(xlabels[i % 2], fontsize=20)
        if i // 2 == 0:
            plt.ylabel(ylabels[i % 2], fontsize=20)

        if i % 2 == 1:
            plt.xticks([0, 2500, 5000], [0, 25, 50], fontsize=20)
            plt.plot(np.arange(5000), 10 * np.arange(5000), linewidth=3, color='k')

        else:
            plt.xticks(fontsize=20)
            plt.title(titles[i // 2], fontsize=20)

        plt.yticks(fontsize=20)

        plt.grid()

        # plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

plt.legend(games, fontsize=20, loc='best', bbox_to_anchor=(.5, -.2, .5, 0) , ncol=len(games))

plt.show()

