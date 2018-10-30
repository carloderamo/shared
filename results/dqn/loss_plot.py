import numpy as np
import scipy.stats as st
from matplotlib import pyplot as plt


def get_mean_and_confidence(data):
    mean = np.mean(data, axis=0)
    se = st.sem(data, axis=0)
    n = len(data)

    interval, _ = st.t.interval(0.95, n - 1, scale=se)

    return mean, interval

alg = 'multidqn'
games = ['cart', 'acro', 'mc', 'coh', 'pend']
reg = ['noreg', 'l1', 'l1-weights']
activation = ['relu', 'sigmoid']

n_cols = len(reg) * len(activation)

k = 1
for act in activation:
    for r in reg:
        title = r + '-' + act
        path = alg + '/' + title + '/'
    
        plt.subplot(3, n_cols, k)
        plt.title(title)
        a = np.load(path + 'loss.npy')
        for i, g in enumerate(games):
            a_mean, a_err = get_mean_and_confidence(a[:, :, i])
            plt.plot(a_mean)
            plt.fill_between(np.arange(a_mean.shape[0]), a_mean - a_err, a_mean + a_err, alpha=.5)
        plt.grid()
        plt.ylim([-0.1, 0.6])

        plt.subplot(3, n_cols, n_cols + k)
        a = np.load(path + 'reg_loss.npy')
        for i, g in enumerate(games):
            a_mean, a_err = get_mean_and_confidence(a[:, :, i])
            plt.plot(a_mean)
            plt.fill_between(np.arange(a_mean.shape[0]), a_mean - a_err, a_mean + a_err, alpha=.5)
        plt.grid()
        plt.ylim([0, 80])

        plt.subplot(3, n_cols, 2 * n_cols + k)
        a = np.load(path + 'v.npy')
        for i, g in enumerate(games):
            a_mean, a_err = get_mean_and_confidence(a[:, :, i])
            plt.plot(a_mean)
            plt.fill_between(np.arange(a_mean.shape[0]), a_mean - a_err, a_mean + a_err, alpha=.5)
        plt.grid()
        plt.ylim([-80, 110])
        
        k += 1

plt.legend(games)
    
plt.show()
    
