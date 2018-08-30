import numpy as np
import scipy.stats as st
from matplotlib import pyplot as plt


def get_mean_and_confidence(data):
    mean = np.mean(data, axis=0)
    se = st.sem(data, axis=0)
    n = len(data)

    interval, _ = st.t.interval(0.95, n - 1, scale=se)

    return mean, interval
    
games = ['cart', 'acro', 'mc', 'coh', 'pend']

plt.subplot(2, 1, 1)
plt.title('LOSS')
a = np.load('loss.npy')
for i, g in enumerate(games):
    a_mean, a_err = get_mean_and_confidence(a[:, ::100, i])
    plt.plot(a_mean)
    plt.fill_between(np.arange(500), a_mean - a_err, a_mean + a_err, alpha=.5)
plt.grid()
plt.legend(games)

plt.subplot(2, 1, 2)
plt.title('L1-LOSS')
a = np.load('l1_loss.npy')
a_mean, a_err = get_mean_and_confidence(a[:, ::100])
plt.plot(a_mean)
plt.fill_between(np.arange(500), a_mean - a_err, a_mean + a_err, alpha=.5)
plt.grid()
    
plt.show()
    
