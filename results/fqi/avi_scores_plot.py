import numpy as np
import scipy.stats as st
from matplotlib import pyplot as plt


def get_mean_and_confidence(data):
    mean = np.mean(data, axis=0)
    se = st.sem(data, axis=0)
    n = len(data)

    interval, _ = st.t.interval(0.95, n - 1, scale=se)

    return mean, interval

games = ['1.000_4.000', '0.800_4.000', '1.000_4.500', '1.200_4.500']

plt.subplot(1, 2, 1)
a = list()
for g in games:
    a.append(np.load(g + '/avi_diff.npy'))
a = np.array(a)

fs = 25

a_mean, a_err = get_mean_and_confidence(a.mean(0))
plt.ylabel(r'$\Vert Q^* - Q^{\pi_K}\Vert$', fontsize=fs)
plt.xlabel('# Iterations', fontsize=fs)
plt.xticks([0, 25, 50], fontsize=fs)
plt.yticks(fontsize=fs)
plt.plot(a_mean, linewidth=3)
plt.fill_between(np.arange(a_mean.shape[-1]), a_mean - a_err, a_mean + a_err, alpha=.5)
    
a = np.load(''.join(games) + '/avi_diff.npy')

a_mean, a_err = get_mean_and_confidence(a)
plt.plot(a_mean, linewidth=3)
plt.fill_between(np.arange(a_mean.shape[-1]), a_mean - a_err, a_mean + a_err, alpha=.5)
plt.grid()
plt.legend(['FQI', 'MULTI'], fontsize=fs)

plt.subplot(1, 2, 2)
a = list()
for g in games:
    a.append(np.load(g + '/scores.npy'))
a = np.array(a)

a_mean, a_err = get_mean_and_confidence(a.mean(0))
plt.ylabel('Performance', fontsize=fs)
plt.xlabel('# Iterations', fontsize=fs)
plt.xticks([0, 25, 50], fontsize=fs)
plt.yticks(fontsize=fs)
plt.plot(a_mean, linewidth=3)
plt.fill_between(np.arange(a_mean.shape[-1]), a_mean - a_err, a_mean + a_err, alpha=.5)
    
a = np.load(''.join(games) + '/scores.npy')

a_mean, a_err = get_mean_and_confidence(a)
plt.fill_between(np.arange(a_mean.shape[-1]), a_mean - a_err, a_mean + a_err, alpha=.5)
plt.plot(a_mean, linewidth=3)
plt.grid()
    
plt.show()
    
