import numpy as np
import scipy.stats as st
from matplotlib import pyplot as plt


def get_mean_and_confidence(data):
    mean = np.mean(data, axis=0)
    se = st.sem(data, axis=0)
    n = len(data)

    interval, _ = st.t.interval(0.95, n - 1, scale=se)

    return mean, interval
    
name = 'acro'

algs = [name + '.npy', 'transfer.npy', 'freeze.npy']

for alg in algs:
    if alg != 'multi.npy':
        a = np.load(alg)[:, 0]
    else:
        a = np.load(alg)[:, 1]

    plt.title(alg)
    a_mean, a_err = get_mean_and_confidence(a)
    
    plt.plot(a_mean)
    plt.fill_between(np.arange(51), a_mean - a_err, a_mean + a_err, alpha=.5)

    plt.grid()
    
plt.legend(algs)
    
plt.show()
    
