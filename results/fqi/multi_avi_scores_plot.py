import numpy as np
import scipy.stats as st
from matplotlib import pyplot as plt


def get_mean_and_confidence(data):
    mean = np.mean(data, axis=0)
    se = st.sem(data, axis=0)
    n = len(data)

    interval, _ = st.t.interval(0.95, n - 1, scale=se)

    return mean, interval

games = ['1.000_4.000', '1.000_4.500', '0.800_4.000', '1.200_4.500', '1.000_4.125', '1.000_4.250', '1.000_4.375', '0.850_4.000']
n_tasks = [1, 4, 8]
    
for i in n_tasks:
    a = np.load(''.join(games[:i]) + '/avi_diff.npy')

    a_mean, a_err = get_mean_and_confidence(a)
    plt.plot(a_mean, linewidth=3)
    plt.fill_between(np.arange(a_mean.shape[-1]), a_mean - a_err, a_mean + a_err, alpha=.5)
plt.grid()
plt.legend(n_tasks, fontsize='xx-large')
    
plt.show()
    
