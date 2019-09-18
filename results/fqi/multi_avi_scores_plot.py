import numpy as np
import scipy.stats as st
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset


def get_mean_and_confidence(data):
    mean = np.mean(data, axis=0)
    se = st.sem(data, axis=0)
    n = len(data)

    interval, _ = st.t.interval(0.95, n - 1, scale=se)

    return mean, interval

games = ['1.000_4.000', '0.800_4.000', '1.000_4.500', '1.200_4.500', '1.000_4.125', '1.000_4.250', '1.000_4.375', '0.850_4.000']
n_tasks = [1, 2, 4, 8]
style = ['-', '-.', '--', ':']

fig, ax = plt.subplots()
for j, i in enumerate(n_tasks):
    a = np.load(''.join(games[:i]) + '/avi_diff.npy')

    a_mean, a_err = get_mean_and_confidence(a)
    ax.plot(a_mean, linewidth=3, linestyle=style[j])
    ax.fill_between(np.arange(a_mean.shape[-1]), a_mean - a_err, a_mean + a_err, alpha=.5)
    
fs = 25
    
plt.xticks([0, 25, 50], fontsize=fs)
plt.yticks(fontsize=fs)
plt.ylabel(r'$\Vert Q^* - Q^{\pi_K}\Vert$', fontsize=fs)
plt.xlabel('# Iterations', fontsize=fs)
plt.grid()
plt.legend(n_tasks, fontsize=fs)

axins = zoomed_inset_axes(ax, 2, loc=9) # zoom-factor: 2.5, location: upper-left
mark_inset(ax, axins, loc1=4, loc2=3, fc="none", ec="0.5")
for j, i in enumerate(n_tasks):
    a = np.load(''.join(games[:i]) + '/avi_diff.npy')
    
    a_mean, a_err = get_mean_and_confidence(a)
    axins.plot(a_mean, linewidth=3, linestyle=style[j])
    axins.fill_between(np.arange(a_mean.shape[-1]), a_mean - a_err, a_mean + a_err, alpha=.5)
    x1, x2, y1, y2 = 40, 49, .155, .225 # specify the limits
    axins.set_xlim(x1, x2) # apply the x-limits
    axins.set_ylim(y1, y2) # apply the y-limits
axins.grid()
axins.set_xticks([])
axins.set_yticks([])
    
plt.show()

