import numpy as np
import scipy.stats as st
from matplotlib import pyplot as plt


def get_mean_and_confidence(data):
    mean = np.mean(data, axis=0)
    se = st.sem(data, axis=0)
    n = len(data)

    interval, _ = st.t.interval(0.95, n-1, scale=se)

    return mean, interval

base_dir = 'multitask'
games = ['puddleworld']
reg = ['single', 'multi', 'kl-1e-2-5', 'kl-1e-2-10', 
       'kl-1e-2-15', 'kl-1e-2-20', 'kl-1e-2-25', 'kl-1e-2-30']
#reg = ['single', 'kl-1e-2-30', 'kl-1e-2-5_30', 'reduced-30']
activation = ['sigmoid']

legend_items = list()

for g in games:
    fig = plt.figure()
    for act in activation:
        for r in reg:
            path = base_dir + '/' + g + '/' + r + '-' + act + '/'
            legend_items.append(r + '-' + act)
            a = np.squeeze(np.load(path + 'scores.npy'))
            a_mean, a_err = get_mean_and_confidence(a)
            plt.plot(a_mean)
            '''plt.fill_between(np.arange(len(a_mean)),
                             a_mean - a_err,
                             a_mean + a_err,
                             alpha=.5)'''

plt.legend(legend_items)
plt.show()
