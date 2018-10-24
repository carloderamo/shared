import numpy as np
import scipy.stats as st
from matplotlib import pyplot as plt


def get_mean_and_confidence(data):
    mean = np.mean(data, axis=0)
    se = st.sem(data, axis=0)
    n = len(data)

    interval, _ = st.t.interval(0.95, n - 1, scale=se)

    return mean, interval
    

def preprocess(dataset, evaluation_frequency=1000):
    n_steps = dataset.shape[1] // evaluation_frequency
    
    if dataset.ndim == 3:
        prepro_dataset = np.zeros((dataset.shape[0], n_steps, dataset.shape[2]))
    else:
        prepro_dataset = np.zeros((dataset.shape[0], n_steps))
    for i in range(n_steps):
        start = i * evaluation_frequency
        stop = start + evaluation_frequency
        prepro_dataset[:, i] = dataset[:, start:stop].mean(1)

    return np.array(prepro_dataset)

    
games = ['cart', 'acro', 'mc', 'coh', 'pend']

plt.subplot(3, 1, 1)
plt.title('LOSS')
a = np.load('loss.npy')
a = preprocess(a)
for i, g in enumerate(games):
    a_mean, a_err = get_mean_and_confidence(a[:, :, i])
    plt.plot(a_mean)
    plt.fill_between(np.arange(a_mean.shape[0]), a_mean - a_err, a_mean + a_err, alpha=.5)
plt.grid()

plt.subplot(3, 1, 2)
plt.title('L1-LOSS')
a = np.load('l1_loss.npy')
a = preprocess(a)
for i, g in enumerate(games):
    a_mean, a_err = get_mean_and_confidence(a[:, :, i])
    plt.plot(a_mean)
    plt.fill_between(np.arange(a_mean.shape[0]), a_mean - a_err, a_mean + a_err, alpha=.5)
plt.grid()

plt.subplot(3, 1, 3)
plt.title('Mean VALUE')
a = np.load('v.npy')
for i, g in enumerate(games):
    a_mean, a_err = get_mean_and_confidence(a[:, :, i])
    plt.plot(a_mean)
    plt.fill_between(np.arange(a_mean.shape[0]), a_mean - a_err, a_mean + a_err, alpha=.5)
plt.grid()
plt.legend(games)
    
plt.show()
    
