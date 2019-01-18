import pickle
import numpy as np
import scipy.stats as st
from matplotlib import pyplot as plt


def get_mean_and_confidence(data):
    mean = np.mean(data, axis=0)
    se = st.sem(data, axis=0)
    n = len(data)

    interval, _ = st.t.interval(0.95, n-1, scale=se)

    return mean, interval


single_folder_name = 'single/'
multi_folder_name = 'multi/'

args = pickle.load(open(multi_folder_name + 'args.pkl', 'rb'))

n_iterations = args.n_iterations
n_samples = args.max_steps
environments = args.pendulum_mass[1:]
n_games = len(environments)
n_exp = 50


# LOAD
delta_q_single_l2 = list()
delta_q_multi_l2 = list()
delta_q_single_max = list()
delta_q_multi_max = list()

max_q_single = list()
max_q_multi = list()


for exp in range(n_exp):
    delta_q_single_l2.append(list())
    delta_q_multi_l2.append(list())
    delta_q_single_max.append(list())
    delta_q_multi_max.append(list())
    max_q_single.append(list())
    max_q_multi.append(list())

    old_t_single = np.zeros(n_samples*n_games)
    old_t_multi = np.zeros(n_samples*n_games)

    for it in range(n_iterations):
        new_t_single = np.zeros(n_samples*n_games)
        new_t_multi = np.zeros(n_samples*n_games)

        delta_t_multi = np.load(
            multi_folder_name + 'delta-exp-%d-%d.npy' % (exp, it))[1:]
        min_t_multi = np.load(
            multi_folder_name + 'min-exp-%d-%d.npy' % (exp, it))[1:]
        target_t_multi = np.load(
            multi_folder_name + 'targets-exp-%d-%d.npy' % (exp, it))[n_samples:]

        for i, m in enumerate(environments):
            start = n_samples * i
            stop = start + n_samples

            delta_t_single = np.load(
                single_folder_name + '%.1f/delta-exp-%d-%d.npy' % (m, exp, it))
            min_t_single = np.load(
                single_folder_name + '%.1f/min-exp-%d-%d.npy' % (m, exp, it))
            target_t_single = np.load(
                single_folder_name + '%.1f/targets-exp-%d-%d.npy' % (m, exp, it))

            new_t_single[start:stop] = target_t_single * delta_t_single \
                                       + min_t_single
            new_t_multi[start:stop] = target_t_multi[start:stop] \
                                      * delta_t_multi[i] + min_t_multi[i]

        current_delta_t_single = new_t_single - old_t_single
        current_delta_t_multi = new_t_multi - old_t_multi

        max_q_single[-1].append(np.max(new_t_single[0:n_samples]))
        max_q_multi[-1].append(np.max(new_t_multi[0:n_samples]))
        delta_q_single_l2[-1].append(np.linalg.norm(current_delta_t_single))
        delta_q_multi_l2[-1].append(np.linalg.norm(current_delta_t_multi))
        delta_q_single_max[-1].append(np.linalg.norm(current_delta_t_single,
                                                     ord=np.inf))
        delta_q_multi_max[-1].append(np.linalg.norm(current_delta_t_multi,
                                                    ord=np.inf))

        old_t_single = new_t_single
        old_t_multi = new_t_multi

delta_q_single_l2 = np.array(delta_q_single_l2)
delta_q_multi_l2 = np.array(delta_q_multi_l2)
delta_q_single_max = np.array(delta_q_single_max)
delta_q_multi_max = np.array(delta_q_multi_max)
max_q_single = np.array(max_q_single)
max_q_multi = np.array(max_q_multi)


# PLOT
fig, ax = plt.subplots(3, 1)

single_mean_l2, single_err_l2 = get_mean_and_confidence(delta_q_single_l2)
multi_mean_l2, multi_err_l2 = get_mean_and_confidence(delta_q_multi_l2)

single_mean_max, single_err_max = get_mean_and_confidence(delta_q_single_max)
multi_mean_max, multi_err_max = get_mean_and_confidence(delta_q_multi_max)


iteration_array = np.arange(n_iterations)

ax[0].plot(single_mean_l2, linewidth=3)
ax[0].fill_between(iteration_array, single_mean_l2 - single_err_l2,
                 single_mean_l2 + single_err_l2, alpha=.5)

ax[0].plot(multi_mean_l2, linewidth=3)
ax[0].fill_between(iteration_array, multi_mean_l2 - multi_err_l2,
                 multi_mean_l2 + multi_err_l2, alpha=.5)

ax[0].set_title('L2')

ax[1].plot(single_mean_max, linewidth=3)
ax[1].fill_between(iteration_array, single_mean_max - single_err_max,
                 single_mean_max + single_err_max, alpha=.5)

ax[1].plot(multi_mean_max, linewidth=3)
ax[1].fill_between(iteration_array, multi_mean_max - multi_err_max,
                 multi_mean_max + multi_err_max, alpha=.5)

ax[1].set_title('L-inf')



single_mean_q, single_err_q = get_mean_and_confidence(max_q_single)
multi_mean_q, multi_err_q = get_mean_and_confidence(max_q_multi)

ax[2].plot(single_mean_q, linewidth=3)
ax[2].fill_between(iteration_array, single_mean_q - single_err_q,
                   single_mean_q + single_err_q, alpha=.5)

ax[2].plot(multi_mean_q, linewidth=3)
ax[2].fill_between(iteration_array, multi_mean_q - multi_err_q,
                   multi_mean_q + multi_err_q, alpha=.5)

ax[2].set_title('Max Q')

plt.legend(['single', 'multi'])

plt.show()

