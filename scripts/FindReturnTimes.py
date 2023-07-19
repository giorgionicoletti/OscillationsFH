import numpy as np
import matplotlib.pyplot as plt

# import modules
import sys
sys.path.append('../modules')
import model
import utils

from scipy.optimize import fsolve

N = 200
Nsteps = 100000
dt = 0.01

Time = np.arange(Nsteps)*dt

epsilon = 0.3

mu = 1
muHat = 1
delta = -1/3
muI = 0.6
alpha = 0.1
beta = 0.8
gamma = 0.7
sigma = 0.1

model_kwargs = {'mu': mu, 'muHat': muHat, 'delta': delta,
                'muI': muI, 'alpha': alpha, 'beta': beta,
                'gamma': gamma, 'sigma': sigma}

Nrep = int(1e5)
N_list = [100, 200, 400]
tol_list = [1e-2, 1e-3]

for N in N_list:
    print('N = {}'.format(N))

    x0rand = np.random.randn(N)
    y0rand = np.random.randn(N)

    x_BR, y_BR = model.simulate_Fitzugh_Nagumo(N, int(2e5), dt, x0rand, y0rand, epsilon = epsilon)
    x_BR = x_BR[-100000:]
    y_BR = y_BR[-100000:]

    xCenter, yCenter = utils.estimate_baricenter(x_BR, y_BR)
    cov_slow, idxs_slow = utils.find_cov_slow(x_BR, y_BR, 0.05, return_idxs = True)

    xcycle, ycycle = utils.interpolate_cycle(x_BR.mean(axis = 1), y_BR.mean(axis = 1), plot_check = False)

    for tol in tol_list:    
        print('\t tol = {}'.format(tol))
        
        t_return_single_BR = model.find_return_times_single(Nrep, N, Nsteps, dt, cov_slow, xCenter, yCenter, epsilon,
                                                        xcycle, ycycle, tol = tol, **model_kwargs)

        t_return_mean_BR = model.find_return_times_mean(Nrep, N, Nsteps, dt, cov_slow, xCenter, yCenter, epsilon,
                                                        xcycle, ycycle, tol = tol, **model_kwargs)
        
        np.save('../data/return_times_single_BR_N{}_tol{}.npy'.format(N, tol), t_return_single_BR)
        np.save('../data/return_times_mean_BR_N{}_tol{}.npy'.format(N, tol), t_return_mean_BR)

for N in N_list:
    print('N = {}'.format(N))

    x0rand = np.random.randn(N)
    y0rand = np.random.randn(N)

    IQ = np.random.randn(N)*sigma + muI

    x_QR, y_QR = model.simulate_quenched_Fitzugh_Nagumo(N, int(2e5), dt, x0rand, y0rand, epsilon = epsilon, **model_kwargs, I = IQ)
    x_QR = x_QR[-100000:]
    y_QR = y_QR[-100000:]

    xCenter_Q, yCenter_Q = utils.estimate_baricenter(x_QR, y_QR)
    xcycle_Q, ycycle_Q = utils.interpolate_cycle(x_QR.mean(axis = 1), y_QR.mean(axis = 1), plot_check = False)
    cov_slow_Q, idxs_slow_Q = utils.find_cov_slow(x_QR, y_QR, 0.002, return_idxs = True)

    for tol in tol_list:
        print('\t tol = {}'.format(tol))

        t_ret_single_QR = model.find_return_times_single_Q(Nrep, N, Nsteps, dt, cov_slow_Q, xCenter_Q, yCenter_Q, epsilon,
                                                           xcycle_Q, ycycle_Q, tol = tol, **model_kwargs)

        t_return_mean_QR = model.find_return_times_mean_Q(Nrep, N, Nsteps, dt, cov_slow_Q, xCenter_Q, yCenter_Q, epsilon,
                                                        xcycle_Q, ycycle_Q, tol = tol, **model_kwargs)
        
        np.save('../data/return_times_single_QR_N{}_tol{}.npy'.format(N, tol), t_ret_single_QR)
        np.save('../data/return_times_mean_QR_N{}_tol{}.npy'.format(N, tol), t_return_mean_QR)