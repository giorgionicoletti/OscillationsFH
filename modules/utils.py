import numpy as np
import numba as nb

from scipy.spatial import ConvexHull, Delaunay
import matplotlib.pyplot as plt

import scipy.signal

from sympy import solve
from sympy.abc import x, y


def find_fixed_points(IQ, epsilon, mu = 1, muHat = 1, delta = -1/3,
                      muI = 0.6, alpha = 0.1,
                      beta = 0.8, gamma = 0.7, sigma = 0.1):
    N = len(IQ)
    xy_UNST_Q = np.zeros((N, 3, 2), dtype=np.complex128)

    for idx, curr_I in enumerate(IQ):
        xy_UNST_Q[idx] = solve([(mu + epsilon)*x + delta*x**3 - muHat*y + curr_I, x - beta*y + gamma], x, y)

    return xy_UNST_Q

def estimate_baricenter(x, y):
    """
    Finds the baricenter of a trajectory in the x-y plane.

    Parameters
    ----------
    x : array
        Trajectory in the x direction.
    y : array
        Trajectory in the y direction.

    Returns
    -------
    C : array
        Baricenter of the trajectory.
    """
    points = np.vstack((x.mean(axis = -1), y.mean(axis = -1))).T

    T = Delaunay(points).simplices
    n = T.shape[0]
    W = np.zeros(n)
    C = 0
    
    for m in range(n):
        sp = points[T[m, :], :]
        W[m] = ConvexHull(sp).volume
        C += W[m] * np.mean(sp, axis=0)
    
    return C / np.sum(W)

@nb.njit
def distance_from_cycle(x0, y0, xcyle, ycyle):
    """
    Computes the distance between a point and the limit cycle.

    Parameters
    ----------
    x0 : float  
        x coordinate of the point.
    y0 : float
        y coordinate of the point.

    Returns
    -------
    dist : float
        Distance between the point and the limit cycle.
    """
    dist = np.inf
    for i in range(xcyle.shape[0]):
        new_dist = np.sqrt((x0 - xcyle[i])**2 + (y0 - ycyle[i])**2)
        if new_dist < dist:
            dist = new_dist

    return dist

@nb.njit
def sample_multivariate_normal(mean, cov, size):
    """
    Samples from a multivariate normal distribution.
    Efficient implementation using the Cholesky decomposition 
    and numba.

    Parameters
    ----------
    mean : array
        Mean of the distribution.
    cov : array
        Covariance matrix of the distribution.
    size : int
        Number of samples to draw.

    Returns
    -------
    samples : array
        Samples from the multivariate normal distribution.
    """
    N = mean.shape[0]
    A = np.linalg.cholesky(cov)
    Z = np.random.randn(N, size)
    
    return A @ Z + np.expand_dims(mean, axis = 1)


def find_cov_slow(x, y, eps = 0.001, return_idxs = False):
    """
    Computes the covarance matrix between x and y 
    along the low-variance slow branch of the limit cycle.

    Parameters
    ----------
    x : array
        x population.
    y : array
        y population.
    eps : float
        Threshold for the variance.

    Returns
    -------
    cov : array
        Covariance matrix between x and y.
    """
    idxs = np.where(np.var(x, axis = 1) < eps)[0]

    cov = np.zeros((2, 2), dtype = np.float64)

    for t in idxs:
        cov += np.cov(x[t, :], y[t, :])

    cov /= len(idxs)
    if return_idxs:
        return cov, idxs
    else:
        return cov


def plot_results(x, y, Time, title, idxs_to_plot = [-7500, -5000, -2500, -1], Nosc_to_plot = 100, steps_to_plot = 10000):
    """
    Plot the simulation results in a readable way.

    Parameters
    ----------
    x : array
        Simulated x population.
    y : array
        Simulated y population.
    Time : array
        Time array.
    title : string
        Title of the plot.
    idxs_to_plot : array
        Indexes of the timesteps at which to plot the population
        of oscillators.
    Nosc_to_plot : int
        Number of oscillators to plot.
    steps_to_plot : int
        Number of time steps to plot.

    Returns
    -------
    fig : figure
        Figure object.
    ax : array
        Array of axes objects.
    """
    fig, ax = plt.subplots(1, 4, figsize=(20, 3))

    ax[0].plot(np.mean(x, axis=1), np.mean(y, axis=1), color='black', alpha=1)
    ax[0].plot(x[:, :Nosc_to_plot], y[:, :Nosc_to_plot], color='gray', alpha=0.5, lw = 0.05, zorder = -1)
    ax[0].set_xlabel(r'$\langle x \rangle$')
    ax[0].set_ylabel(r'$\langle y \rangle$')
    ax[0].set_title(title)
    ax[0].scatter(x[0].mean(), y[0].mean(), color='black', alpha=1, lw = 0.05, zorder = -1, 
                  label = r'$t = {:.0f}$'.format(Time[0]))
    ax[0].legend(loc = 'upper left')

    for idx in idxs_to_plot:
        ax[1].scatter(x[idx, :Nosc_to_plot], y[idx, :Nosc_to_plot], alpha=1, lw = 0.05, zorder = -1,
                      label = r'$t = {:.0f}$'.format(Time[idx]))
    ax[1].legend(loc = 'upper left', ncol = 2)
    ax[1].plot(np.mean(x, axis=1)[x.shape[0]//2:], np.mean(y, axis=1)[x.shape[0]//2:], color='black', alpha=0.5)
    ax[1].set_xlabel(r'$x_i$')
    ax[1].set_ylabel(r'$y_i$')
    ax[1].set_xlim(ax[0].get_xlim())
    ax[1].set_ylim(ax[0].get_ylim())

    ax[2].plot(Time[:steps_to_plot], x[:steps_to_plot, :Nosc_to_plot], alpha=0.5, lw = 0.5, zorder = -1)
    ax[2].set_xlabel(r'$t$')
    ax[2].set_ylabel(r'$x_i$')
    
    ax[3].plot(Time, np.mean(x, axis=1), label = r'$\langle x \rangle$', alpha=1)
    ax[3].plot(Time, np.mean(y, axis=1), label = r'$\langle y \rangle$', alpha=1)
    ax[3].set_xlabel(r'$t$')
    ax[3].set_ylabel(r'$\langle x \rangle$, $\langle y \rangle$')
    ax[3].legend()

    plt.subplots_adjust(wspace=0.3)

    return fig, ax

@nb.njit
def difference_all_points(x, reg = 0.1):
    """
    Computes the difference between all points in x.

    Parameters
    ----------
    x : array
        Array of points.
    reg : float
        Regularization parameter.

    Returns
    -------
    diff : array
        Array of differences.
    """
    N = x.shape[0]
    diff = np.zeros(N, dtype=np.float64)
    for i in range(N):
        for j in range(N):
            if j != i:
                diff[i] += 1/(reg + np.abs(x[i] - x[j]))
    return diff

def interpolate_cycle(xmean, ymean, order = 200, plot_check = False):
    minima_idxs = scipy.signal.argrelextrema(xmean, np.less, order = order)[0]

    if plot_check:
        plt.scatter(minima_idxs, xmean[minima_idxs], color = 'darkred', zorder = 10)
        plt.plot(xmean, color = 'black', alpha = 0.5)
        plt.plot(ymean, color = 'black', alpha = 0.5)
        plt.show()
    
    cycles = []
    for i in range(len(minima_idxs) - 1):
        cycles.append((xmean[minima_idxs[i]:minima_idxs[i+1]], ymean[minima_idxs[i]:minima_idxs[i+1]]))

    cycle_length = np.mean([len(cycle[0]) for cycle in cycles])
    xcycle_avg = np.zeros(int(cycle_length))
    ycycle_avg = np.zeros(int(cycle_length))
    for cycle in cycles:
        xcycle_avg += scipy.signal.resample(cycle[0], int(cycle_length))
        ycycle_avg += scipy.signal.resample(cycle[1], int(cycle_length))
    xcycle_avg /= len(cycles)
    ycycle_avg /= len(cycles)

    if plot_check:
        plt.scatter(xcycle_avg, ycycle_avg, color = 'darkred', s = 1, zorder = 10)
        plt.plot(xmean, ymean, ls = '--')
        plt.show()

    return xcycle_avg, ycycle_avg


