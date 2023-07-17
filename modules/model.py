import numpy as np
import numba as nb

import utils

def equations(p, *args):
    """
    Fitzugh-Nagumo equations.

    Parameters
    ----------
    p : array
        Population state.
    args : tuple
        Parameters of the model.

    Returns
    -------
    dxdt : float
        Time derivative of the x population.
    dydt : float
        Time derivative of the y population.
    """
    mu, muHat, delta, muI, epsilon, alpha, beta, gamma = args
    x, y = p
    return (mu + epsilon)*x + delta*x**3 - muHat*y + muI, alpha*(x - beta*y + gamma)


@nb.njit
def simulate_Fitzugh_Nagumo(N, Nsteps, dt, x0, y0,
                            epsilon,
                            mu = 1, muHat = 1, delta = -1/3,
                            muI = 0.6, alpha = 0.1,
                            beta = 0.8, gamma = 0.7, sigma = 0.1):
    """
    Simulate the Fitzugh-Nagumo model with additive noise.

    Parameters
    ----------
    N : int
        Number of neurons in the x population.
    Nsteps : int
        Number of time steps to simulate.
    dt : float
        Time step size.
    x0 : array
        Initial condition for the x population.
    y0 : array
        Initial condition for the y population.
    epsilon : float
        Interaction strength between the x population and
        the mean of the x population.
    mu : float
        Self-coupling of the x population.
    muHat : float
        Coupling from the y to the x population.
    delta : float
        Nonlinearity of the x population.
    muI : float
        External input to the x population.
    alpha : float
        Coupling from the x to the y population.
    beta : float
        Self-coupling of the y population.
    gamma : float
        External input to the y population.
    sigma : float
        Noise level.

    Returns
    -------
    x : array
        Simulated x population.
    y : array
        Simulated y population.
    """

    x = np.zeros((Nsteps, N), dtype=np.float64)
    y = np.zeros((Nsteps, N), dtype=np.float64)

    x[0] = x0
    y[0] = y0
    sqdt = np.sqrt(dt)

    for t in range(1, Nsteps):
        x[t] = x[t-1] + dt * (mu*x[t-1] + delta*x[t-1]**3 - muHat*y[t-1] + muI + epsilon*np.mean(x[t-1]))
        x[t] += sigma*sqdt*np.random.randn(N)

        y[t] = y[t-1] + dt * alpha*(x[t-1] - beta*y[t-1] + gamma)
        y[t] += sigma*sqdt*np.random.randn(N)

    return x, y

@nb.njit
def simulate_quenched_Fitzugh_Nagumo(N, Nsteps, dt, x0, y0,
                                     epsilon,
                                     mu = 1, muHat = 1, delta = -1/3,
                                     muI = 0.6, alpha = 0.1,
                                     beta = 0.8, gamma = 0.7, sigma = 0.1, I = None):
    """
    Simulate the Fitzugh-Nagumo model with quenched currents.

        Parameters
    ----------
    N : int
        Number of neurons in the x population.
    Nsteps : int
        Number of time steps to simulate.
    dt : float
        Time step size.
    x0 : array
        Initial condition for the x population.
    y0 : array
        Initial condition for the y population.
    epsilon : float
        Interaction strength between the x population and
        the mean of the x population.
    mu : float
        Self-coupling of the x population.
    muHat : float
        Coupling from the y to the x population.
    delta : float
        Nonlinearity of the x population.
    muI : float
        Mean of the quenched currents.
    epsilon : float
        Interaction strength between the x population and
        the mean of the x population.
    alpha : float
        Coupling from the x to the y population.
    beta : float
        Self-coupling of the y population.
    gamma : float
        External input to the y population.
    sigma : float
        Variance of the quenched currents.
    currents : bool
        If True, 

    Returns
    -------
    x : array
        Simulated x population.
    y : array
        Simulated y population.
    """
    x = np.zeros((Nsteps, N), dtype=np.float64)
    y = np.zeros((Nsteps, N), dtype=np.float64)

    if I is None:
        I = np.random.randn(N)*sigma + muI

    x[0] = x0
    y[0] = y0

    for t in range(1, Nsteps):
        x[t] = x[t-1] + dt * (mu*x[t-1] + delta*x[t-1]**3 - muHat*y[t-1] + epsilon*np.mean(x[t-1]) + I)

        y[t] = y[t-1] + dt * alpha*(x[t-1] - beta*y[t-1] + gamma)

    return x, y

@nb.njit
def simulate_quenched_return_FH(Nsteps, dt, x0, y0,
                                epsilon, I, xcycle, ycycle, tol = 1e-3,
                                mu = 1, muHat = 1, delta = -1/3,
                                alpha = 0.1, beta = 0.8, gamma = 0.7):
    """
    Simulate the Fitzugh-Nagumo model with quenched currents from a given
    initial condition until the trajectory returns to the limit cycle.

    Parameters
    ----------
    Nsteps : int
        Number of maximum time steps to simulate.
    dt : float
        Time step size.
    x0 : array
        Initial condition for the x population.
    y0 : array
        Initial condition for the y population.
    epsilon : float
        Interaction strength between the x population and
        the mean of the x population.
    I : array
        Quenched currents.
    xcycle : array
        Limit cycle of the x population.
    ycycle : array
        Limit cycle of the y population.
    tol : float
        Tolerance for returning to the limit cycle.
    mu : float
        Self-coupling of the x population.
    muHat : float
        Coupling from the y to the x population.
    delta : float
        Nonlinearity of the x population.
    epsilon : float
        Interaction strength between the x population and
        the mean of the x population.
    alpha : float
        Coupling from the x to the y population.
    beta : float
        Self-coupling of the y population.
    gamma : float
        External input to the y population.

    Returns
    -------
    t : int
        Time to return to the limit cycle.
    x : array
        Final state of the x population.
    y : array
        Final state of the y population.
    """

    x = x0
    y = y0
    
    for t in range(1, Nsteps):
        xnew = x + dt * (mu*x + delta*x**3 - muHat*y + epsilon*np.mean(x) + I)
        ynew = y + dt * alpha*(x - beta*y + gamma)

        dist_cycle = utils.distance_from_cycle(np.mean(xnew), np.mean(ynew), xcycle, ycycle)

        if dist_cycle > tol:
            x = xnew
            y = ynew
        else:
            break

    return t*dt, x, y




@nb.njit(parallel = True)
def find_quenched_return_times(Nrep, N, Nsteps, dt, cov0, mux0, muy0,
                               epsilon, I, xcycle, ycycle, tol = 1e-3,
                               mu = 1, muHat = 1, delta = -1/3,
                               alpha = 0.1,
                               beta = 0.8, gamma = 0.7):
    """
    Returns the distribution of the time to return to the limit cycle
    for a given set of quenched currents, averaged over Nrep realizations
    of the initial conditions.

    Parameters
    ----------
        Nsteps : int
        Number of maximum time steps to simulate.
    dt : float
        Time step size.
    x0 : array
        Initial condition for the x population.
    y0 : array
        Initial condition for the y population.
    epsilon : float
        Interaction strength between the x population and
        the mean of the x population.
    I : array
        Quenched currents.
    xcycle : array
        Limit cycle of the x population.
    ycycle : array
        Limit cycle of the y population.
    tol : float
        Tolerance for returning to the limit cycle.
    mu : float
        Self-coupling of the x population.
    muHat : float
        Coupling from the y to the x population.
    delta : float
        Nonlinearity of the x population.
    epsilon : float
        Interaction strength between the x population and
        the mean of the x population.
    alpha : float
        Coupling from the x to the y population.
    beta : float
        Self-coupling of the y population.
    gamma : float
        External input to the y population.

    Returns
    -------
    t : array
        Times to return to the limit cycle.
    """
    t = np.zeros(Nrep)
    
    for idx in nb.prange(Nrep):
        x0, y0 = utils.sample_multivariate_normal(np.array([mux0, muy0]), cov0, N)
        t[idx], _, _ = simulate_quenched_return_FH(Nsteps, dt, x0, y0,
                                                   epsilon, I, xcycle, ycycle, tol,
                                                   mu, muHat, delta,
                                                   alpha, beta, gamma)
        
    return t

@nb.njit
def simulate_pulse_Fitzugh_Nagumo(N, Nsteps, dt, x0, y0, theta = 0.001,
                                  pulse_val = 0., pulse_start = 0, pulse_end = 0,
                                  mu = 1, muHat = 1, delta = -1/3,
                                  muI = 0.6, epsilon = 0.5, alpha = 0.1,
                                  beta = 0.8, gamma = 0.7, sigma = 0.1):
    """
    Simulate the Fitzugh-Nagumo model with additive noise. A pulse is given
    to all oscillators at a given time. An extra parameter theta is added
    to the x population to account for a repulsive interaction from the mean.

    Parameters
    ----------
    N : int
        Number of neurons in the x population.
    Nsteps : int
        Number of time steps to simulate.
    dt : float
        Time step size.
    x0 : array
        Initial condition for the x population.
    y0 : array
        Initial condition for the y population.
    theta : float
        Interaction strength between the x population and
        the mean of the x population, repulsive.
    pulse_val : float
        Value of the pulse.
    pulse_start : int
        Time step at which the pulse starts.
    pulse_end : int
        Time step at which the pulse ends.
    epsilon : float
        Interaction strength between the x population and
        the mean of the x population.
    mu : float
        Self-coupling of the x population.
    muHat : float
        Coupling from the y to the x population.
    delta : float
        Nonlinearity of the x population.
    muI : float
        External input to the x population.
    alpha : float
        Coupling from the x to the y population.
    beta : float
        Self-coupling of the y population.
    gamma : float
        External input to the y population.
    sigma : float
        Noise level.

    Returns
    -------
    x : array
        Simulated x population.
    y : array
        Simulated y population.
    """

    x = np.zeros((Nsteps, N), dtype=np.float64)
    y = np.zeros((Nsteps, N), dtype=np.float64)

    x[0] = x0
    y[0] = y0
    sqdt = np.sqrt(dt)

    pulse = np.zeros(Nsteps, dtype=np.float64)
    pulse[pulse_start:pulse_end] = pulse_val

    for t in range(1, Nsteps):
        m = np.mean(x[t-1])
        x[t] = x[t-1] + dt * (mu*x[t-1] + delta*x[t-1]**3 - muHat*y[t-1] + muI + epsilon*m + theta*(x[t-1] - m) + pulse[t])
        x[t] += sigma*sqdt*np.random.randn(N)

        y[t] = y[t-1] + dt * alpha*(x[t-1] - beta*y[t-1] + gamma + pulse[t])
        y[t] += sigma*sqdt*np.random.randn(N)

    return x, y



@nb.njit
def simulate_repulsive_Fitzugh_Nagumo(N, Nsteps, dt, x0, y0, k, epsilon,
                                      mu = 1, muHat = 1, delta = -1/3,
                                      muI = 0.6, alpha = 0.1,
                                      beta = 0.8, gamma = 0.7, sigma = 0.1):
    """
    Simulate the Fitzugh-Nagumo model with additive noise. A repulsive
    interaction is added to the x population.

    Parameters
    ----------
    N : int
        Number of neurons in the x population.
    Nsteps : int
        Number of time steps to simulate.
    dt : float
        Time step size.
    x0 : array
        Initial condition for the x population.
    y0 : array
        Initial condition for the y population.
    k : float
        Repulsive interaction strength.
    epsilon : float
        Interaction strength between the x population and
        the mean of the x population.
    mu : float
        Self-coupling of the x population.
    muHat : float
        Coupling from the y to the x population.
    delta : float
        Nonlinearity of the x population.
    muI : float
        External input to the x population.
    alpha : float
        Coupling from the x to the y population.
    beta : float
        Self-coupling of the y population.
    gamma : float
        External input to the y population.
    sigma : float
        Noise level.

    Returns
    -------
    x : array
        Simulated x population.
    y : array
        Simulated y population.
    """

    x = np.zeros((Nsteps, N), dtype=np.float64)
    y = np.zeros((Nsteps, N), dtype=np.float64)

    x[0] = x0
    y[0] = y0
    sqdt = np.sqrt(dt)

    for t in range(1, Nsteps):
        x[t] = x[t-1] + dt * (mu*x[t-1] + delta*x[t-1]**3 - muHat*y[t-1] + muI + epsilon*np.mean(x[t-1]) + k*utils.difference_all_points(x[t-1]))
        x[t] += sigma*sqdt*np.random.randn(N)

        y[t] = y[t-1] + dt * alpha*(x[t-1] - beta*y[t-1] + gamma)
        y[t] += sigma*sqdt*np.random.randn(N)

    return x, y




##############################
# FUNCTIONS FOR RETURN TIMES #
##############################

@nb.njit
def simulate_return_single_FH(Nsteps, dt, x0, y0,
                              epsilon, xcycle, ycycle, tol = 1e-3,
                              mu = 1, muHat = 1, delta = -1/3,
                              muI = 0.6, alpha = 0.1,
                              beta = 0.8, gamma = 0.7, sigma = 0.1):
    
    N = len(x0)
    x = x0
    y = y0
    sqdt = np.sqrt(dt)
    
    t_return = np.zeros(N, dtype = np.float64)
    returned = 0

    for t in range(1, Nsteps):
        xnew = x + dt * (mu*x + delta*x**3 - muHat*y + muI + epsilon*np.mean(x)) + sigma*sqdt*np.random.randn(N)
        ynew = y + dt * alpha*(x - beta*y + gamma) + sigma*sqdt*np.random.randn(N)

        for i in range(N):
            if t_return[i] == 0:
                dist_cycle = utils.distance_from_cycle(xnew[i], ynew[i], xcycle, ycycle)
                if dist_cycle < tol:
                    t_return[i] = t*dt
                    returned += 1

        if returned != N:
            x = xnew
            y = ynew
        else:
            break

    return t_return



@nb.njit(parallel = True)
def find_return_times_single(Nrep, N, Nsteps, dt, cov0, mux0, muy0,
                             epsilon, xcycle, ycycle, tol = 1e-3,
                             mu = 1, muHat = 1, delta = -1/3,
                             muI = 0.6, alpha = 0.1,
                             beta = 0.8, gamma = 0.7, sigma = 0.1):
    
    t = np.zeros((Nrep, N), dtype=np.float64)
    
    for idx in nb.prange(Nrep):
        x0, y0 = utils.sample_multivariate_normal(np.array([mux0, muy0]), cov0, N)
        t[idx] = simulate_return_single_FH(Nsteps, dt, x0, y0,
                                           epsilon, xcycle, ycycle, tol,
                                           mu = mu, muHat = muHat, delta = delta,
                                           muI = muI, alpha = alpha,
                                           beta = beta, gamma = gamma, sigma = sigma)
        
    return t



@nb.njit
def simulate_return_FH_mean(Nsteps, dt, x0, y0,
                            epsilon, xcycle, ycycle, tol = 1e-3,
                            mu = 1, muHat = 1, delta = -1/3,
                            muI = 0.6, alpha = 0.1,
                            beta = 0.8, gamma = 0.7, sigma = 0.1):

    N = len(x0)
    x = x0
    y = y0
    sqdt = np.sqrt(dt)
    
    for t in range(1, Nsteps):
        xnew = x + dt * (mu*x + delta*x**3 - muHat*y + muI + epsilon*np.mean(x)) + sigma*sqdt*np.random.randn(N)
        ynew = y + dt * alpha*(x - beta*y + gamma) + sigma*sqdt*np.random.randn(N)

        dist_cycle = utils.distance_from_cycle(np.mean(xnew), np.mean(ynew), xcycle, ycycle)

        if dist_cycle > tol:
            x = xnew
            y = ynew
        else:
            break

    return t*dt, x, y



@nb.njit(parallel = True)
def find_return_times_mean(Nrep, N, Nsteps, dt, cov0, mux0, muy0,
                           epsilon, xcycle, ycycle, tol = 1e-3,
                           mu = 1, muHat = 1, delta = -1/3,
                           muI = 0.6, alpha = 0.1,
                           beta = 0.8, gamma = 0.7, sigma = 0.1):
    
    t = np.zeros(Nrep)
    
    for idx in nb.prange(Nrep):
        x0, y0 = utils.sample_multivariate_normal(np.array([mux0, muy0]), cov0, N)
        t[idx], _, _ = simulate_return_FH_mean(Nsteps, dt, x0, y0,
                                               epsilon, xcycle, ycycle, tol,
                                               mu = mu, muHat = muHat, delta = delta,
                                               muI = muI, alpha = alpha,
                                               beta = beta, gamma = gamma, sigma = sigma)
        
    return t