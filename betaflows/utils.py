from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import numpy as np

def get_beta_schedule(samples, n, plot=False):
    beta = np.logspace(-6, 0, 100) # will had lower limit as 1e-10 but 1e-6 is probably okay?
    D_KL = samples.D_KL(beta=beta)
    D_KLi = np.linspace(0, D_KL.max(), n)
    beta = np.interp(D_KLi, D_KL, beta)
    # in my code I discard the first beta value as it looks like an outlier... need
    # to talk to Will...

    if plot:
        plt.plot(beta, D_KLi, '.')
        plt.xscale('log')
        plt.show()
    return beta

def approx_uniform_prior_bounds(samples, nDims):
    """
    Function to approximate prior bounds from a set of samples on a posterior.
    Assumes the prior was uniform.

    returns:
    a: upper bound of prior
    b: lower bound of prior
    """
    sample_weights = samples.get_weights()
    n = np.sum(sample_weights)**2 / \
            np.sum(sample_weights**2)

    theta_max = np.max(samples.values[:, :nDims], axis=0)
    theta_min = np.min(samples.values[:, :nDims], axis=0)
    a = ((n-2)*theta_max-theta_min)/(n-3)
    b = ((n-2)*theta_min-theta_max)/(n-3)
    return a, b