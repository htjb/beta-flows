import numpy as np
from lsbi.stats import mixture_normal
from anesthetic import MCMCSamples, read_chains
import os
import matplotlib.pyplot as plt
import pypolychord
from pypolychord.priors import UniformPrior
from pypolychord.settings import  PolyChordSettings

def create_mixture(nmixtures, ndims):
    logws = [0]*nmixtures
    means = np.random.uniform(-10, 10, (nmixtures, ndims))

    covs = []
    for i in range(nmixtures):
        A = np.random.uniform(-3, 3, (ndims, ndims))
        covs.append(np.dot(A, A.transpose()))
    covs = np.array(covs)

    dist = mixture_normal(logws, means, covs)
    return dist, covs, means

def create_likelihood(dist):
    def likelihood(theta):
        log_prob = dist.logpdf(theta)
        return log_prob, 0
    return likelihood

def prior(cube):
    theta = np.zeros_like(cube)
    for i in range(len(cube)):
        theta[i] = UniformPrior(-25, 25)(cube[i])
    return theta

base_dir = 'mixture_models/'
if not os.path.exists(base_dir):
    os.mkdir(base_dir)

ndims = [2, 3, 4, 5, 6, 7, 8]
nmixtures = 5

for i, n in enumerate(ndims):
    dist, cov, means = create_mixture(nmixtures, n)
    likelihood = create_likelihood(dist)

    settings = PolyChordSettings(n, 0) #settings is an object
    settings.base_dir = base_dir + f'ndims_{n}_nmixtures_{nmixtures}/'
    settings.read_resume = False

    if not os.path.exists(settings.base_dir):
        os.mkdir(settings.base_dir)

    dist_samples = dist.rvs(1000)
    samples = MCMCSamples(dist_samples)
    samples.plot_2d(np.arange(0, n))
    plt.savefig(settings.base_dir + 'example_samples_from_mixture.png', dpi=300, bbox_inches='tight')
    plt.close()

    np.save(settings.base_dir + 'true_means.npy', means)
    np.save(settings.base_dir + 'true_covs.npy', cov)

    output = pypolychord.run_polychord(likelihood, n, 0, settings, prior)

    samples = read_chains(settings.base_dir + 'test')
    axes = samples.plot_2d(np.arange(0, n))
    plt.savefig(settings.base_dir + 'polychord_sampled_corner.png', dpi=300, bbox_inches='tight')
    plt.close()