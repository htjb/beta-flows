import numpy as np
import matplotlib.pyplot as plt
from anesthetic import read_chains
import pypolychord
from pypolychord.priors import UniformPrior
from pypolychord.settings import  PolyChordSettings


def likelihood(theta):
    log_prob = -np.log(1 + (1-theta[0])**2 + 100*(theta[1] - theta[0]**2)**2)
    return log_prob, 0

def prior(cube):
    theta = np.zeros_like(cube)
    theta[0] = UniformPrior(-5, 5)(cube[0])
    theta[1] = UniformPrior(-5, 5)(cube[1])
    return theta

nDims = 2

settings = PolyChordSettings(nDims, 0) #settings is an object
settings.base_dir = 'rosenbrock/' #string
settings.read_resume = False
settings.nlive = 50 # should be about 25*nDims or really low for testing

output = pypolychord.run_polychord(likelihood, nDims, 0, settings, prior)
paramnames = [('p%i' % i, r'\theta_%i' % i) for i in range(nDims)]
output.make_paramnames_files(paramnames)

from anesthetic import read_chains

samples = read_chains('rosenbrock/test')
print(samples)
names = ['p0', 'p1']
axes = samples.plot_2d(names)
plt.show()