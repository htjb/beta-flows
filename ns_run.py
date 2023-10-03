"""
Code to generate some nested samples for testing the beta flows against.
"""

import numpy as np
import matplotlib.pyplot as plt
from anesthetic import read_chains
import pypolychord
from pypolychord.priors import UniformPrior, GaussianPrior
from pypolychord.settings import  PolyChordSettings

y = np.array([0.9, 1])
    
def prior(cube):
    theta = np.ones_like(cube)
    theta[0] = UniformPrior(-3, 2)(cube[0])
    theta[1] = UniformPrior(0, 3)(cube[1])
    return theta

def likelihood(theta):
    parameters = theta.copy()
    noise=0.1
    loglikelihood = (-0.5*np.log(2*np.pi*(noise**2))-0.5 \
        *((y - [parameters[0]*parameters[1]**2,
                 parameters[1]/2]) / noise)**2).sum()
    return loglikelihood, 0

nDims = 2

settings = PolyChordSettings(nDims, 0) #settings is an object
settings.base_dir = 'ns_run/' #string
settings.read_resume = False

output = pypolychord.run_polychord(likelihood, nDims, 0, settings, prior)
paramnames = [('p%i' % i, r'\theta_%i' % i) for i in range(nDims)]
output.make_paramnames_files(paramnames)

from anesthetic import read_chains

samples = read_chains('ns_run/test')
print(samples)
names = ['p0', 'p1']
axes = samples.plot_2d(names)
plt.savefig('ns_run.png')
plt.show()