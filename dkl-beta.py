import numpy as np
import matplotlib.pyplot as plt
from anesthetic.examples.perfect_ns import planck_gaussian
from scipy.interpolate import interp1d

samples = planck_gaussian()
beta = np.logspace(-10, 0, 100)
D_KL = samples.D_KL(beta=beta)
schedule = interp1d(D_KL, beta, fill_value='extrapolate')
D_KL = np.linspace(0, D_KL.max(), 50)
beta = schedule(D_KL)
plt.plot(beta, D_KL, '.')
plt.xscale('log')
plt.show()

print(samples['logL'].values[-3:])
prior = samples.prior()
print(prior['logL'].values[-3:])
"""print(samples.values[:3, :6])
print(prior.values[:3, :6])"""