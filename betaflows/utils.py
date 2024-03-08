from scipy.interpolate import interp1d
import numpy as np

def get_beta_schedule(samples, n):
    beta = np.logspace(-10, 0, 100)
    D_KL = samples.D_KL(beta=beta)
    schedule = interp1d(D_KL, beta, fill_value='extrapolate')
    D_KL = np.linspace(0, D_KL.max(), n)
    beta = schedule(D_KL)
    return beta