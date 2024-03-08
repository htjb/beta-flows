import tensorflow as tf
import tensorflow_probability as tfp
from anesthetic.plot import kde_contour_plot_2d
import numpy as np
import matplotlib.pyplot as plt
from anesthetic import read_chains
from betaflows.betaflows import BetaFlow
from betaflows.utils import get_beta_schedule, approx_uniform_prior_bounds

# define tensorflow stuff
tfk = tf.keras
tfkl = tf.keras.layers
tfpl = tfp.layers
tfd = tfp.distributions
tfb = tfp.bijectors

nbeta_samples = 200
LOAD = False
NUMBER_NETS = 2
HIDDEN_LAYERS = [50]

base_dir = 'ns_run/'

ns = read_chains(base_dir + 'test')

prior_bounds = approx_uniform_prior_bounds(ns, 2)
print(prior_bounds)
bounds = []
for i in range(2):
    bounds.append([round(prior_bounds[0][i], 1), round(prior_bounds[1][i], 1)])
bounds = np.array(bounds)
print(bounds)
prior_log_prob = np.log(np.prod((1/(bounds[:, 1] - bounds[:, 0]))))

# beta schedule
beta = get_beta_schedule(ns, nbeta_samples)
#print(beta)
#print(1-beta)
#beta = 1- beta
"""plt.hist(beta, bins=20, histtype='step', label='beta')
plt.hist(1-beta, bins=20, histtype='step', label='1-beta')
plt.legend()
plt.show()
sys.exit(1)"""


import random
theta, sample_weights, beta_values = [], [], []
for i,b in enumerate(beta):
    s = ns.set_beta(b)
    idx = random.sample(range(len(s.values)), 50)
    idx = np.arange(len(s.values))
    theta.append(s.values[:, :2][idx])
    sample_weights.append(s.get_weights()[idx])
    beta_values.append([b]*len(idx))
theta = np.concatenate(theta).astype(np.float32)
sample_weights = np.concatenate(sample_weights).astype(np.float32)
conditional = np.concatenate(beta_values).astype(np.float32)

if LOAD:
    bflow = BetaFlow.load(base_dir + 'beta_flow.pkl')
else:
    bflow = BetaFlow(theta, weights=sample_weights, 
        number_networks=NUMBER_NETS, hidden_layers=HIDDEN_LAYERS,)
    bflow.training(conditional, epochs=10000,
                    loss_type='sum', early_stop=True)
    bflow.save(base_dir + 'beta_flow.pkl')


fig, axes = plt.subplots(2, 6, figsize=(10, 6))
test_beta = [0.01, 0.1, 0.3, 0.5, 0.7, 1]
for i in range(len(test_beta)):
    nsamplesGen = len(ns.values)
    samples = bflow.sample(10000, test_beta[i])
    samples = samples.numpy()

    s = ns.set_beta(test_beta[i])
    try:
        kde_contour_plot_2d(axes[0, i], s.values[:, 0], s.values[:, 1], weights=s.get_weights())
        kde_contour_plot_2d(axes[1, i], samples[:, 0], samples[:, 1])
    except:
        axes[0, i].hist2d(s.values[:, 0], s.values[:, 1], weights=s.get_weights(), bins=50)
        axes[1, i].hist2d(samples[:, 0], samples[:, 1], bins=50)

ax = axes.flatten()
for i in range(len(ax)):
    ax[i].set_xlim([0, 2])
    ax[i].set_ylim([0, 2])
    ax[i].set_xlabel(r'$\theta_1$')
ax[0].set_ylabel('Truth\n' + r'$\theta_2$')
ax[5].set_ylabel('Flow\n' + r'$\theta_2$')
for i in range(len(test_beta)):
    ax[i].set_title(r'$\beta=$'+str(test_beta[i]))
plt.tight_layout()
plt.savefig(base_dir + 'beta_flow.png', dpi=300)
plt.show()

############## Train beta=1 flow (normal margarine) ############

from margarine.maf import MAF

values = ns.values[:, :2]
mask = np.isfinite(values[:, 0]) & np.isfinite(values[:, 1])
values = values[mask]
weights = ns.get_weights()[mask]

if LOAD:
    flow = MAF.load(base_dir + 'normal_flow.pkl')
else:
    flow = MAF(values, weights=weights, 
        number_networks=NUMBER_NETS,
        hidden_layers=HIDDEN_LAYERS)
    flow.train(5000, early_stop=True)
    flow.save(base_dir + 'normal_flow.pkl')

#print(bflow.mades[0]._network.summary())
#from tensorflow import keras
#keras.utils.plot_model(bflow.mades[0]._network, "conditional_made.png", show_shapes=True)
#print(flow.mades[0]._network.summary())
#keras.utils.plot_model(flow.mades[0]._network, "normal_made.png", show_shapes=True)

nflp = flow.log_prob(values).numpy()
cnflp = bflow.log_prob(values, 1)

from scipy.special import logsumexp
posterior_probs = ns['logL'] + prior_log_prob - ns.stats(1000)['logZ'].mean()
posterior_probs = posterior_probs.values[mask]

print('Normal Flow Average Like: ', np.average(nflp, weights=weights))
print('CNF Average Like: ', np.average(cnflp, weights=weights))
print('NF Difference: ', logsumexp(posterior_probs, -nflp))
print('CNF Difference: ', logsumexp(posterior_probs, -cnflp))

fig, axes = plt.subplots(1, 3, figsize=(6.3, 3))

axes[0].scatter(posterior_probs, nflp, marker='+', c=posterior_probs, cmap='viridis_r')
axes[1].scatter(posterior_probs, cnflp, marker='*', c=posterior_probs, cmap='viridis_r')

for i in range(2):
    axes[i].plot(posterior_probs, posterior_probs, linestyle='--', color='k')
    axes[i].set_xlabel('True log-posterior')
    axes[i].set_ylabel('Flow log-posterior')
    axes[i].legend()
       
axes[2].scatter(values[:, 0], values[:, 1], c=posterior_probs, cmap='viridis_r', marker='.')
axes[2].set_title('Samples')
axes[2].set_xlabel(r'$\theta_1$')
axes[2].set_ylabel(r'$\theta_2$')
axes[0].set_title('Normal margarine')
axes[1].set_title('beta flow')
plt.tight_layout()
plt.savefig(base_dir + 'likleihood_comparison.png', dpi=300)
plt.show()