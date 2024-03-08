import tensorflow as tf
import tensorflow_probability as tfp
from margarine.processing import (_forward_transform, _inverse_transform,
                                  pure_tf_train_test_split)
from anesthetic.plot import kde_contour_plot_2d
import numpy as np
import matplotlib.pyplot as plt
from anesthetic import read_chains
from betaflows.betaflows import BetaFlow
from betaflows.utils import get_beta_schedule

# define tensorflow stuff
tfk = tf.keras
tfkl = tf.keras.layers
tfpl = tfp.layers
tfd = tfp.distributions
tfb = tfp.bijectors

nbeta_samples = 20
LOAD = False

base_dir = 'ns_run/'

ns = read_chains(base_dir + 'test')

# beta schedule
beta = get_beta_schedule(ns, nbeta_samples)
print(beta)

theta, sample_weights, beta_values = [], [], []
for i,b in enumerate(beta):
    s = ns.set_beta(b)
    theta.append(s.values[:, :2])
    sample_weights.append(s.get_weights())
    beta_values.append([b]*len(s.values))
theta = np.concatenate(theta).astype(np.float32)
sample_weights = np.concatenate(sample_weights).astype(np.float32)
conditional = np.concatenate(beta_values).astype(np.float32)

theta_min = np.min(theta, axis=0)
theta_max = np.max(theta, axis=0)

if LOAD:
    bflow = BetaFlow.load(base_dir + 'beta_flow.pkl')
else:
    bflow = BetaFlow(theta, weights=sample_weights, number_networks=2,)
    bflow.training(conditional, epochs=10000,
                    loss_type='mean', early_stop=True)
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
    flow = MAF(values, weights=weights, number_networks=2)
    flow.train(5000, early_stop=True)
    flow.save(base_dir + 'normal_flow.pkl')

print(bflow.mades[0]._network.summary())
from tensorflow import keras
keras.utils.plot_model(bflow.mades[0]._network, "conditional_made.png", show_shapes=True)
print(flow.mades[0]._network.summary())
keras.utils.plot_model(flow.mades[0]._network, "normal_made.png", show_shapes=True)


nflp = flow.log_prob(values).numpy()
cnflp = bflow.log_prob(values, 1)

from scipy.special import logsumexp
posterior_probs = ns['logL'] + (ns.logw() - logsumexp(ns.logw())) - ns.stats(1000)['logZ'].mean()
posterior_probs = posterior_probs.values[mask]

print('Normal Flow Average Like: ', np.average(nflp, weights=weights))
print('CNF Average Like: ', np.average(cnflp, weights=weights))

fig, axes = plt.subplots(2, 3, figsize=(10, 10))

for j in range(2):
    axes[j, 0].scatter(posterior_probs, nflp, marker='+', c=posterior_probs, cmap='viridis_r')
    axes[j, 1].scatter(posterior_probs, cnflp, marker='*', c=posterior_probs, cmap='viridis_r')
for i in range(2):
    for j in range(2):
        axes[j, i].plot(posterior_probs, posterior_probs, linestyle='--', color='k')
        axes[j, i].set_xlabel('True log-posterior')
        axes[j, i].set_ylabel('Flow log-posterior')
        axes[j, i].legend()
        #axes[j, i].text(0.25, 0., 'Over Predict', transform=axes[j, i].transAxes, rotation=45)
        #axes[j, i].text(0.25, 0.2, 'Under Predict', transform=axes[j, i].transAxes, rotation=45)
    #axes[0, i].set_xlim([posterior_probs.min(), posterior_probs.max()])
    #axes[0, i].set_ylim([posterior_probs.min(), posterior_probs.max()])

axes[0, 2].scatter(values[:, 0], values[:, 1], c=posterior_probs, cmap='viridis_r', marker='.')
axes[0, 2].set_title('Samples')
axes[0, 2].set_xlabel(r'$\theta_1$')
axes[0, 2].set_ylabel(r'$\theta_2$')
axes[1, 2].axis('off')
axes[0, 0].set_title('Normal margarine')
axes[0, 1].set_title('beta flow')
plt.tight_layout()
plt.savefig(base_dir + 'likleihood_comparison.png', dpi=300)
plt.show()