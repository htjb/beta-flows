import tensorflow as tf
import tensorflow_probability as tfp
from anesthetic.plot import kde_contour_plot_2d
import numpy as np
import matplotlib.pyplot as plt
from anesthetic import read_chains
from betaflows.betaflows import BetaFlow
from betaflows.utils import get_beta_schedule, approx_uniform_prior_bounds
import os

nbeta_samples = 25
nmixtures = 5
ndims = 6
NUMBER_NETS = 2
HIDDEN_LAYERS = [50]
bepochs = 10000
mepochs = 10000
LOAD = True
KDE = False

base_dir = 'figures/'
if not os.path.exists(base_dir):
    os.makedirs(base_dir)

print("ndims = ", 2)
chains = read_chains(f'mixture_models/ndims_{ndims}_nmixtures_{nmixtures}/test')

beta = get_beta_schedule(chains, nbeta_samples)

theta, sample_weights, beta_values = [], [], []
for j,b in enumerate(beta):
    if j > 0:
        s = chains.set_beta(b)
        theta.append(s.values[:, :2])
        sample_weights.append(s.get_weights()/s.get_weights().sum())
        beta_values.append([b]*len(s.get_weights()))
theta = np.concatenate(theta).astype(np.float32)
sample_weights = np.concatenate(sample_weights).astype(np.float32)
conditional = np.concatenate(beta_values).astype(np.float32)

if LOAD:
    bflow = BetaFlow.load(base_dir + f'figure_1_beta_flow.pkl')
else:
    bflow = BetaFlow(theta, weights=sample_weights, 
                    theta_min=np.array([-25]*2).astype(np.float32),
                    theta_max=np.array([25]*2).astype(np.float32),
        number_networks=NUMBER_NETS, hidden_layers=HIDDEN_LAYERS,)
    bflow.training(conditional, epochs=bepochs,
                    early_stop=True, patience=250)
    bflow.save(base_dir + f'figure_1_beta_flow.pkl')

from margarine.maf import MAF

values = chains.values[:, :2]
weights = chains.get_weights()

if LOAD:
    flow = MAF.load(base_dir + f'figure_1_normal_flow.pkl')
else:
    flow = MAF(values, weights=weights, 
            theta_min=np.array([-25]*2).astype(np.float32), 
            theta_max=np.array([25]*2).astype(np.float32), 
        number_networks=NUMBER_NETS,
        hidden_layers=HIDDEN_LAYERS)
    flow.train(mepochs, early_stop=True)
    flow.save(base_dir + f'figure_1_normal_flow.pkl')

prior_log_prob = np.log(np.prod([1/50]*2))

posterior_probs = chains['logL'] + prior_log_prob - chains.stats(1000)['logZ'].mean()

nflp = flow.log_prob(values).numpy()
cnflp = bflow.log_prob(values, 1)

mask = np.isfinite(nflp) & np.isfinite(cnflp)
nflp = nflp[mask]
cnflp = cnflp[mask]
weights = weights[mask]

pps = posterior_probs.values[mask]

nflp_error = np.average(pps - nflp, weights=weights)
cnflp_error = np.average(pps - cnflp, weights=weights)

print('nflp: ', nflp_error)
print('cnflp: ', cnflp_error)


fig, axes = plt.subplots(3, 4, figsize=(6.3, 4), sharex=True, sharey=True)
ax = axes.flatten()

for i in range(len(ax)):
    ax[i].set_yticks([])
    ax[i].set_xticks([])
ax[8].axis('off')
ax[11].axis('off')

cnflp_samples = bflow.sample(len(chains.values), 1).numpy()
print(cnflp_samples)
if KDE:
    kde_contour_plot_2d(ax[0], chains.values[:, 0], chains.values[:, 1], weights=chains.get_weights())
    kde_contour_plot_2d(ax[7], cnflp_samples[:, 0], cnflp_samples[:, 1])
    kde_contour_plot_2d(ax[9], cnflp_samples[:, 0], cnflp_samples[:, 1])
else:
    ax[3].hist2d(chains.values[:, 0], chains.values[:, 1], weights=chains.get_weights(), bins=50)
    ax[7].hist2d(cnflp_samples[:, 0], cnflp_samples[:, 1], bins=50)
    ax[9].hist2d(cnflp_samples[:, 0], cnflp_samples[:, 1], bins=50)

nflp_samples = flow.sample(len(chains.values)).numpy()
ax[10].hist2d(nflp_samples[:, 0], nflp_samples[:, 1], bins=50)
ax[10].set_title('MAF\n ' + r'$D_{KL}=$' + str(np.round(nflp_error, 2)))

ax[9].set_title(r'$\beta$-' + 'Flow\n ' + r'$D_{KL}=$' + str(np.round(cnflp_error, 2)))

test_beta = [0.01, 0.4, 0.75]
for i, tb in enumerate(test_beta):
    axes[0, i].set_title(r'$\beta=$'+str(tb))
    axes[1, i].set_title(r'$\beta=$'+str(tb))
    s = chains.set_beta(tb)
    if KDE:
        kde_contour_plot_2d(axes[0, i], s.values[:, 0], s.values[:, 1], weights=s.get_weights())
    else:
        axes[0, i].hist2d(s.values[:, 0], s.values[:, 1], weights=s.get_weights(), bins=50)
    cnflp_samples = bflow.sample(len(chains.values), tb).numpy()
    if KDE:
        kde_contour_plot_2d(axes[1, i], cnflp_samples[:, 0], cnflp_samples[:, 1])
    else:
        axes[1, i].hist2d(cnflp_samples[:, 0], cnflp_samples[:, 1], bins=50)

axes[0, 0].set_ylabel('Truth')
axes[1, 0].set_ylabel(r'$\beta$-' + 'Flow')
axes[0, 3].set_title(r'$\beta=1$')
axes[1, 3].set_title(r'$\beta=1$')
plt.tight_layout()
plt.show()