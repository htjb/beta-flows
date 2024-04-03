import tensorflow as tf
import tensorflow_probability as tfp
from anesthetic.plot import kde_contour_plot_2d
import numpy as np
import matplotlib.pyplot as plt
from anesthetic import read_chains
from betaflows.betaflows import BetaFlow
from betaflows.utils import get_beta_schedule, approx_uniform_prior_bounds
import random

nbeta_samples = [10, 25, 50]
LOAD_BETA_FLOW = True
LOAD_MAF = True
PLOT_FLOW = False
PLOT_LIKE = True
PLOT_IMSHOWS = True
NUMBER_NETS = [2, 4, 6, 8]
HIDDEN_LAYERS = [[50]]
ndims = 2
epochs= 10000

chains_dir = 'rosenbrock/'
#chains_dir = 'mixture_models/ndims_4_nmixtures_5/'
ns = read_chains(chains_dir + 'test')

base_dir = 'parameter_sweep/'
import os
if not os.path.exists(base_dir):
    os.makedirs(base_dir)

prior_bounds = [[-5, -5], [5, 5]]

#prior_bounds = approx_uniform_prior_bounds(ns, ndims)

bounds = []
for i in range(ndims):
    bounds.append([round(prior_bounds[0][i], 1), round(prior_bounds[1][i], 1)])
bounds = np.array(bounds)
prior_log_prob = np.log(np.prod((1/(bounds[:, 1] - bounds[:, 0]))))

# make combinations of parameters
import itertools
params = list(itertools.product(nbeta_samples, NUMBER_NETS, HIDDEN_LAYERS))


for i in range(len(params)):
    # beta schedule
    beta = get_beta_schedule(ns, params[i][0])

    theta, sample_weights, beta_values = [], [], []
    for j, b in enumerate(beta):
        if j >0:
            s = ns.set_beta(b)
            theta.append(s.values[:, :ndims])
            sample_weights.append(s.get_weights()/s.get_weights().sum())
            beta_values.append([np.log10(b)]*len(s.values))
    theta = np.concatenate(theta).astype(np.float32)
    sample_weights = np.concatenate(sample_weights).astype(np.float32)
    conditional = np.concatenate(beta_values).astype(np.float32)
    """plt.hist(conditional, bins=100)
    plt.show()
    sys.exit(1)"""

    """idx = random.sample(range(len(theta)), len(ns.values))
    theta = theta[idx]
    sample_weights = sample_weights[idx]
    conditional = conditional[idx]"""

    if LOAD_BETA_FLOW:
        try:
            bflow = BetaFlow.load(base_dir + f'beta_flow_nbeta_{params[i][0]}_nns_{params[i][1]}_' +
                              f'hls_{params[i][2]}.pkl')
        except FileNotFoundError:
            bflow = BetaFlow(theta, weights=sample_weights, 
                             theta_min=np.array([-5.0, -5.0]).astype(np.float32), 
                             theta_max=np.array([5.0, 5.0]).astype(np.float32),
                number_networks=params[i][1], hidden_layers=params[i][2],)
            bflow.training(conditional, epochs=epochs,
                            loss_type='mean', early_stop=True)
            bflow.save(base_dir + f'beta_flow_nbeta_{params[i][0]}_nns_{params[i][1]}_' +
                              f'hls_{params[i][2]}.pkl')
    else:
        bflow = BetaFlow(theta, weights=sample_weights, 
                         theta_min=np.array([-5.0, -5.0]).astype(np.float32), 
                         theta_max=np.array([5.0, 5.0]).astype(np.float32),
            number_networks=params[i][1], hidden_layers=params[i][2],)
        bflow.training(conditional, epochs=epochs,
                        loss_type='mean', early_stop=True)
        bflow.save(base_dir + f'beta_flow_nbeta_{params[i][0]}_nns_{params[i][1]}_' +
                              f'hls_{params[i][2]}.pkl')

    if PLOT_FLOW:
        fig, axes = plt.subplots(2, 6, figsize=(10, 6))
        test_beta = [0.01, 0.1, 0.3, 0.5, 0.7, 1]
        for f in range(len(test_beta)):
            nsamplesGen = len(ns.values)
            samples = bflow.sample(10000, np.log10(test_beta[f]))
            samples = samples.numpy()

            s = ns.set_beta(test_beta[f])
            try:
                kde_contour_plot_2d(axes[0, f], s.values[:, 0], s.values[:, 1], weights=s.get_weights())
                kde_contour_plot_2d(axes[1, f], samples[:, 0], samples[:, 1])
            except:
                axes[0, f].hist2d(s.values[:, 0], s.values[:, 1], weights=s.get_weights(), bins=50)
                axes[1, f].hist2d(samples[:, 0], samples[:, 1], bins=50)

        ax = axes.flatten()
        for j in range(len(ax)):
            ax[j].set_xlabel(r'$\theta_1$')
        ax[0].set_ylabel('Truth\n' + r'$\theta_2$')
        ax[5].set_ylabel('Flow\n' + r'$\theta_2$')
        for j in range(len(test_beta)):
            ax[j].set_title(r'$\beta=$'+str(test_beta[j]))
        plt.tight_layout()
        plt.savefig(base_dir + f'beta_flow_nbeta_{params[i][0]}_nns_{params[i][1]}_' +
                              f'hls_{params[i][2]}.png', dpi=300,
                              bbox_inches='tight')
        #plt.show()
        plt.close()

############## Train beta=1 flow (normal margarine) ############

from margarine.maf import MAF

values = ns.values[:, :ndims]
weights = ns.get_weights()
weights /= weights.sum()

mafparams = list(itertools.product(NUMBER_NETS, HIDDEN_LAYERS))

for i in range(len(mafparams)):
    if LOAD_MAF:
        try:
            flow = MAF.load(base_dir + f'normal_flow_nns_{mafparams[i][0]}_' +
                        f'hls_{mafparams[i][1]}.pkl')
        except FileNotFoundError:
            flow = MAF(values, weights=weights,
                theta_min=np.array([-5.0, -5.0]).astype(np.float32),
                theta_max=np.array([5.0, 5.0]).astype(np.float32), 
                number_networks=params[i][1],
                hidden_layers=params[i][2])
            flow.train(epochs, early_stop=True, loss_type='mean')
            flow.save(base_dir + f'normal_flow_nns_{mafparams[i][0]}_' +
                            f'hls_{mafparams[i][1]}.pkl')
    else:
        flow = MAF(values, weights=weights,
            theta_min=np.array([-5.0, -5.0]).astype(np.float32),
            theta_max=np.array([5.0, 5.0]).astype(np.float32), 
            number_networks=params[i][1],
            hidden_layers=params[i][2])
        flow.train(epochs, early_stop=True, loss_type='mean')
        flow.save(base_dir + f'normal_flow_nns_{mafparams[i][0]}_' +
                        f'hls_{mafparams[i][1]}.pkl')

fdiff_nflp, fdiff_cnflp = [], []
for i in range(len(params)):

    bflow = BetaFlow.load(base_dir + f'beta_flow_nbeta_{params[i][0]}_nns_{params[i][1]}_' +
                              f'hls_{params[i][2]}.pkl')
    flow = MAF.load(base_dir + f'normal_flow_nns_{params[i][1]}_' +
                        f'hls_{params[i][2]}.pkl')
    
    nflp = flow.log_prob(values).numpy()
    #beta = tf.math.log(1.0)
    beta = np.log10(1)
    cnflp = bflow.log_prob(values, beta)

    mask = np.isfinite(nflp) & np.isfinite(cnflp)
    nflp = nflp[mask]
    cnflp = cnflp[mask]
    weights = weights[mask]

    posterior_probs = ns['logL'] + prior_log_prob - ns.stats(1000)['logZ'].mean()
    posterior_probs = posterior_probs.values[mask]

    if PLOT_LIKE:
        fig, axes = plt.subplots(1, 2, figsize=(6.3, 3), sharey=True)

        axes[0].scatter(posterior_probs, nflp, marker='.', c=posterior_probs, cmap='viridis_r')
        axes[1].scatter(posterior_probs, cnflp, marker='.', c=posterior_probs, cmap='viridis_r')

        for j in range(2):
            axes[j].plot(posterior_probs, posterior_probs, linestyle='--', color='k')
            axes[j].set_xlabel('True log-posterior')
            axes[j].set_ylabel('Flow log-posterior')
            axes[j].legend()
            axes[j].set_ylim(-25, 10)

        for j in range(len(nflp)):
            if nflp[j] < -25:
                norm_prob = (posterior_probs[j] - posterior_probs.min())/ \
                        (posterior_probs.max() - posterior_probs.min())
                axes[0].arrow(posterior_probs[j], -23, 
                              0, -0.8, head_width=0.3, 
                              head_length=0.7, 
                              fc=plt.get_cmap('viridis_r')(norm_prob),
                              ec='k',#plt.get_cmap('viridis_r')(norm_prob),
                              lw=0.4)
            if cnflp[j] < -25:
                axes[1].arrow(posterior_probs[j], -23, 
                              0, -0.8, head_width=0.3, 
                              head_length=0.7, 
                              fc=plt.get_cmap('viridis_r')(norm_prob),
                              ec='k',#plt.get_cmap('viridis_r')(norm_prob),
                              lw=0.4)
        
        axes[0].set_title('Normal Flow')
        axes[1].set_title('CNF')
        plt.suptitle(f'nbeta={params[i][0]}, nns={params[i][1]}, hls={params[i][2]}')
        plt.tight_layout()
        plt.savefig(base_dir + f'likleihood_comparison__nbeta_{params[i][0]}_nns_{params[i][1]}_' +
                                f'hls_{params[i][2]}.png', dpi=300)
        #plt.show()
        plt.close()

    fractional_diff_nflp = np.average(posterior_probs - nflp, weights=weights)
    fractional_diff_cnflp = np.average(posterior_probs - cnflp, weights=weights)
    fdiff_cnflp.append(fractional_diff_cnflp)
    fdiff_nflp.append(fractional_diff_nflp)

fdiff_cnflp = np.array(fdiff_cnflp)
fdiff_nflp = np.array(fdiff_nflp)

maximum = np.max([np.max(fdiff_cnflp), np.max(fdiff_nflp)])
minimum = np.min([np.min(fdiff_cnflp), np.min(fdiff_nflp)])

fdiff_cnflp50s = fdiff_cnflp
fdiff_nflp50s = fdiff_nflp
params50 = params
print(fdiff_cnflp, fdiff_nflp)
print(fdiff_cnflp50s, fdiff_nflp50s)

params50 = np.array([[params50[i][0], params50[i][1]] for i in range(len(params50))])
print(params50)
params50 = params50.reshape(2, len(nbeta_samples), len(NUMBER_NETS))
grid_nflp50 = fdiff_nflp50s.reshape(len(nbeta_samples), len(NUMBER_NETS))
grid_cnflp50 = fdiff_cnflp50s.reshape(len(nbeta_samples), len(NUMBER_NETS))

fig, axes = plt.subplots(1, 2, figsize=(8, 4))
axes = axes.flatten()
im1 = axes[0].imshow(grid_nflp50, cmap='viridis', interpolation='none',
                     vmin=minimum, vmax=maximum)
axes[0].set_xticks(np.arange(len(NUMBER_NETS)))
axes[0].set_yticks(np.arange(len(nbeta_samples)))
axes[0].set_xticklabels(NUMBER_NETS)
axes[0].set_yticklabels(np.arange(len(nbeta_samples)))
axes[0].set_xlabel('Number Nets')
axes[0].set_ylabel('Repeats')
axes[0].set_title('Normal Flow')
for i in range(len(nbeta_samples)):
    for j in range(len(NUMBER_NETS)):
        axes[0].text(j, i, '{:.2f}'.format(grid_nflp50[i, j]), 
                     ha='center', va='center', color='k',
                bbox=dict(facecolor='white', lw=0), fontsize=10)
        
plt.colorbar(im1, ax=axes[0], label=r'$D_{KL}$')

im2 = axes[1].imshow(grid_cnflp50, cmap='viridis', interpolation='none',
                     vmin=minimum, vmax=maximum)
axes[1].set_xticks(np.arange(len(NUMBER_NETS)))
axes[1].set_yticks(np.arange(len(nbeta_samples)))
axes[1].set_xticklabels(NUMBER_NETS)
axes[1].set_yticklabels(nbeta_samples)
axes[1].set_xlabel('Number Nets')
axes[1].set_ylabel('Number Beta Samples')
axes[1].set_title('CNF')
plt.colorbar(im2, ax=axes[1], label=r'$D_{KL}$')
for i in range(len(nbeta_samples)):
    for j in range(len(NUMBER_NETS)):
        axes[1].text(j, i, '{:.2f}'.format(grid_cnflp50[i, j]), 
                     ha='center', va='center', color='k',
                bbox=dict(facecolor='white', lw=0), fontsize=10)

plt.tight_layout()
plt.savefig(base_dir + 'fractional_diff.png', dpi=300)
plt.show()

