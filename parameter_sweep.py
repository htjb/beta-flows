import tensorflow as tf
import tensorflow_probability as tfp
from anesthetic.plot import kde_contour_plot_2d
import numpy as np
import matplotlib.pyplot as plt
from anesthetic import read_chains
from betaflows.betaflows import BetaFlow
from betaflows.utils import get_beta_schedule, approx_uniform_prior_bounds

nbeta_samples = [10, 25, 50, 100, 250]
LOAD_BETA_FLOW = True
LOAD_MAF = True
PLOT_FLOW = False
PLOT_LIKE =False
PLOT_HIST_LIKE = False
PLOT_IMSHOWS = True
NUMBER_NETS = [2, 4, 6]
HIDDEN_LAYERS = [[50], [50, 50]]
ndims = 2

chains_dir = 'rosenbrock/'
ns = read_chains(chains_dir + 'test')

base_dir = 'parameter_sweep/'
import os
if not os.path.exists(base_dir):
    os.makedirs(base_dir)

prior_bounds = approx_uniform_prior_bounds(ns, ndims)
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
        s = ns.set_beta(b)
        theta.append(s.values[:, :ndims])
        sample_weights.append(s.get_weights())
        beta_values.append([b]*len(s.values))
    theta = np.concatenate(theta).astype(np.float32)
    sample_weights = np.concatenate(sample_weights).astype(np.float32)
    conditional = np.concatenate(beta_values).astype(np.float32)

    if LOAD_BETA_FLOW:
        bflow = BetaFlow.load(base_dir + f'beta_flow_nbeta_{params[i][0]}_nns_{params[i][1]}_' +
                              f'hls_{params[i][2]}.pkl')
    else:
        bflow = BetaFlow(theta, weights=sample_weights, 
            number_networks=params[i][1], hidden_layers=params[i][2],)
        bflow.training(conditional, epochs=5000,
                        loss_type='sum', early_stop=True)
        bflow.save(base_dir + f'beta_flow_nbeta_{params[i][0]}_nns_{params[i][1]}_' +
                              f'hls_{params[i][2]}.pkl')

    if PLOT_FLOW:
        fig, axes = plt.subplots(2, 6, figsize=(10, 6))
        test_beta = [0.01, 0.1, 0.3, 0.5, 0.7, 1]
        for f in range(len(test_beta)):
            nsamplesGen = len(ns.values)
            samples = bflow.sample(10000, test_beta[f])
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

mafparams = list(itertools.product(NUMBER_NETS, HIDDEN_LAYERS))

for i in range(len(mafparams)):
    if LOAD_MAF:
        flow = MAF.load(base_dir + f'normal_flow_nns_{mafparams[i][0]}_' +
                        f'hls_{mafparams[i][1]}.pkl')
    else:
        flow = MAF(values, weights=weights, 
            number_networks=params[i][1],
            hidden_layers=params[i][2])
        flow.train(5000, early_stop=True)
        flow.save(base_dir + f'normal_flow_nns_{mafparams[i][0]}_' +
                        f'hls_{mafparams[i][1]}.pkl')

fdiff_nflp, fdiff_cnflp = [], []
for i in range(len(params)):

    bflow = BetaFlow.load(base_dir + f'beta_flow_nbeta_{params[i][0]}_nns_{params[i][1]}_' +
                              f'hls_{params[i][2]}.pkl')
    flow = MAF.load(base_dir + f'normal_flow_nns_{params[i][1]}_' +
                        f'hls_{params[i][2]}.pkl')
    
    nflp = flow.log_prob(values).numpy()
    cnflp = bflow.log_prob(values, 1)

    mask = np.isfinite(nflp) & np.isfinite(cnflp)
    nflp = nflp[mask]
    cnflp = cnflp[mask]
    weights = weights[mask]

    posterior_probs = ns['logL'] + prior_log_prob - ns.stats(1000)['logZ'].mean()
    posterior_probs = posterior_probs.values[mask]

    if PLOT_LIKE:
        fig, axes = plt.subplots(1, 2, figsize=(6.3, 3), sharey=True)

        axes[0].scatter(posterior_probs, nflp, marker='+', c=posterior_probs, cmap='viridis_r')
        axes[1].scatter(posterior_probs, cnflp, marker='*', c=posterior_probs, cmap='viridis_r')

        for j in range(2):
            axes[j].plot(posterior_probs, posterior_probs, linestyle='--', color='k')
            axes[j].set_xlabel('True log-posterior')
            axes[j].set_ylabel('Flow log-posterior')
            axes[j].legend()
        axes[0].set_title('Normal Flow')
        axes[1].set_title('CNF')
        plt.suptitle(f'nbeta={params[i][0]}, nns={params[i][1]}, hls={params[i][2]}')
        plt.tight_layout()
        plt.savefig(base_dir + f'likleihood_comparison__nbeta_{params[i][0]}_nns_{params[i][1]}_' +
                                f'hls_{params[i][2]}.png', dpi=300)
        plt.show()

    if PLOT_HIST_LIKE:
        fig, axes = plt.subplots(1, 1, figsize=(4, 4), sharey=True)

        axes.hist(posterior_probs, bins=20, alpha=0.5, label='True log-posterior', color='b', histtype='step')
        axes.hist(nflp, bins=20, alpha=0.5, label='Normal Flow log-posterior', color='r', histtype='step')
        axes.hist(cnflp, bins=20, alpha=0.5, label='CNF log-posterior', color='g', histtype='step')
        
        axes.set_xlabel('Log-posterior')
        axes.set_ylabel('Frequency')
        axes.legend()
        axes.set_yscale('log')
        plt.suptitle(f'nbeta={params[i][0]}, nns={params[i][1]}, hls={params[i][2]}')
        plt.tight_layout()
        plt.savefig(base_dir + f'hist_likleihood_comparison_nbeta_{params[i][0]}_nns_{params[i][1]}_' +
                                f'hls_{params[i][2]}.png', dpi=300)
        plt.show()

    fractional_diff_nflp = np.mean(np.abs(1 - np.exp(nflp - posterior_probs)))
    fractional_diff_cnflp = np.mean(np.abs(1 - np.exp(cnflp - posterior_probs)))
    fdiff_cnflp.append(fractional_diff_cnflp)
    fdiff_nflp.append(fractional_diff_nflp)

fdiff_cnflp = np.array(fdiff_cnflp)
fdiff_nflp = np.array(fdiff_nflp)

grid_cnflp = []
grid_nflp = []
for i in range(len(nbeta_samples)):
    nbeta_step_cnflp = []
    nbeta_step_nflp = []
    for k in range(len(NUMBER_NETS)):
        for j in range(len(HIDDEN_LAYERS)):
            if j == 0:
                nbeta_step_cnflp.append(fdiff_cnflp[i+k+j])
                nbeta_step_nflp.append(fdiff_nflp[i+k+j])
    grid_cnflp.append(nbeta_step_cnflp)
    grid_nflp.append(nbeta_step_nflp)

if PLOT_IMSHOWS:
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    im = axes[0].imshow(grid_nflp, cmap='viridis', interpolation='none')
    axes[0].set_xticks(np.arange(len(NUMBER_NETS)))
    axes[0].set_yticks(np.arange(len(nbeta_samples)))
    axes[0].set_xticklabels(NUMBER_NETS)
    axes[0].set_yticklabels(nbeta_samples)
    axes[0].set_xlabel('Repeats')
    axes[0].set_ylabel('Number of beta samples')
    axes[0].set_title('Normal Flow')

    im = axes[1].imshow(grid_cnflp, cmap='viridis', interpolation='none')
    axes[1].set_xticks(np.arange(len(NUMBER_NETS)))
    axes[1].set_yticks(np.arange(len(nbeta_samples)))
    axes[1].set_xticklabels(NUMBER_NETS)
    axes[1].set_yticklabels(nbeta_samples)
    axes[1].set_xlabel('Number Nets')
    axes[1].set_ylabel('Number of beta samples')
    axes[1].set_title('CNF')

    plt.tight_layout()
    [plt.colorbar(im, ax=axes[i]) for i in range(2)]
    plt.savefig(base_dir + 'fractional_diff.png', dpi=300)
    plt.show()
