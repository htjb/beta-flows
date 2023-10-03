import numpy as np
from anesthetic import read_chains
from beta_flow.flow import MAF
import matplotlib.pyplot as plt

# load the test samples in
file = 'gauss_ns_run'
samples = read_chains(file + '/test')
beta = [0.001, 0.25, 1.]
# train the flow
f = MAF(samples, beta=beta, hidden_layers=[50, 50], number_networks=20, lr=5e-3)
f.train(10000, early_stop=True)

plt.plot(f.loss_history)
plt.plot(f.test_loss_history)
plt.show()

# grid over the sampled space
x = np.linspace(samples['p0'].min(), samples['p0'].max(), 100).astype(np.float32)
y = np.linspace(samples['p1'].min(), samples['p1'].max(), 100).astype(np.float32)
xv, yv = np.meshgrid(x, y, indexing='ij')
plotting_order = np.array([xv.flatten(), yv.flatten()]).T

fig, axes = plt.subplots(3, 3, figsize=(10, 5))

for i, b in enumerate(beta[::-1]):
    # plot the actual samples for different betas
    cb = axes[0, i].hist2d(samples['p0'], samples['p1'], 
                            weights=samples.set_beta(b).get_weights(), 
                            cmap='Blues', bins=50, density=True,
                            vmin=0, vmax=samples['logL'].max())
    plt.colorbar(cb[3],fraction=0.046, pad=0.04)
    axes[0, i].set_title(f'Beta={b}')

# define a set of beta for testing
flow_lps = []
for i, b in enumerate(beta[::-1]):
    
    # get the log probability from the beta flow for a given beta
    flow_lps.append(f.log_prob(np.array([xv.flatten(), yv.flatten()]).T, b).numpy())

lpmax = np.max(flow_lps)
for i, b in enumerate(beta[::-1]):
    # plot the meshgrid samples colored based on their log probability as
    # calculated by the beta flow
    cb = axes[1, i].scatter(plotting_order[:, 0], plotting_order[:, 1], c=np.exp(flow_lps[i]),
                            cmap='Blues', s=1, vmin=0, vmax=lpmax)
    plt.colorbar(cb,fraction=0.046, pad=0.04)
    axes[1, i].set_title(f'Beta={b}')

"""for i, b in enumerate(beta[::-1]):

    flow_samples = f.sample(10000, beta=b)
    flow_samples = flow_samples.numpy()
    mask = np.isfinite(flow_samples)
    mask = [m.all() for m in mask]
    flow_samples = flow_samples[mask]

    cb = axes[2, i].hist2d(flow_samples[:, 0], flow_samples[:, 1],
                            bins=50, density=True,
                            cmap='Blues', vmin=0, vmax=lpmax)
    plt.colorbar(cb[3],fraction=0.046, pad=0.04)
    axes[2, i].set_title(f'Beta={b}')"""

"""for j in range(len(axes)):
    for i in range(len(axes[j])):
        axes[j, i].set_xlim(0.5, 2.5)
        axes[j, i].set_ylim(0., 2.5)"""

axes[0, 0].set_ylabel('Nested Sampling')
axes[1, 0].set_ylabel('Beta Flow\nLog Prob')
axes[2, 0].set_ylabel('Beta Flow\nSamples')
plt.tight_layout()
plt.savefig('flow_probs_' + file + '.png', dpi=300)
plt.show()
