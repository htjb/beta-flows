import numpy as np
from anesthetic import read_chains
from beta_flow.flow import MAF
import matplotlib.pyplot as plt

file = 'test_ns_run_complicated_gauss_gauss'
samples = read_chains(file + '/test')
f = MAF(samples)
f.train(10000, early_stop=True)

x = np.linspace(samples['p0'].min(), samples['p0'].max(), 100).astype(np.float32)
y = np.linspace(samples['p1'].min(), samples['p1'].max(), 100).astype(np.float32)
xv, yv = np.meshgrid(x, y, indexing='ij')
plotting_order = np.array([xv.flatten(), yv.flatten()]).T

fig, axes = plt.subplots(3, 5, figsize=(10, 5))
beta = [0.001, 0.1, 0.25, 0.7, 1]
for i, b in enumerate(beta[::-1]):
    
    lp = f.log_prob(np.array([xv.flatten(), yv.flatten()]).T, b).numpy()
    #lp = lp.reshape(xv.shape)
    
    #if b == 1:
    lpmax = np.exp(lp).max()
    if not lpmax > samples['logL'].max():
        lpmax = samples['logL'].max()
    cb = axes[1, i].scatter(plotting_order[:, 0], plotting_order[:, 1], c=np.exp(lp),
                            cmap='Blues', vmin=0, vmax=lpmax, s=1)
    plt.colorbar(cb,fraction=0.046, pad=0.04)
    axes[1, i].set_title(f'Beta={b}')

    #s = samples.set_beta(b)
    cb = axes[0, i].hist2d(samples['p0'], samples['p1'], 
                            weights=samples.set_beta(b).get_weights(), 
                            cmap='Blues', bins=50, density=True,
                            vmin=0, vmax=lpmax)
    plt.colorbar(cb[3],fraction=0.046, pad=0.04)
    axes[0, i].set_title(f'Beta={b}')

    flow_samples, flow_weights = f.sample(10000, beta=b)
    flow_samples = flow_samples.numpy()
    mask = np.isfinite(flow_samples)
    mask = [m.all() for m in mask]
    flow_samples = flow_samples[mask]
    flow_weights = flow_weights[mask]

    cb = axes[2, i].hist2d(flow_samples[:, 0], flow_samples[:, 1],
                            weights=flow_weights, bins=50, density=True,
                            cmap='Blues', vmin=0, vmax=lpmax)
    plt.colorbar(cb[3],fraction=0.046, pad=0.04)
    axes[2, i].set_title(f'Beta={b}')

for j in range(len(axes)):
    for i in range(len(axes[j])):
        axes[j, i].set_xlim(0.5, 2.5)
        axes[j, i].set_ylim(0., 2.5)

axes[0, 0].set_ylabel('Nested Sampling')
axes[1, 0].set_ylabel('Beta Flow\nLog Prob')
axes[2, 0].set_ylabel('Beta Flow\nSamples')
plt.tight_layout()
plt.savefig('flow_probs_' + file + '.png', dpi=300)
plt.show()