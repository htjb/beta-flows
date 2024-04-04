import numpy as np
from betaflows.betaflows import BetaFlow
from betaflows.utils import get_beta_schedule
from anesthetic import read_chains
import matplotlib.pyplot as plt

ns = read_chains("rosenbrock/test")
values = ns.values[:, :2].astype(np.float32)

beta = get_beta_schedule(ns, 25)
#beta = 10**np.linspace(-10, 0, 25)
beta = 1-beta
print(beta)
ndims = 2

"""plt.plot(np.log10(beta), ns.D_KL(beta=beta), marker='.', ls='--')
plt.xlabel("log10(beta)")
plt.ylabel("D_KL")
plt.tight_layout()
plt.savefig("DKL.png", dpi=300, bbox_inches='tight')
plt.show()"""

theta, sample_weights, beta_values = [], [], []
for j, b in enumerate(beta):
    if j < len(beta)-1:
        s = ns.set_beta(b)
        theta.append(s.values[:, :ndims])
        sample_weights.append(s.get_weights()/s.get_weights().sum())
        beta_values.append([np.log10(b)]*len(s.values))
theta = np.concatenate(theta).astype(np.float32)
sample_weights = np.concatenate(sample_weights).astype(np.float32)
conditional = np.concatenate(beta_values).astype(np.float32)

bflow = BetaFlow(theta, weights=sample_weights, 
                    theta_min=np.array([-5.0, -5.0]).astype(np.float32), 
                    theta_max=np.array([5.0, 5.0]).astype(np.float32),
    number_networks=8, hidden_layers=[50],)
bflow.training(conditional, epochs=1000,
                loss_type='mean', early_stop=True)

evidence = ns['logL'].values + bflow.log_prob(values, np.log10(beta[-2])).numpy() \
    - bflow.log_prob(values, np.log10(1)).numpy()

mask = np.isfinite(evidence)
evidence = evidence[mask]

bounds = np.array([[-5, 5], [-5, 5]])
prior_log_prob = np.log(np.prod((1/(bounds[:, 1] - bounds[:, 0]))))
posterior_probs = ns['logL'].values + prior_log_prob - ns.stats(1000)['logZ'].mean()
posterior_probs = posterior_probs[mask]

print(ns.get_weights()[mask].sum())

print('KL: ', np.average(posterior_probs - bflow.log_prob(values, np.log10(1)).numpy()[mask],
                         weights=ns.get_weights()[mask] / ns.get_weights()[mask].sum()))

plt.hist(evidence, bins=50, histtype='step')
plt.hist(ns.logZ(len(evidence)), bins=50, histtype='step')
plt.xlabel("log(evidence)")
plt.ylabel("Frequency")
plt.legend(["BetaFlow", "Anesthetic"])
plt.tight_layout()
#plt.savefig("evidence.png", dpi=300, bbox_inches='tight')
plt.show()
