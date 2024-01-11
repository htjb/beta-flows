import tensorflow as tf
import tensorflow_probability as tfp
from margarine.processing import (_forward_transform, _inverse_transform,
                                  pure_tf_train_test_split)
from anesthetic.plot import kde_contour_plot_2d
import tqdm
import numpy as np
import matplotlib.pyplot as plt
import re

# define tensorflow stuff
tfk = tf.keras
tfkl = tf.keras.layers
tfpl = tfp.layers
tfd = tfp.distributions
tfb = tfp.bijectors

def _training(theta, sample_weights, conditional, maf,
                  theta_min, theta_max, epochs=100,
                  loss_type='mean', early_stop=True):

    """Training the masked autoregressive flow.
    
    maf is the transformed distribution.
    """

    phi = _forward_transform(theta, theta_min, theta_max)
    mask = np.isfinite(np.sum(phi, axis=1))
    phi = phi[mask]
    sample_weights = sample_weights[mask]
    conditional = conditional[mask]
    weights_phi = sample_weights/tf.reduce_sum(sample_weights)
    phi = np.hstack([phi, conditional.reshape(-1, 1)]).astype(np.float32)

    phi_train, phi_test, weights_phi_train, weights_phi_test = \
        pure_tf_train_test_split(phi, weights_phi, test_size=0.2)
    
    conditional_train = phi_train[:, -1]
    conditional_test = phi_test[:, -1]
    phi_train = phi_train[:, :-1]
    phi_test = phi_test[:, :-1]


    loss_history = []
    test_loss_history = []
    c = 0
    for i in tqdm.tqdm(range(epochs)):
        loss = _train_step(phi_train,
                                weights_phi_train, conditional_train,
                                loss_type, maf)
        loss_history.append(loss)

        test_loss_history.append(_test_step(phi_test,
                                        weights_phi_test, conditional_test,
                                        loss_type, maf))

        if early_stop:
            c += 1
            if i == 0:
                minimum_loss = test_loss_history[-1]
                minimum_epoch = i
                minimum_model = None
            else:
                if test_loss_history[-1] < minimum_loss:
                    minimum_loss = test_loss_history[-1]
                    minimum_epoch = i
                    minimum_model = maf.copy()
                    c = 0
            if minimum_model:
                if c == round((epochs/100)*2):
                    print('Early stopped. Epochs used = ' + str(i) +
                            '. Minimum at epoch = ' + str(minimum_epoch))
                    return minimum_model, loss_history, test_loss_history
    return maf, loss_history, test_loss_history

@tf.function(jit_compile=True)
def _test_step(x, w, c_, loss_type, maf):

    r"""
    This function is used to calculate the test loss value at each epoch
    for early stopping.
    """
    c_ = tf.reshape(c_, (-1, 1))
    if loss_type == 'sum':
        loss = -tf.reduce_sum(w*maf.log_prob(
                x, bijector_kwargs=make_bijector_kwargs(maf.bijector, {'maf.': {'conditional_input': c_}}
            )))
    elif loss_type == 'mean':
        loss = -tf.reduce_mean(w*maf.log_prob(
                x, bijector_kwargs=make_bijector_kwargs(maf.bijector, {'maf.': {'conditional_input': c_}})
            ))
    return loss

@tf.function(jit_compile=True)
def _train_step(x, w, c_, loss_type, maf):

    r"""
    This function is used to calculate the loss value at each epoch and
    adjust the weights and biases of the neural networks via the
    optimizer algorithm.
    """
    c_ = tf.reshape(c_, (-1, 1))
    with tf.GradientTape() as tape:
        if loss_type == 'sum':
            loss = -tf.reduce_sum(w*maf.log_prob(
                x, bijector_kwargs=make_bijector_kwargs(maf.bijector, {'maf.': {'conditional_input': c_}}
            )))
        elif loss_type == 'mean':
            #loss = -tf.reduce_mean(w*maf.log_prob(x, made1={'conditional_input', c_}, made2={'conditional_input', c_}))
            loss = -tf.reduce_mean(w*maf.log_prob(
                x, bijector_kwargs=make_bijector_kwargs(maf.bijector, {'maf.': {'conditional_input': c_}}
            )))
    gradients = tape.gradient(loss, maf.trainable_variables)
    optimizer.apply_gradients(
        zip(gradients,
            maf.trainable_variables))
    return loss

@tf.function(jit_compile=True, reduce_retracing=True)
def log_prob(params, theta_min, theta_max, maf, c_):

        # Enforce float32 dtype
        if params.dtype != tf.float32:
            params = tf.cast(params, tf.float32)

        def calc_log_prob(mins, maxs, maf):

            """Function to calculate log-probability for a given MAF."""

            def norm_jac(y):
                return transform_chain.inverse_log_det_jacobian(
                    y, event_ndims=0)

            transformed_x = _forward_transform(params, mins, maxs)

            transform_chain = tfb.Chain([
                tfb.Invert(tfb.NormalCDF()),
                tfb.Scale(1/(maxs - mins)), tfb.Shift(-mins)])

            correction = norm_jac(transformed_x)
            logprob = (maf.log_prob(transformed_x,
                                    bijector_kwargs=make_bijector_kwargs(
                                    maf.bijector, {'maf.': {'conditional_input': c_}})) -
                       tf.reduce_sum(correction, axis=-1))
            return logprob

        logprob = calc_log_prob(theta_min, theta_max, maf)

        return logprob

def make_bijector_kwargs(bijector, name_to_kwargs):

    if hasattr(bijector, 'bijectors'):
        return {b.name: make_bijector_kwargs(b, name_to_kwargs) for b in bijector.bijectors}
    else:
        for name_regex, kwargs in name_to_kwargs.items():
            if re.match(name_regex, bijector.name):
                return kwargs
    return {}

optimizer = tf.keras.optimizers.legacy.Adam(
                learning_rate=1e-3)

# define the input and conditional dimensions
nbeta_samples = 10
dims = 2
conditional_dims = 1
number_networks = 6

# standard autoregressive network
mades = [tfb.AutoregressiveNetwork(
    params=2,
    hidden_units=[50],
    event_shape=(dims,),
    conditional=True,
    conditional_event_shape=(conditional_dims,),
    activation="sigmoid",
    dtype=np.float32
) for i in range(number_networks)]

# make the masked autoregressive flow
bij = tfb.Chain([tfb.MaskedAutoregressiveFlow(made, name='maf'+str(i))
                 for i,made in enumerate(mades)])

base = tfd.Blockwise(
            [tfd.Normal(loc=0, scale=1)
             for _ in range(dims)])

# make the transformed distribution
maf = tfd.TransformedDistribution(
    distribution=base,
    bijector=bij,
)

from anesthetic import read_chains

ns = read_chains('gauss_ns_run/test')

# beta schedule
beta = np.linspace(-3, 0, nbeta_samples)
beta = 10**beta

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

maf, loss_history, test_loss_history = _training(theta, sample_weights, conditional, maf,
                  theta_min, theta_max, epochs=10000,
                  loss_type='mean', early_stop=True)

plt.plot(loss_history, label='train')
plt.plot(test_loss_history, label='test')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig('gauss_loss.png', dpi=300)
plt.show()

fig, axes = plt.subplots(2, 6, figsize=(10, 6))
test_beta = [0.01, 0.1, 0.3, 0.5, 0.7, 1]
for i in range(len(test_beta)):
    nsamplesGen = len(ns.values)
    samples = maf.sample(nsamplesGen,
            bijector_kwargs=make_bijector_kwargs(
                maf.bijector, 
                {'maf.': {'conditional_input': 
                np.array([test_beta[i]]*int(nsamplesGen)).reshape(-1, 1)}}))
    samples = _inverse_transform(samples, theta_min, theta_max)
    samples = samples.numpy()

    s = ns.set_beta(test_beta[i])
    try:
        kde_contour_plot_2d(axes[0, i], s.values[:, 0], s.values[:, 1], weights=s.get_weights())
        kde_contour_plot_2d(axes[1, i], samples[:, 0], samples[:, 1], weights=s.get_weights())
    except:
        axes[0, i].hist2d(s.values[:, 0], s.values[:, 1], weights=s.get_weights(), bins=50)
        axes[1, i].hist2d(samples[:, 0], samples[:, 1], weights=s.get_weights(), bins=50)

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
plt.savefig('gauss_beta_flow.png', dpi=300)
plt.show()

############## Train beta=1 flow (normal margarine) ############

from margarine.maf import MAF

ns = read_chains('gauss_ns_run/test')
try:
    flow = MAF.load('gauss_normal_flow.pkl')
except FileNotFoundError:
    flow = MAF(ns.values[:, :2], weights=ns.get_weights())
    flow.train(5000, early_stop=True)
    flow.save('gauss_normal_flow.pkl')

nflp = flow.log_prob(ns.values[:, :2]).numpy()
cnflp = log_prob(ns.values[:, :2], theta_min, 
                 theta_max, maf, np.array([1]*len(ns.values)).astype(np.float32).reshape(-1, 1))

posterior_probs = ns['logL'] + np.log(ns.get_weights()) - ns.stats(1000)['logZ'].mean()
posterior_probs = posterior_probs.values

mask = np.isfinite(cnflp)

print('Normal Flow Average Like: ', np.average(nflp, weights=ns.get_weights()))
print('CNF Average Like: ', np.average(cnflp[mask], weights=ns.get_weights()[mask]))

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
    axes[0, i].set_xlim([posterior_probs.min(), posterior_probs.max()])
    axes[0, i].set_ylim([posterior_probs.min(), posterior_probs.max()])

axes[0, 2].scatter(ns.values[:, 0], ns.values[:, 1], c=posterior_probs, cmap='viridis_r', marker='.')
axes[0, 2].set_title('Samples')
axes[0, 2].set_xlabel(r'$\theta_1$')
axes[0, 2].set_ylabel(r'$\theta_2$')
axes[1, 2].axis('off')
axes[0, 0].set_title('Normal margarine')
axes[0, 1].set_title('beta flow')
plt.tight_layout()
plt.savefig('gauss_likleihood_comparison.png', dpi=300)
plt.show()

from scipy.special import logsumexp
fig, axes = plt.subplots(2, 2, figsize=(10, 10))
for i in range(2):
    axes[i, 0].scatter(posterior_probs/np.log(10), (posterior_probs- nflp)/np.log(10), marker='+', c=posterior_probs, cmap='viridis_r', label='Normal margarine')
    axes[i, 1].scatter(posterior_probs/np.log(10), (posterior_probs - cnflp)/np.log(10), marker='*', c=posterior_probs, cmap='viridis_r', label='beta flow')
    for j in range(2):
        axes[i, j].axhline(0, linestyle='--', color='k')
    axes[1, i].set_xlabel(r'$\log_{10} P_{true}$')
    axes[0, i].set_ylabel(r'$\log_{10} \frac{P_{true}}{P_{flow}}$')

axes[0, 0].set_title('Normal margarine')
axes[0, 1].set_title('beta flow')
axes[1, 0].set_ylim([-2000, 2000])
axes[1, 1].set_ylim([-2000, 2000])


plt.ylabel('True log-posterior - Flow log-posterior')
plt.legend()
plt.savefig('gauss_likleihood_comparison_residuals.png', dpi=300)
plt.show()