import tensorflow as tf
import tensorflow_probability as tfp
from margarine.processing import (_forward_transform, _inverse_transform,
                                  pure_tf_train_test_split)
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
                    return minimum_model
    return maf

#@tf.function(jit_compile=True)
def _test_step(x, w, c_, loss_type, maf):

    r"""
    This function is used to calculate the test loss value at each epoch
    for early stopping.
    """
    c_ = tf.reshape(c_, (-1, 1))
    if loss_type == 'sum':
        loss = -tf.reduce_sum(w*maf.log_prob(x, made1={'conditional_input': c_}, made2={'conditional_input': c_}))
    elif loss_type == 'mean':
        loss = -tf.reduce_mean(w*maf.log_prob(
                x, bijector_kwargs=make_bijector_kwargs(maf.bijector, {'maf.': {'conditional_input': c_}})
            ))
    return loss

#@tf.function(jit_compile=True)
def _train_step(x, w, c_, loss_type, maf):

    r"""
    This function is used to calculate the loss value at each epoch and
    adjust the weights and biases of the neural networks via the
    optimizer algorithm.
    """
    c_ = tf.reshape(c_, (-1, 1))
    with tf.GradientTape() as tape:
        if loss_type == 'sum':
            loss = -tf.reduce_sum(w*maf.log_prob(x, made1={'conditional_input': c_}, made2={'conditional_input': c_}))
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
nsamples = 5000
dims = 2
conditional_dims = 1
number_networks = 2

# standard autoregressive network
mades = [tfb.AutoregressiveNetwork(
    params=2,
    hidden_units=[64],
    event_shape=(dims,),
    conditional=True,
    conditional_event_shape=(conditional_dims,),
    activation="relu",
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

# define some data
theta1 = np.random.multivariate_normal(np.random.uniform(0, 3, dims), 
                                      np.eye(dims)*np.random.uniform(0, 1, dims), 
                                      size=int(nsamples/2)).astype(np.float32)
theta2 = np.random.multivariate_normal(np.random.uniform(-5, -1, dims),
                                        np.eye(dims)*np.random.uniform(0, 1, dims),
                                        size=int(nsamples/2)).astype(np.float32)
theta = np.vstack([theta1, theta2])

sample_weights = np.ones(nsamples).astype(np.float32)
conditional = np.hstack([[1]*int(int(nsamples/2)), [2]*int(nsamples/2)]).astype(np.float32)

theta_min = np.min(theta, axis=0)
theta_max = np.max(theta, axis=0)

maf = _training(theta, sample_weights, conditional, maf,
                  theta_min, theta_max, epochs=1000,
                  loss_type='mean', early_stop=True)

nsamplesGen = 1000
samples = maf.sample(nsamplesGen,
        bijector_kwargs=make_bijector_kwargs(
            maf.bijector, 
            {'maf.': {'conditional_input': np.array([1]*int(nsamplesGen)).reshape(-1, 1)}}))
samples = _inverse_transform(samples, theta_min, theta_max)

plt.scatter(theta[:int(nsamples/2), 0], theta[:int(nsamples/2), 1], 
            c='g', marker='.', alpha=0.5,label='Cond=1')
plt.scatter(theta[int(nsamples/2):, 0], theta[int(nsamples/2):, 1], 
            c='b', marker='.', alpha=0.5, label='Cond=2')
plt.scatter(samples[:, 0], samples[:, 1], c='r', 
            marker='.', alpha=0.5, label='Generated with Cond=1')

samples = maf.sample(nsamplesGen,
        bijector_kwargs=make_bijector_kwargs(
            maf.bijector, 
            {'maf.': {'conditional_input': np.array([2]*int(nsamplesGen)).reshape(-1, 1)}}))
samples = _inverse_transform(samples, theta_min, theta_max)

plt.scatter(samples[:, 0], samples[:, 1], c='k',
            marker='.', alpha=0.5, label='Generated with Cond=2')

plt.legend()
plt.show()


