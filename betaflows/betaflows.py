import tensorflow as tf
import tensorflow_probability as tfp
from margarine.processing import (_forward_transform, _inverse_transform,
                                  pure_tf_train_test_split)
from anesthetic.plot import kde_contour_plot_2d
import tqdm
import numpy as np
import matplotlib.pyplot as plt
import re
import pickle

# define tensorflow stuff
tfk = tf.keras
tfkl = tf.keras.layers
tfpl = tfp.layers
tfd = tfp.distributions
tfb = tfp.bijectors

class BetaFlow():
    def __init__(self, theta, **kwargs):
        self.number_networks = kwargs.pop('number_networks', 6)
        self.learning_rate = kwargs.pop('learning_rate', 1e-3)
        self.hidden_layers = kwargs.pop('hidden_layers', [50, 50])
        self.parameters = kwargs.pop('parameters', None)
        self.conditional_dims = kwargs.pop('conditional_dims', 1)
        self.sample_weights = kwargs.pop('weights', None)
        self.dims = theta.shape[-1]
        self.theta = theta

        self.n = tf.math.reduce_sum(self.sample_weights)**2 / \
            tf.math.reduce_sum(self.sample_weights**2)

        theta_max = tf.math.reduce_max(self.theta, axis=0)
        theta_min = tf.math.reduce_min(self.theta, axis=0)
        a = ((self.n-2)*theta_max-theta_min)/(self.n-3)
        b = ((self.n-2)*theta_min-theta_max)/(self.n-3)
        self.theta_min = kwargs.pop('theta_min', b)
        self.theta_max = kwargs.pop('theta_max', a)

        self.optimizer = tf.keras.optimizers.legacy.Adam(
                learning_rate=self.learning_rate)

        # standard autoregressive network
        self.mades = [tfb.AutoregressiveNetwork(
            params=2,
            hidden_units=self.hidden_layers,
            event_shape=(self.dims,),
            conditional=True,
            conditional_event_shape=(self.conditional_dims,),
            activation="tanh",
            dtype=np.float32,
            input_order="random",
        ) for i in range(self.number_networks)]

        # make the masked autoregressive flow
        self.bij = tfb.Chain([tfb.MaskedAutoregressiveFlow(made, name='maf'+str(i))
                        for i,made in enumerate(self.mades)])

        self.base = tfd.Blockwise(
                    [tfd.Normal(loc=0, scale=1)
                    for _ in range(self.dims)])

        # make the transformed distribution
        self.maf = tfd.TransformedDistribution(
            distribution=self.base,
            bijector=self.bij,
        )


    def training(self, conditional, epochs=100,
                    loss_type='sum', early_stop=True, patience=None):

        """Training the masked autoregressive flow.
        
        maf is the transformed distribution.
        """

        if patience is None:
            patience = round((epochs/100)*2)

        phi = _forward_transform(self.theta, self.theta_min, self.theta_max)
        mask = np.isfinite(np.sum(phi, axis=1))
        phi = phi[mask]
        conditional = conditional[mask]
        # I was normalising the weights here but need to normalise for each beta...
        weights_phi = self.sample_weights[mask]
        phi = np.hstack([phi, conditional.reshape(-1, 1)]).astype(np.float32)

        phi_train, phi_test, weights_phi_train, weights_phi_test = \
            pure_tf_train_test_split(phi, weights_phi, test_size=0.2)
        
        conditional_train = phi_train[:, -1]
        conditional_test = phi_test[:, -1]
        phi_train = phi_train[:, :-1]
        phi_test = phi_test[:, :-1]


        self.loss_history = []
        self.test_loss_history = []
        c = 0
        for i in tqdm.tqdm(range(epochs)):
            loss = self._train_step(phi_train,
                                    weights_phi_train, conditional_train,
                                    loss_type, self.maf)
            self.loss_history.append(loss)

            self.test_loss_history.append(self._test_step(phi_test,
                                            weights_phi_test, conditional_test,
                                            loss_type, self.maf))

            if early_stop:
                c += 1
                if i == 0:
                    minimum_loss = self.test_loss_history[-1]
                    minimum_epoch = i
                    minimum_model = None
                else:
                    if self.test_loss_history[-1] < minimum_loss:
                        minimum_loss = self.test_loss_history[-1]
                        minimum_epoch = i
                        minimum_model = self.maf.copy()
                        c = 0
                if minimum_model:
                    if c == patience:
                        print('Early stopped. Epochs used = ' + str(i) +
                                '. Minimum at epoch = ' + str(minimum_epoch))
                        self.maf = minimum_model.copy()
                        break

    @tf.function(jit_compile=True)
    def _test_step(self,x, w, c_, loss_type, maf):

        r"""
        This function is used to calculate the test loss value at each epoch
        for early stopping.
        """
        c_ = tf.reshape(c_, (-1, 1))
        if loss_type == 'sum':
            loss = -tf.reduce_sum(w*maf.log_prob(
                    x, bijector_kwargs=self.make_bijector_kwargs(maf.bijector, 
                    {'maf.': {'conditional_input': c_}}
                )))
        elif loss_type == 'mean':
            loss = -tf.reduce_mean(w*maf.log_prob(
                    x, bijector_kwargs=self.make_bijector_kwargs(maf.bijector,
                    {'maf.': {'conditional_input': c_}})
                ))
        return loss

    @tf.function(jit_compile=True)
    def _train_step(self, x, w, c_, loss_type, maf):

        r"""
        This function is used to calculate the loss value at each epoch and
        adjust the weights and biases of the neural networks via the
        optimizer algorithm.
        """
        c_ = tf.reshape(c_, (-1, 1))
        with tf.GradientTape() as tape:
            if loss_type == 'sum':
                loss = -tf.reduce_sum(w*maf.log_prob(
                    x, bijector_kwargs=self.make_bijector_kwargs(maf.bijector, 
                    {'maf.': {'conditional_input': c_}}
                )))
            elif loss_type == 'mean':
                loss = -tf.reduce_mean(w*maf.log_prob(
                    x, bijector_kwargs=self.make_bijector_kwargs(maf.bijector, 
                    {'maf.': {'conditional_input': c_}}
                )))
        gradients = tape.gradient(loss, maf.trainable_variables)
        self.optimizer.apply_gradients(
            zip(gradients,
                maf.trainable_variables))
        return loss

    #@tf.function(jit_compile=True, reduce_retracing=True)
    def log_prob(self, params, c_):

            # Enforce float32 dtype
            if params.dtype != tf.float32:
                params = tf.cast(params, tf.float32)

            length = params.shape[0]
            
            # check if dimensions of c are > 1
            if type(c_) is not np.ndarray:
                c_ = np.array([c_]*length).astype(np.float32).reshape(-1, 1)
            elif c_.shape[0] != length:
                c_ = c_.reshape(-1, 1)
 
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
                                        bijector_kwargs=self.make_bijector_kwargs(
                                        maf.bijector, {'maf.': {'conditional_input': c_}})) -
                        tf.reduce_sum(correction, axis=-1))
                return logprob

            logprob = calc_log_prob(self.theta_min, self.theta_max, self.maf)

            return logprob

    #@tf.function(jit_compile=True)
    def __call__(self, u, c_):
        r"""

        This function is used when calling the MAF class to transform
        samples from the unit hypercube to samples on the MAF.

        **Parameters:**

            u: **numpy array**
                | Samples on the uniform hypercube.

        """

        if u.dtype != tf.float32:
            u = tf.cast(u, tf.float32)
        
        length = u.shape[0]
        
        if len(c_.shape) > 1:
                c_ = c_.reshape(-1, 1)
        else:
            c_ = np.array([c_]*length).astype(np.float32).reshape(-1, 1)

        x = _forward_transform(u)
        bijector_kwargs = self.make_bijector_kwargs(self.bij, 
                    {'maf.': {'conditional_input': c_}})
        x = self.bij.forward(x, **bijector_kwargs)
        x = _inverse_transform(x, self.theta_min, self.theta_max)

        return x

    #@tf.function(jit_compile=True)
    def sample(self, length, c_):

        r"""

        This function is used to generate samples on the MAF via the
        MAF __call__ function.

        **Kwargs:**

            length: **int / default=1000**
                | This should be an integer and is used to determine how many
                    samples are generated when calling the MAF.

        """
        if type(length) is not int:
            raise TypeError("'length' must be an integer.")

        u = tf.random.uniform((length, len(self.theta_min)))
        c_ = np.array([c_]*length).astype(np.float32).reshape(-1, 1)
        return self(u, c_)

    def make_bijector_kwargs(self, bijector, name_to_kwargs):

        if hasattr(bijector, 'bijectors'):
            return {b.name: 
                    self.make_bijector_kwargs(b, name_to_kwargs) 
                    for b in bijector.bijectors}
        else:
            for name_regex, kwargs in name_to_kwargs.items():
                if re.match(name_regex, bijector.name):
                    return kwargs
        return {}

    def save(self, filename):

        nn_weights = [made.get_weights() for made in self.mades]
        with open(filename, 'wb') as f:
            pickle.dump([self.theta,
                        nn_weights,
                        self.sample_weights,
                        self.number_networks,
                        self.hidden_layers,
                        self.learning_rate,
                        self.theta_min,
                        self.theta_max,
                        self.test_loss_history,
                        self.loss_history], f)

    @classmethod
    def load(cls, filename):

        with open(filename, 'rb') as f:
            data = pickle.load(f)
            try:
                theta, nn_weights, \
                    sample_weights, \
                    number_networks, \
                    hidden_layers, \
                    learning_rate, theta_min, theta_max,\
                    test_loss_history, loss_history = data
            except:
                theta, nn_weights, \
                    sample_weights, \
                    number_networks, \
                    hidden_layers, \
                    learning_rate, theta_min, theta_max = data
                print('No loss history found.')
                test_loss_history = None
                loss_history = None
                

        bijector = cls(
            theta, weights=sample_weights, number_networks=number_networks,
            learning_rate=learning_rate, hidden_layers=hidden_layers,
            theta_min=theta_min, theta_max=theta_max)
        bijector.loss_history = loss_history
        bijector.test_loss_history = test_loss_history
        bijector(
            np.random.uniform(0, 1, size=(len(theta), theta.shape[-1])),
            np.array([1]*len(theta)).reshape(-1, 1))
        for made, nn_weights in zip(bijector.mades, nn_weights):
            made.set_weights(nn_weights)

        return bijector