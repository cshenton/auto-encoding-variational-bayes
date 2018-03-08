"""Decoder builds the decoder network on a given latent variable."""
import tensorflow as tf
from tensorflow import distributions as ds


def decoder(latent, img_size, units):
    """Decoder builds a decoder network on the given latent variable tensor.

    Args:
        lv (tf.Tensor): sample_size x batch_size x latent_size latent tensor.

    Returns:
        (tf.distribution.Normal): The batch_shape = (sample x batch x img)
            normal distributions representing the sampled img likelihoods.
    """
    hidden = tf.layers.dense(latent, units, tf.tanh)

    loc = tf.layers.dense(hidden, img_size, tf.nn.sigmoid)
    scale = tf.layers.dense(hidden, img_size, tf.nn.softplus) + 1e-8
    return ds.Normal(loc, scale)
