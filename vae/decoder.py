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
    hidden = tf.layers.dense(latent, units)

    loc = tf.layers.dense(hidden, img_size)
    scale = tf.layers.dense(hidden, img_size)
    return ds.Normal(loc, scale)
