"""Decoder builds the decoder network on a given latent variable."""
import tensorflow as tf
from tensorflow import distributions as ds


def decoder(latent, img_size=225, units=500):
    """Decoder builds a decoder network on the given latent variable tensor.

    Args:
        lv (tf.Tensor): The latent variable tensor.

    Returns:
        (tf.distribution.Normal): The batched normal distribution representing
            the likelihood over output images given the latent variable.
    """
    hidden = tf.layers.dense(latent, units)

    loc = tf.layers.dense(hidden, img_size)
    scale = tf.layers.dense(hidden, img_size)
    return ds.Normal(loc, scale)
