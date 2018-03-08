"""Encoder builds the encoder network on a given input image."""
import tensorflow as tf
from tensorflow import distributions as ds


def encoder(img, latent_size, units):
    """Encoder builds an encoder network against the provided image tensor.

    Args:
        img (tf.Tensor): batch_size x img_size tensor of flat images.

    Returns:
        (tf.distribution.Normal): The batch_shape = (batch_size, latent_size)
            batch of posterior normal distributions.
    """
    hidden = tf.layers.dense(img, units, tf.tanh)

    loc = tf.layers.dense(hidden, latent_size)
    scale = tf.layers.dense(hidden, latent_size, tf.nn.softplus) + 1e-8
    return ds.Normal(loc, scale)
