"""Encoder builds the encoder network on a given input image."""
import tensorflow as tf
from tensorflow import distributions as ds


def encoder(img, latent_size=10, units=500):
    """Encoder builds an encoder network against the provided image tensor.

    Args:
        img (tf.Tensor): img_size x batch_size tensor of flat images.

    Returns:
        (tf.distribution.Normal): The batch of normal distributions each
            repesenting the posterior LV distribution for that image.
    """
    hidden = tf.layers.dense(img, units)

    loc = tf.layers.dense(hidden, latent_size)
    scale = tf.layers.dense(hidden, latent_size)
    return ds.Normal(loc, scale)
