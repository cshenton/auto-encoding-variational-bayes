"""The prior distribution over latent-variable values."""
import tensorflow as tf
from tensorflow import distributions as ds


def prior(latent):
    """Prior builds the prior distribution against the provided latent tensor.

    Args:
        latent (tf.Tensor): latent_size x batch_size latent variable tensor.

    Returns:
        tf.distributions.Normal: The distribution.
    """
    loc = tf.zeros(latent.shape)
    scale = tf.ones(latent.shape)
    return ds.Normal(loc, scale)
