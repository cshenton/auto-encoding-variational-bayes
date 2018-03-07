"""The prior distribution over latent-variable values."""
import tensorflow as tf
from tensorflow import distributions as ds


def prior(sample_size, batch_size, latent_size):
    """Prior builds the prior distribution against the provided latent tensor.

    Args:
        latent (tf.Tensor): sample_size x batch_size x latent_size tensor
            containing the sampled latent variables.

    Returns:
        tf.distributions.Normal: The prior distribution over the sample.
    """
    shp = [sample_size, batch_size, latent_size]
    loc = tf.zeros(shp)
    scale = tf.ones(shp)
    return ds.Normal(loc, scale)
