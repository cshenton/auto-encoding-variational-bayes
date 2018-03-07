"""The prior distribution over latent-variable values."""
import tensorflow as tf
from tensorflow import distributions as ds


def prior(latent_size):
    """Prior builds the prior distribution against the provided latent tensor.

    Args:
        latent_size (int): The dimension of the latent space.

    Returns:
        tf.distributions.Normal: The prior over a single latent tensor.
    """
    shp = [latent_size]
    loc = tf.zeros(shp)
    scale = tf.ones(shp)
    return ds.Normal(loc, scale)
