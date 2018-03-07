"""The prior distribution over latent-variable values."""
import tensorflow as tf
from tensorflow import distributions as ds


def prior(lv):
    """Prior builds the prior distribution against the provided LV tensor.

    Args:
        lv (tf.Tensor): latent_size x batch_size latent variable tensor.

    Returns:
        tf.distributions.Normal: The distribution.
    """
    loc = tf.zeros(lv.shape)
    scale = tf.ones(lv.shape)
    return ds.Normal(loc, scale)
