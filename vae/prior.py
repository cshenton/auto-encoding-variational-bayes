"""The prior distribution over latent-variable values."""
import tensorflow as tf
import tensorflow.distributions as ds


def prior(lv):
    """Prior builds the prior distribution against the provided LV tensor.

    Returns:
        tf.distributions.Normal: The distribution.
    """
    return ds.Normal(0.0, 1.0)
