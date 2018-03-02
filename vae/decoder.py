"""Decoder builds the decoder network on a given latent variable."""
import tensorflow as tf
import tensorflow.distributions as ds


def decoder(lv):
    """Decoder builds a decoder network on the given latent variable tensor.

    Args:
        lv (tf.Tensor): The latent variable tensor.

    Returns:
        (tf.distribution.Normal): The batched normal distribution representing
            the likelihood over output images given the latent variable.
    """
    pass
