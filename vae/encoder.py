"""Encoder builds the encoder network on a given input image."""
import tensorflow as tf
import tensorflow.distributions as ds


def encoder(img):
    """Encoder builds an encoder network against the provided image tensor.

    Args:
        img (tf.Tensor): The mini-batched, flattened image tensor.

    Returns:
        (tf.distribution.Normal): The batch of normal distributions each
            repesenting the posterior LV distribution for that image.
    """
    pass
