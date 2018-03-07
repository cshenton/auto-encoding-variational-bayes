"""VAE contains the variational auto-encoder convenience class."""
import tensorflow as tf

from vae.decoder import decoder
from vae.encoder import encoder
from vae.prior import prior


class VAE:
    """VAE is a wrapper around a full variational auto-encoder graph.

    Attributes:
        input (tf.Tensor): Points to image input placeholder.
        latent (tf.Tensor): Points to latent variable sample tensor.
        loss (tf.Tensor): Points to the ELBO loss tensor.
        prior (tf.distribution.Normal): Prior distribution.
        encoder (tf.distribution.Normal): Encoder / recognition distribution.
        decoder (tf.distribution.Normal): Decoder distribution.
    """

    def __init__(self, img_size=225, latent_size=10, sample_size=1, units=500):
        """Creates a new instance of VAE.

        This creates the complete static graph, which is accessed afterwards
        only through session runs.

        Args:
            img_size (int): Flattened dim of input image.
            batch_size (int): The minibatch size, determines input tensor dims.
            latent_size (int): Dimension of the latent normal variable.
            sample_size (int): The sample size drawn from the recognition model.
                Usually 1, since we do stochastic integration.
        """
        self.input = tf.placeholder(tf.float32, [None, img_size])
        self.encoder = encoder(self.input, latent_size, units)
        self.latent = self.encoder.sample(sample_size)
        self.decoder = decoder(self.latent, img_size, units)
        self.prior = prior(latent_size)

        self.likelihood = self.decoder.log_prob(tf.expand_dims(self.input, 0))

        loss_batch = -(
            0.5 * tf.reduce_sum(
                tf.ones_like(self.encoder.loc) +
                tf.log(tf.square(self.encoder.loc)) -
                tf.square(self.encoder.loc) -
                tf.square(self.encoder.scale),
                1
            ) +
            tf.reduce_mean(self.likelihood, [0, 2]) / sample_size
        )
        self.loss = tf.reduce_mean(loss_batch)

    def decode(self, latent):
        """Decodes the provided latent array, returns a sample from the output.

        Args:
            latent (np.ndarray): A sample_size x batch_size x latent_size
                latent variable array.

        Returns:
            np.ndarray: A sample_size x batch_size, img_size array of sampled
                and decoded images.
        """
        sess = tf.Session()
        img = sess.run(self.decoder.sample(), data_dict={self.latent: latent})
        return img

    def encode(self, img):
        """Encodes the provided images, returns a sample from the latent posterior.

        Args:
            img (np.ndarray): A batch_size x img_size array of flattened images.

        Returns:
            np.ndarray: A sample_size x batch_size x latent_size ndarray of
                latent variables.
        """
        sess = tf.Session()
        latent = sess.run(self.latent, data_dict={self.input: img})
        return latent
