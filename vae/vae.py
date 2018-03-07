"""VAE contains the variational auto-encoder convenience class."""
import tensorflow as tf

from vae.decoder import decoder
from vae.encoder import encoder


class VAE:
    """VAE is a wrapper around a ful variational auto-encoder graph.

    Attributes:
        input (tf.Tensor): Points to image input placeholder.
        lv (tf.Tensor): Points to latent variable sample tensor.
        loss (tf.Tensor): Points to the ELBO loss tensor.
        prior (tf.distribution.Normal): Prior distribution.
        encoder (tf.distribution.Normal): Encoder / recognition distribution.
        decoder (tf.distribution.Normal): Decoder distribution.
    """

    def __init__(self, img_shape, batch_size=100, latent_size=10, sample_size=1):
        """Creates a new instance of VAE.

        This creates the complete static graph, which is accessed afterwards
        only through session runs.

        Args:
            img_shape (array of ints): Dimensions of the input images.
            batch_size (int): The minibatch size, determines input tensor dims.
            latent_size (int): Dimension of the latent normal variable.
            sample_size (int): The sample size drawn from the recognition model.
                Usually 1, since we do stochastic integration.
        """
        # create input tensor
        #   self.input = tf.Placeholder(img_shape + [batch_size])
        #   img_shape x batch_size
        # create encoder distribution
        #   self.encoder = encoder(self.input, img_shape, latent_size)
        # sample from ^ then
        #   self.lv = self.encoder.sample(sample_size)
        #   latent_size x batch_size
        # create decoder distribution
        #   self.decoder = decoder(self.lv, img_shape, latent_size)
        # create prior dist
        #   self.prior = prior(latent_size)
        # create decoder likelihood (against input)
        #   likelihood = self.decoder.log_prob(self.input)
        #   prior = self.prior.log_prob(self.lv)
        #   posterior = self.encoder.log_prob(self.lv)
        #   self.loss = likelihood + posterior - prior (or something)
        pass

    def decode(self, lv):
        """Decodes the provided lv array, returns a sample from the output.

        Args:
            lv (np.ndarray): A draw_size x latent_size array of LV values.

        Returns:
            np.ndarray: A draw_size x img_shape array of generated images
        """
        # img = sess.run(self.decoder.sample(), data_dict={self.lv: lv})
        pass

    def encode(self, img):
        """Encodes the provided images, returns a sample from the LV posterior.

        Args:
            img (np.ndarray): A draw_size x img_shape array of images.

        Returns:
            np.ndarray: A draw_size x latent_size array of LV values.
        """
        # lv = sess.run(self.lv, data_dict={self.input: img})
        pass
