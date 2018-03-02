"""VAE contains the variational auto-encoder convenience class."""


class VAE:
    """VAE is a wrapper around a ful variational auto-encoder graph.

    Attributes:
        input (tf.Tensor): Points to image input placeholder.
        loss (tf.Tensor): Points to the ELBO loss tensor.

    """

    def __init__(self, batch_size=100, sample_size=1):
        """Creates a new instance of VAE.

        This creates the complete static graph, which is accessed afterwards
        only through session runs.

        Args:
            batch_size (int): The minibatch size, determines input tensor dims.
            sample_size (int): The sample size drawn from the recognition model.
                Usually 1, since we do stochastic integration.
        """
        pass
