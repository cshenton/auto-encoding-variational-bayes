"""Training methods for MNIST data"""
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
from vae.vae import VAE

def train_mnist(latent_size=10, batch_size=100):
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    v = VAE(img_size=784, latent_size=latent_size)
    train = tf.train.AdamOptimizer(0.0001, 0.9, 0.999, 1e-6).minimize(v.loss)

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    n = 0

    while n < 1e8:
        minibatch, _ = mnist.train.next_batch(batch_size)
        _, l = sess.run([train, v.loss], feed_dict={v.input: minibatch})
        if i % 50 == 0:
            print(n, ':', l)
        n += batch_size
