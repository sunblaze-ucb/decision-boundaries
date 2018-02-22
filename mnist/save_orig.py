import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=False)
xs = mnist.test.images
ys = mnist.test.labels

np.save('orig_images.npy', xs)
np.save('orig_labels.npy', ys)
