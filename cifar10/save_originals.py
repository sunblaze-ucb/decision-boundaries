import numpy as np

import cifar10_input

cifar = cifar10_input.CIFAR10Data('cifar10_data')
x_nat = cifar.eval_data.xs
print 'images', x_nat.shape, x_nat.dtype
y_nat = cifar.eval_data.ys
print 'labels', y_nat.shape, y_nat.dtype

np.save('orig_images.npy', x_nat)
np.save('orig_labels.npy', y_nat)
