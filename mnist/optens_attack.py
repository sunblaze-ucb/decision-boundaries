import tensorflow as tf
import numpy as np
import tqdm

# usage: python optens_attack.py [<init spec>]
import sys
if len(sys.argv) >= 2:
    init_spec = sys.argv[1]
    if init_spec == 'fromgrad':
        # start from boundary in gradient direction
        init_type = 'fromgrad'
    else:
        # random initialization
        init_type = 'rand'
        rii = int(init_spec)
else:
    init_type = 'orig'
    rii = None

model_dir = 'models/natural'
step = 24900
gxr_dir = 'gxr_madry_nat'
out_dir = 'optens_madry_nat'
offset = 3152
count = 100

import l2_attack_ensemble

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=False)
data_slice = slice(offset, offset + count)
xs = mnist.test.images[data_slice]
xsi = xs.reshape((-1, 28, 28, 1))
ys = mnist.test.labels[data_slice]
ys = np.eye(10, dtype=np.float32)[ys]

np.set_printoptions(suppress=True)

sess = tf.Session()

import model_alt
model = model_alt.Model()
saver = tf.train.Saver()
saver.restore(sess, '%s/checkpoint-%d' % (model_dir, step))

if init_type == 'fromgrad':
    grad_bound = np.zeros_like(xsi)
    cg_raw = np.load('%s/center_grads_test%d_step%d.npy' % (gxr_dir, offset, step))
    cg = cg_raw.reshape((-1, 28, 28, 1))
    cg_linf = np.amax(np.abs(cg), axis=(1, 2, 3))
    for j in range(len(xsi)):
        index = j + offset
        conf = np.load('%s/test%d_step%d.npy' % (gxr_dir, index, step))
        conf_amin = np.amin(conf[1000])
        if conf_amin == 0:
            conf_argmin = np.argmin(conf[1000])
            boundary_dist = conf_argmin * 0.02
            grad_unit = cg[j] / cg_linf[j]
            grad_bound[j] = np.clip(xsi[j] + boundary_dist * grad_unit, 0, 1)
        else:
            # gxr did not find a boundary in the gradient dir.
            # in this case, we just fall back to the original image \:
            grad_bound[j] = xsi[j]
    np.save('%s/cwl2_test%d_step%d_fromgrad_grad_bound.npy' % (out_dir, offset, step), grad_bound)
    atan_orig = np.arctanh((xsi - 0.5) / 0.50001)
    atan_adv = np.arctanh((grad_bound - 0.5) / 0.50001)
    mod = atan_adv - atan_orig
    attack_params = {'modifier_init': mod}
elif init_type == 'rand':
    attack_params = {'random_init': rii}
else:
    attack_params = {}

attack = l2_attack_ensemble.CarliniL2Ensemble(sess, model,
                                              batch_size=len(xsi),
                                              targeted=False,
                                              binary_search_steps=4,
                                              max_iterations=1000,
                                              learning_rate=0.2,
                                              initial_const=0.2,
                                              **attack_params)

if init_type == 'rand':
    np.save('%s/cwl2_test%d_step%d_ri%d_modifier_init.npy' % (out_dir, offset, step, rii), attack.modifier_init)

adv_imgs = attack.attack(xsi, ys)
if init_type == 'fromgrad':
    np.save('%s/cwl2_test%d_step%d_fromgrad.npy' % (out_dir, offset, step), adv_imgs)
elif init_type == 'rand':
    np.save('%s/cwl2_test%d_step%d_ri%d.npy' % (out_dir, offset, step, rii), adv_imgs)
else:
    np.save('%s/cwl2_test%d_step%d.npy' % (out_dir, offset, step), adv_imgs)
