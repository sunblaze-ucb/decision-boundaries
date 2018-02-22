import tensorflow as tf
import numpy as np
import tqdm

# usage: python optens_attack.py offset count [<random init index>]
import sys
offset = int(sys.argv[1])
count = int(sys.argv[2])
if len(sys.argv) >= 4:
    rii = int(sys.argv[3])
else:
    rii = None

model_dir = 'models/naturally_trained'
step = 70000
out_dir = 'optenscore_madry_nat/parts'

import l2_attack_ensemble

import cifar10_input
cifar = cifar10_input.CIFAR10Data('cifar10_data')
data_slice  = slice(offset, offset + count)
xs = cifar.eval_data.xs[data_slice]
ys = cifar.eval_data.ys[data_slice]
ys = np.eye(10, dtype=np.float32)[ys]

np.set_printoptions(suppress=True)

sess = tf.Session()
model_var_slice = None

class ModelShim:
    def __init__(self):
        self.image_size = 32
        self.num_channels = 3
        self.num_labels = 10
    def predict(self, x):
        global model_var_slice
        if model_var_slice is not None:
            raise Exception('multiple calls to predict')
        model_var_start = len(tf.global_variables())
        import model_2
        model = model_2.Model('eval', x)
        model_var_end = len(tf.global_variables())
        model_var_slice = slice(model_var_start, model_var_end)
        return model.pre_softmax
model_shim = ModelShim()
random_init = rii is not None
attack = l2_attack_ensemble.CarliniL2Ensemble(sess, model_shim,
                                              batch_size=len(xs),
                                              targeted=False,
                                              binary_search_steps=4,
                                              max_iterations=1000,
                                              initial_const=1.,
                                              random_init=random_init)

if random_init:
    np.save('%s/cwl2_test%d_step%d_ri%d_modifier_init.npy' % (out_dir, offset, step, rii), attack.modifier_init)

saver = tf.train.Saver(tf.global_variables()[model_var_slice])
saver.restore(sess, '%s/checkpoint-%d' % (model_dir, step))

adv_imgs = attack.attack(xs, ys)
if random_init:
    np.save('%s/cwl2_test%d_step%d_ri%d.npy' % (out_dir, offset, step, rii), adv_imgs)
else:
    np.save('%s/cwl2_test%d_step%d.npy' % (out_dir, offset, step), adv_imgs)
