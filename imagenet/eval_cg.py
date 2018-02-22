import numpy as np
import tensorflow as tf
import tqdm

imin = 0.
imax = 1.
ih = 299
iw = 299
ic = 3
ifeat = ih * iw * ic
num_classes = 1001

# cg_num_per = 100
# cg_tau = 0.02
cg_num_per = 1
cg_tau = 0.

# usage: python eval_cg.py <model> <dataset name> <images path> <labels path> <count>
import sys
modelname, dataset, images_path, labels_path, count = sys.argv[1:]
count = int(count)

import stuff_v2 as stuff
sess = stuff.get_session()
m = stuff.M(modelname)

x = tf.placeholder(shape=[ifeat], dtype=tf.float32)
perturbation = tf.random_uniform((cg_num_per, ifeat), minval=-cg_tau, maxval=cg_tau, dtype=tf.float32)
avg_rms = tf.reduce_mean(tf.sqrt(tf.reduce_mean(tf.square(perturbation), axis=1)))
x_samples = tf.clip_by_value(x[None] + perturbation, imin, imax)
logits = m.predict(tf.reshape(x_samples, [cg_num_per, ih, iw, ic]))
preds = tf.argmax(logits, axis=1)
pred_count = tf.unsorted_segment_sum(data=tf.ones(cg_num_per, dtype=tf.int32), segment_ids=preds, num_segments=num_classes)

xs = np.load(images_path, 'r')
if xs.shape[1] != ifeat:
    xs = xs.reshape((-1, ifeat))
ys = np.load(labels_path, 'r')

import os
if not os.path.isdir('cgrc_%s' % modelname):
    os.mkdir('cgrc_%s' % modelname)

pred_counts_all = np.zeros((count, num_classes), dtype=np.int32)
top_preds = np.zeros(count, dtype=np.uint16)
rms_all = np.zeros(count, dtype=np.float32)
for j in tqdm.trange(count, leave=False):
    pred_counts_all[j], rms_all[j] = sess.run([pred_count, avg_rms], feed_dict={x: xs[j]})
    top_preds[j] = np.argmax(pred_counts_all[j])
    
np.save('cgrc_%s/%s_count.npy' % (modelname, dataset), pred_counts_all)
print 'accuracy', np.count_nonzero(np.equal(top_preds, ys[:count])) / float(count), 'avg rms', np.mean(rms_all)
