import numpy as np
import tensorflow as tf
import tqdm

imin = 0.
imax = 255.
ih = 32
iw = 32
ic = 3
ifeat = ih * iw * ic
num_classes = 10

# cg_num_per = 100
# cg_tau = 5.1
cg_num_per = 1
cg_tau = 0.

# usage: python eval_cg.py <model> <step> <dataset name> <images path> <labels path> <count>
import sys
modelname, step, dataset, images_path, labels_path, count = sys.argv[1:]
step = int(step)
count = int(count)

checkpoint_dirs = {
    'madry_robust': 'models/adv_trained/checkpoint-%d',
    'madry_robust_keep': 'models/robust_with_intermediates/checkpoint-%d',
    'madry_nat': 'models/naturally_trained/checkpoint-%d'
}
if modelname not in checkpoint_dirs:
    raise Exception('unrecognized model')
checkpoint_dir = checkpoint_dirs[modelname]

if modelname in ['madry_robust', 'madry_robust_keep', 'madry_nat']:
    import model_2
    def predict(imgs):
        m = model_2.Model('eval', imgs)
        return m.pre_softmax
else:
    raise Exception('unrecognized model')

x = tf.placeholder(shape=[ifeat], dtype=tf.float32)
perturbation = tf.random_uniform((cg_num_per, ifeat), minval=-cg_tau, maxval=cg_tau, dtype=tf.float32)
avg_rms = tf.reduce_mean(tf.sqrt(tf.reduce_mean(tf.square(perturbation), axis=1)))
x_samples = tf.clip_by_value(x[None] + perturbation, imin, imax)
logits = predict(tf.reshape(x_samples, [cg_num_per, ih, iw, ic]))
preds = tf.argmax(logits, axis=1)
pred_count = tf.unsorted_segment_sum(data=tf.ones(cg_num_per, dtype=tf.int32), segment_ids=preds, num_segments=num_classes)

saver = tf.train.Saver()
sess = tf.Session()
saver.restore(sess, checkpoint_dir % step)

xs = np.load(images_path, 'r')
if xs.shape[1] != ifeat:
    xs = xs.reshape((-1, ifeat))
ys = np.load(labels_path, 'r')

import os
if not os.path.isdir('cgrc_%s' % modelname):
    os.mkdir('cgrc_%s' % modelname)

pred_counts_all = np.zeros((count, num_classes), dtype=np.int32)
top_preds = np.zeros(count, dtype=np.uint8)
rms_all = np.zeros(count, dtype=np.float32)
for j in tqdm.trange(count, leave=False):
    pred_counts_all[j], rms_all[j] = sess.run([pred_count, avg_rms], feed_dict={x: xs[j]})
    top_preds[j] = np.argmax(pred_counts_all[j])
    
np.save('cgrc_%s/step%s_%s_count.npy' % (modelname, step, dataset), pred_counts_all)
print 'accuracy', np.count_nonzero(np.equal(top_preds, ys[:count])) / float(count), 'avg rms', np.mean(rms_all)
