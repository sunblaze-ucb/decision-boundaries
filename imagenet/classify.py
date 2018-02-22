import numpy as np
import tensorflow as tf

import classify_common

batch_size = 350 # %%% 128
# learning_rate = 0.0001 # for opt and optens
learning_rate = 0.001
train_steps = 4000
dist_dims = 1000
purity_count = 3

# usage: python classify.py <modelname> <adv_set>
num_classes = 1000
import sys
modelname, adv_set = sys.argv[1:]

# Assemble input
# %%% TODO: add opt dist too
p_dist_benign = tf.placeholder(shape=[None, dist_dims], dtype=tf.float32)
p_purity_benign = tf.placeholder(shape=[None, purity_count], dtype=tf.float32)
p_dist_adv = tf.placeholder(shape=[None, dist_dims], dtype=tf.float32)
p_purity_adv = tf.placeholder(shape=[None, purity_count], dtype=tf.float32)
p_correct_benign = tf.placeholder(shape=[None], dtype=tf.bool)
p_correct_adv = tf.placeholder(shape=[None], dtype=tf.bool)
dat = tf.contrib.data.Dataset.from_tensor_slices([p_dist_benign, p_purity_benign, p_correct_benign, p_dist_adv, p_purity_adv, p_correct_adv])
dat = dat.shuffle(350)
dat = dat.repeat()
dat = dat.batch(batch_size)
di = dat.make_initializable_iterator()

# -
b_dist_benign, b_purity_benign, b_correct_benign, b_dist_adv, b_purity_adv, b_correct_adv = di.get_next()
x_dist = tf.concat([b_dist_benign, b_dist_adv], axis=0)
x_purity = tf.concat([b_purity_benign, b_purity_adv], axis=0)
logits = classify_common.m(x_dist, x_purity, training=True)
b_adv_success = tf.logical_and(b_correct_benign, tf.logical_not(b_correct_adv))
# use_mask = tf.cast(tf.concat([b_correct_benign, b_adv_success], axis=0), tf.float32)

# %%%
f_correct_benign = tf.cast(b_correct_benign, tf.float32)
f_correct_benign /= tf.reduce_mean(f_correct_benign)
f_adv_success = tf.cast(b_adv_success, tf.float32)
f_adv_success /= tf.reduce_mean(f_adv_success)
use_mask = tf.concat([f_correct_benign, f_adv_success], axis=0)

saver = tf.train.Saver()

# Batches are half benign and half adversarial
labels = tf.concat([tf.zeros(batch_size, dtype=tf.int64), tf.ones(batch_size, dtype=tf.int64)], axis=0)
loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
loss = loss * use_mask # only learn from correct benign and successful adv.

opt = tf.train.AdamOptimizer(learning_rate)
train_step = opt.minimize(loss)

pred = tf.argmax(logits, axis=1)
accuracy = tf.reduce_mean(tf.cast(tf.equal(pred, labels), tf.float32))

sess = tf.Session()

# Load data
print 'load, preprocess, init' # %%%
def load_seq(template, count):
    return np.asarray([np.load(template % j) for j in range(count)])
def load_dist(setname):
    dist = load_seq('gxr3_%s/%s%%d_dist.npy' % (modelname, setname), 450)
    # dist = np.load('gxr3big_mnist/%s/%s_dist.npy' % (modelname, setname), 'r')
    dist = np.sort(dist, axis=1)
    return dist
dist_benign = load_dist('none')
dist_adv = load_dist(adv_set)
def compute_purity(b):
    b = b[b < num_classes]
    c = np.bincount(b, minlength=num_classes)
    cs = np.sort(c)[::-1]
    cscs = np.cumsum(cs)
    prop = cscs / float(max(1, len(b)))
    return prop
def load_purity(setname):
    boundary = load_seq('gxr3_%s/%s%%d_boundary.npy' % (modelname, setname), 450)
    # boundary = np.load('gxr3big_mnist/%s/%s_boundary.npy' % (modelname, setname), 'r')
    purity = np.asarray([compute_purity(b) for b in boundary])
    return purity
purity_benign = load_purity('none')
purity_adv = load_purity(adv_set)
correctness_benign = np.load('correctness_%s/%s.npy' % (modelname, 'none'), 'r')
correctness_adv = np.load('correctness_%s/%s.npy' % (modelname, adv_set), 'r')

train_split = 350

sess.run(di.initializer, feed_dict={
    p_dist_benign: dist_benign[:train_split],
    p_purity_benign: purity_benign[:train_split, :purity_count],
    p_dist_adv: dist_adv[:train_split],
    p_purity_adv: purity_adv[:train_split, :purity_count],
    p_correct_benign: correctness_benign[:train_split],
    p_correct_adv: correctness_adv[:train_split],
})
sess.run(tf.global_variables_initializer())
print 'done' # %%%

# Run training
for i in range(train_steps):
    train_acc, _ = sess.run([accuracy, train_step])
    if i % 50 == 0: # ceil(train_split / batch_size)
        print 'step', i, 'train accuracy', train_acc

# Save result
saver.save(sess, 'classifier_models/%s_%s' % (modelname, adv_set))

# # below is invalid due to dropout \:
# test_acc = sess.run(accuracy, feed_dict={
#     b_dist_benign: dist_benign[train_split:],
#     b_purity_benign: purity_benign[train_split:, :purity_count],
#     b_dist_adv: dist_adv[train_split:],
#     b_purity_adv: purity_adv[train_split:, :purity_count],
# })
# print 'test accuracy', test_acc
