import numpy as np
import tensorflow as tf

import classify_common

dist_dims = 1000
purity_count = 3

# usage: python classify_test.py <modelname> <adv_set> <classifier checkpoint>
num_classes = 10
import sys
modelname, adv_set, classifier_checkpoint = sys.argv[1:]

x_dist = tf.placeholder(shape=[None, dist_dims], dtype=tf.float32)
x_purity = tf.placeholder(shape=[None, purity_count], dtype=tf.float32)
logits = classify_common.m(x_dist, x_purity, training=False)
preds = tf.argmax(logits, axis=1)

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
adv_success = np.logical_and(correctness_benign, np.logical_not(correctness_adv))

train_split = 350
# test_count = 100 - train_split

# Load weights
sess = tf.Session()
saver = tf.train.Saver()
saver.restore(sess, classifier_checkpoint)

# Run test
preds_benign = sess.run(preds, feed_dict={
    x_dist: dist_benign[train_split:],
    x_purity: purity_benign[train_split:, :purity_count],
})
detect_benign = np.equal(preds_benign, 1)
false_positive = np.logical_and(detect_benign, correctness_benign[train_split:])
false_positive_num = np.count_nonzero(false_positive)
false_positive_denom = np.count_nonzero(correctness_benign[train_split:])
print 'false positive', '%.1f' % (100. * false_positive_num / float(false_positive_denom)), 'of', false_positive_denom
print 'indices', np.where(false_positive) # %%%

preds_adv = sess.run(preds, feed_dict={
    x_dist: dist_adv[train_split:],
    x_purity: purity_adv[train_split:, :purity_count],
})
notdetect_adv = np.equal(preds_adv, 0)
false_negative = np.logical_and(notdetect_adv, adv_success[train_split:])
false_negative_num = np.count_nonzero(false_negative)
false_negative_denom = np.count_nonzero(adv_success[train_split:])
print 'false negative', '%.1f' % (100. * false_negative_num / float(false_negative_denom)), 'of', false_negative_denom
print 'indices', np.where(false_negative) # %%%
