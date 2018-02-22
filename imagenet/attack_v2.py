import numpy as np
import tensorflow as tf
import tqdm

# usage attack.py model_name attack_name offset count

import sys
model_name, attack_name, offset, count = sys.argv[1:]
offset = int(offset)
count = int(count)

import stuff_v2 as stuff
nhwc, labels = stuff.load_data()
nhwc_slice = nhwc[offset:offset+count]
labels_slice = labels[offset:offset+count]

sess = stuff.get_session()
m = stuff.M(model_name)

if 'none' == attack_name:
    imgs = nhwc_slice.astype(np.float32) / 255.
    np.save('%s_%s_%d_%d.npy' % (model_name, attack_name, offset, count), imgs)
elif 'opt' == attack_name:
    batch_size = 20
    import l2_attack
    a = l2_attack.CarliniL2(sess, m, batch_size=batch_size,
                            confidence=0, targeted=False,
                            learning_rate=1e-3, max_iterations=1000, binary_search_steps=4, initial_const=1e-1,
                            boxmin=0., boxmax=1.)
    nbhwc = nhwc_slice.reshape((-1, batch_size, m.image_size, m.image_size, m.num_channels))
    labels_batches = labels_slice.reshape((-1, batch_size))
    for i in tqdm.trange(nbhwc.shape[0]):
        offset_batch = offset + i * batch_size
        imgs = nbhwc[i].astype(np.float32) / 255.
        labels = labels_batches[i]
        labels_one_hot = np.zeros((batch_size, m.num_labels), dtype=np.float32)
        labels_one_hot[range(batch_size), labels] = 1.
        adv_imgs = a.attack_batch(imgs, labels_one_hot)
        np.save('%s_%s_%d_%d.npy' % (model_name, attack_name, offset_batch, batch_size), adv_imgs)
elif 'optens' == attack_name:
    batch_size = 1
    import l2_attack_ensemble
    a = l2_attack_ensemble.CarliniL2Ensemble(sess, m, batch_size=batch_size,
                                             confidence=0, targeted=False,
                                             learning_rate=1e-3, max_iterations=1000,
                                             binary_search_steps=4, initial_const=1e-1)
    nbhwc = nhwc_slice.reshape((-1, batch_size, m.image_size, m.image_size, m.num_channels))
    labels_batches = labels_slice.reshape((-1, batch_size))
    for i in tqdm.trange(nbhwc.shape[0]):
        offset_batch = offset + i * batch_size
        imgs = nbhwc[i].astype(np.float32) / 255.
        labels = labels_batches[i]
        labels_one_hot = np.zeros((batch_size, m.num_labels), dtype=np.float32)
        labels_one_hot[range(batch_size), labels] = 1.
        adv_imgs = a.attack_batch(imgs, labels_one_hot)
        np.save('%s_%s/parts/%d_%d.npy' % (model_name, attack_name, offset_batch, batch_size), adv_imgs)
elif 'fgsm' == attack_name:
    epsilon = 8. / 255.
    batch_size = 20
    x = tf.placeholder(shape=[None, m.image_size, m.image_size, m.num_channels], dtype=tf.float32)
    y = tf.placeholder(shape=[None], dtype=tf.uint16)
    logits = m.predict(x)
    xent = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=tf.cast(y, tf.int32))
    xent_tot = tf.reduce_sum(xent)
    grad, = tf.gradients(xent_tot, x)
    x_adv = tf.clip_by_value(x + epsilon * tf.sign(grad), 0., 1.)

    nbhwc = nhwc_slice.reshape((-1, batch_size, m.image_size, m.image_size, m.num_channels))
    labels_batches = labels_slice.reshape((-1, batch_size))
    buf = np.zeros(nhwc_slice.shape, dtype=np.float32)
    buf_batches = buf.reshape(nbhwc.shape)
    for i in tqdm.trange(nbhwc.shape[0]):
        offset_batch = offset + i * batch_size
        imgs = nbhwc[i].astype(np.float32) / 255.
        labels = labels_batches[i]
        buf_batches[i] = sess.run(x_adv, feed_dict={x: imgs, y: labels})
    np.save('%s_%s_%d_%d.npy' % (model_name, attack_name, offset, count), buf)
else:
    raise Exception('unknown attack')

