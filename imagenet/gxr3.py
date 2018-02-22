import sys

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

# usage: python gxr3.py <model> <dataset name> <images path> <index>,...

modelname, dataset, images_path, indices = sys.argv[1:]
indices = np.asarray([int(v) for v in indices.split(',')])

num_images = len(indices)

mags = np.arange(0, 0.5, 0.01, dtype=np.float32)
assert mags[0] == 0
num_mags = len(mags)

max_batch_size = 100

import stuff_v2 as stuff
sess = stuff.get_session()
m = stuff.M(modelname)
m_x = tf.placeholder(shape=(None, ih, iw, ic), dtype=tf.float32)
m_pre_softmax = m.predict(m_x)
m_pred = tf.argmax(m_pre_softmax, axis=1)

num_dirs = 1000
import os
if os.path.isfile('gxr3_dirs.npy'):
    dirs = np.load('gxr3_dirs.npy')
else:
    randmat = np.random.normal(size=(ifeat, num_dirs))
    q, r = np.linalg.qr(randmat)
    assert(q.shape == (ifeat, num_dirs))
    dirs = q.T * np.sqrt(float(ifeat)) # gxr3 operates in RMS error
    np.save('gxr3_dirs.npy', dirs)

if not os.path.isdir('gxr3_%s' % modelname):
    os.mkdir('gxr3_%s' % modelname)

xs = np.load(images_path, 'r')[indices]
if xs.shape[1] != ifeat:
    xs = xs.reshape((-1, ifeat))

dist = np.zeros(num_dirs, dtype=np.float32)
boundary = np.zeros(num_dirs, dtype=np.uint16)

for j in tqdm.trange(num_images, leave=True):
    index = indices[j]
    orig = xs[j]

    # compute center info
    center_preds = sess.run(m_pred, feed_dict={
        m_x: orig[None].reshape((1, ih, iw, ic)),
    })
    np.save('gxr3_%s/%s%d_center_preds.npy' % (modelname, dataset, index), center_preds)

    # initialize per-direction info
    running = np.arange(num_dirs, dtype=np.int32)

    tr = tqdm.trange(1, num_mags, leave=False)
    tr.set_postfix(running=num_dirs)
    for i in tr:
        mag = mags[i]
        imgs = orig[None] + mag * dirs[running]
        imgs = np.clip(imgs, imin, imax)

        total = len(imgs)
        preds = np.zeros(total, dtype=np.uint16)
        cursor = 0
        while cursor < total:
            end = min(total, cursor + max_batch_size)
            preds[cursor:end] = sess.run(m_pred, feed_dict={
                m_x: imgs[cursor:end].reshape((-1, ih, iw, ic)),
            })
            cursor = end

        homo = np.equal(preds, center_preds.squeeze(0))
        homo_count = np.count_nonzero(homo)
        hetero = np.logical_not(homo)

        stopped = running[hetero]
        dist[stopped] = mag
        boundary[stopped] = preds[hetero]

        tr.set_postfix(running=str(homo_count))

        running = running[homo]

        if homo_count == 0:
            tr.close()
            break

    dist[running] = imax
    boundary[running] = 65535

    np.save('gxr3_%s/%s%d_dist.npy' % (modelname, dataset, index), dist)
    np.save('gxr3_%s/%s%d_boundary.npy' % (modelname, dataset, index), boundary)
