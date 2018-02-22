import sys

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

# usage: python gxr3.py <model> <step>,... <dataset name> <images path> <index>,...

modelname, steps, dataset, images_path, indices = sys.argv[1:]
steps = [int(v) for v in steps.split(',')]
indices = np.asarray([int(v) for v in indices.split(',')])

num_steps = len(steps)
num_images = len(indices)

mags = np.arange(0, 127.5, 2, dtype=np.float32)
assert mags[0] == 0
num_mags = len(mags)

checkpoint_dirs = {
    'madry_robust': 'models/adv_trained/checkpoint-%d',
    'madry_robust_keep': 'models/robust_with_intermediates/checkpoint-%d',
    'madry_nat': 'models/naturally_trained/checkpoint-%d'
}
if modelname not in checkpoint_dirs:
    raise Exception('unrecognized model')
checkpoint_dir = checkpoint_dirs[modelname]

if modelname in ['madry_robust', 'madry_robust_keep', 'madry_nat']:
    import model
    m = model.Model('eval')
    m_x = m.x_input # n*ifeat
    m_y = m.y_input
    m_prob = tf.nn.softmax(m.pre_softmax)
    m_loss = m.xent
else:
    raise Exception('unrecognized model')
m_pred = tf.argmax(m_prob, axis=1)

saver = tf.train.Saver()
sess = tf.Session()

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
# dirs = np.concatenate([dirs, -dirs])

if not os.path.isdir('gxr3_%s' % modelname):
    os.mkdir('gxr3_%s' % modelname)

xs = np.load(images_path, 'r')[indices]
if xs.shape[1] != ifeat:
    xs = xs.reshape((-1, ifeat))

dist = np.zeros(num_dirs, dtype=np.float32)
boundary = np.zeros(num_dirs, dtype=np.uint8)

for step in steps:
    saver.restore(sess, checkpoint_dir % step)

    for j in tqdm.trange(num_images, leave=False):
        index = indices[j]
        orig = xs[j]

        # compute center info
        center_preds = sess.run(m_pred, feed_dict={
            m_x: orig[None].reshape((1, ih, iw, ic)),
        })
        np.save('gxr3_%s/step%d_%s%d_center_preds.npy' % (modelname, step, dataset, index), center_preds)

        # initialize per-direction info
        running = np.arange(num_dirs, dtype=np.int32)

        tr = tqdm.trange(1, num_mags, leave=False)
        tr.set_postfix(running=num_dirs, step=step)
        for i in tr:
            mag = mags[i]
            imgs = orig[None] + mag * dirs[running]
            imgs = np.clip(imgs, imin, imax)
            preds = sess.run(m_pred, feed_dict={
                m_x: imgs.reshape((-1, ih, iw, ic)),
            })

            homo = np.equal(preds, center_preds.squeeze(0))
            homo_count = np.count_nonzero(homo)
            hetero = np.logical_not(homo)

            stopped = running[hetero]
            dist[stopped] = mag
            boundary[stopped] = preds[hetero]

            tr.set_postfix(running=str(homo_count), step=str(step))

            if homo_count == 0:
                tr.close()
                break

            running = running[homo]

        dist[running] = imax
        boundary[running] = 255

        np.save('gxr3_%s/step%d_%s%d_dist.npy' % (modelname, step, dataset, index), dist)
        np.save('gxr3_%s/step%d_%s%d_boundary.npy' % (modelname, step, dataset, index), boundary)
