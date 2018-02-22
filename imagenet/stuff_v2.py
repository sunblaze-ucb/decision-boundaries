import sys

import numpy as np
import tensorflow as tf

sess = None

def get_session():
    global sess
    if sess is None:
        sess = tf.Session()
    return sess

ckpt_dir = '../imagenet/models'
def apply_model(x, net_fn, arg_scope_cls, ckpt_name, no_background=False, **kwargs):
    mv_start = len(tf.global_variables())
    arg_scope = arg_scope_cls()
    print 'building model', net_fn.__name__,
    sys.stdout.flush()
    with tf.contrib.slim.arg_scope(arg_scope):
        logits, end_points = net_fn(x, is_training=False, **kwargs)
    print 'done'
    mv_end = len(tf.global_variables())
    model_vars = tf.global_variables()[mv_start:mv_end]

    # for v in model_vars: # %%%
    #     print v
    # exit(1) # %%%

    saver = tf.train.Saver(model_vars)
    print 'restoring model', ckpt_name,
    sys.stdout.flush()
    saver.restore(sess, ckpt_dir + '/' + ckpt_name)
    print 'done'
    if no_background:
        logits = tf.concat([tf.fill([tf.shape(logits)[0], 1], -1e5), logits], axis=1)
    return logits

class M:
    def __init__(self, model_name):
        self.model_name = model_name
        self.image_size = 299
        self.num_channels = 3
        self.num_labels = 1001
    def predict(self, x):
        x_inception = x * 2 - 1
        # x_224 = tf.image.resize_bilinear(x, [224, 244])
        # x_inception_224 = x_224 * 2 - 1
        # x_vgg = x_224 * 255 - VGG_NHWC_MEAN

        if 'inception_resnet_v2' == self.model_name:
            from nets import inception_resnet_v2
            return apply_model(x_inception,
                               inception_resnet_v2.inception_resnet_v2,
                               inception_resnet_v2.inception_resnet_v2_arg_scope,
                               'inception_resnet_v2_2016_08_30.ckpt')

        if 'inception_v4' == self.model_name:
            from nets import inception_v4
            return apply_model(x_inception,
                               inception_v4.inception_v4,
                               inception_v4.inception_v4_arg_scope,
                               'inception_v4.ckpt')

        # if 'mobilenet_v1' == self.model_name:
        #     from nets import mobilenet_v1
        #     return apply_model(x_inception_224,
        #                        mobilenet_v1.mobilenet_v1,
        #                        mobilenet_v1.mobilenet_v1_arg_scope,
        #                        'mobilenet_v1_1.0_224.ckpt',
        #                        num_classes=1001)

        # if 'resnet_v2_152' == self.model_name:
        #     raise Exception('broken. joystick or binoculars')
        #     from nets import resnet_v2
        #     return apply_model(x_vgg,
        #                        resnet_v2.resnet_v2_152,
        #                        resnet_v2.resnet_arg_scope,
        #                        'resnet_v2_152.ckpt',
        #                        num_classes=1001)

        # if 'vgg_16' == self.model_name:
        #     from nets import vgg
        #     return apply_model(x_vgg,
        #                        vgg.vgg_16,
        #                        vgg.vgg_arg_scope,
        #                        'vgg_16.ckpt',
        #                        no_background=True)

        if 'adv_inception_v3' == self.model_name:
            from nets import inception_v3
            return apply_model(x_inception,
                               inception_v3.inception_v3,
                               inception_v3.inception_v3_arg_scope,
                               'adv_inception_v3.ckpt',
                               num_classes=1001)

        if 'base_inception_model' == self.model_name:
            from nets import inception_v3
            return apply_model(x_inception,
                               inception_v3.inception_v3,
                               inception_v3.inception_v3_arg_scope,
                               'inception_v3.ckpt',
                               num_classes=1001)

        if 'ens_adv_inception_resnet_v2' == self.model_name:
            from nets import inception_resnet_v2
            return apply_model(x_inception,
                               inception_resnet_v2.inception_resnet_v2,
                               inception_resnet_v2.inception_resnet_v2_arg_scope,
                               'ens_adv_inception_resnet_v2.ckpt')

def load_data():
    print 'loading dataset',
    nhwc = np.load('../imagenet/val_5k_images.npy', 'r')
    labels = np.load('../imagenet/val_5k_labels.npy', 'r')
    print 'done'
    return nhwc, labels
