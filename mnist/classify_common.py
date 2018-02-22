import tensorflow as tf

def m(dist_sorted, purity, training):
    t = dist_sorted[:, :, None]
    t = tf.layers.conv1d(t, filters=8, kernel_size=8, strides=4, activation=tf.nn.relu)
    t = tf.layers.dropout(t, rate=0.5, training=training)
    t = tf.layers.conv1d(t, filters=16, kernel_size=8, strides=4, activation=tf.nn.relu)
    t = tf.layers.dropout(t, rate=0.5, training=training)
    t = tf.contrib.layers.flatten(t)
    t = tf.concat([t, purity], axis=1)
    t = tf.layers.dense(t, units=32, activation=tf.nn.relu)
    t = tf.layers.dropout(t, rate=0.5, training=training)
    t = tf.layers.dense(t, units=2)
    return t
