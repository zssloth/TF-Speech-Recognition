from numpy import nan
from pylint.checkers.similar import Similar

from collections import namedtuple

import numpy as np
import tensorflow as tf
import six

def ctc_label_dense_to_sparse(labels, label_lengths, init_len):
    """
    TODO: the number of non-zeros in every row of 'labels' must less than the corresponding value in 'label_lengths'
    """
    label_shape = labels.get_shape().as_list()
    len_shape = label_lengths.get_shape().as_list()[0]
    batch = label_shape[0]
    assert(batch == len_shape)
    max_len = tf.reduce_max(init_len)
    
    cur_len = tf.constant(2.0)
    cur_len = tf.tile(tf.expand_dims(cur_len,axis=-1),[batch])
    
    mask = tf.cast(tf.sequence_mask(label_lengths,max_len), tf.int32)
    #labels_split, buf  = tf.split(labels, [max_len,-1], axis=1)
    #buf = tf.reduce_sum(buf)
    #tf.summary.scalar('buf', buf)
    
    labels = tf.multiply(labels, mask)
    #min_len = tf.arg_min(label_lengths, dimension=0)
    #mask = tf.fill(label_shape, 0)
    
    where_val = tf.less(tf.constant(0), labels)
    indices = tf.where(where_val)
    
    vals_sparse = tf.gather_nd(labels, indices)
    return indices, vals_sparse, tf.shape(labels), cur_len, mask

dense = tf.constant([[1, 2, 3, 4],[5, 6, 0, 0],[9, 10, 11, 0]])
lens = tf.constant([3,2,2])
len_init = tf.constant([4,2,3])



sess = tf.Session()


ope = ctc_label_dense_to_sparse(dense, lens, len_init)
a, b, c, _, d = sess.run(ope)

print a
print b 
print c
print d 