import warpctc_tensorflow
from collections import namedtuple

import numpy as np
import tensorflow as tf
import six
#import data

from tensorflow.python.training import moving_averages
from tensorflow.python.ops import functional_ops

from utils import ResidualWrapper

HParams = namedtuple('HParams',
                     'batch_size, num_classes, min_lrn_rate, lrn_rate, '
                     'num_layers, weight_decay_rate, '
                     'relu_leakiness, optimizer')


class ASRModel(object):
    
    def __init__(self, hps, mode, mfccs, labels, seq_len, label_len):
        """Model constructor.
        Args:
         hps: Hyperparameters.
         mode: One of 'train' and 'eval'.
        """
        self.hps = hps
        self.mode = mode
        self.mfccs = mfccs
        self.labels = labels
        self.seq_len = seq_len
        self.label_len = label_len
        self._extra_train_ops = []
    
    def build_graph(self):
        """Build a whole graph for the model."""
        self.global_step = tf.contrib.framework.get_or_create_global_step()
        #with tf.variable_scope("Model", initializer = tf.random_uniform_initializer(-0.05,0.05))
        self._build_model()
        if self.mode == 'train':
            self._build_train_op()
        self.summaries = tf.summary.merge_all()
    
    def _stride_arr(self, stride1, stride2):
        """Map a stride scalar to the stride array for tf.nn.conv2d."""
        return [1, stride1, stride2, 1]
    
    def _build_model(self):
        # Hyper-parameters
        with tf.variable_scope('init'):
            x = self._conv2('conv', self.mfccs, 7, 7, 3, 32, self._stride_arr(2,1))
            x = self._batch_norm('init_bn', x)
            x = self._relu(x, self.hps.relu_leakiness)
        with tf.variable_scope('conv1'):
            x = self._conv2('conv1', x, 5, 5, 32, 32, self._stride_arr(2,1))
            #x = self._batch_norm('bn1', x)
            #x = self._relu(x, self.hps.relu_leakiness)
        """    
        with tf.variable_scope('conv2'):
            x = self._conv2('conv2', x, 3, 3, 32, 32, self._stride_arr(1,1))   
            #x = self._batch_norm('bn2', x)
            #x = self._relu(x, self.hps.relu_leakiness)
        """
        with tf.variable_scope('res_unit0_0'):
            x = self._residual(x, 32, 64, self._stride_arr(1,1), False)
            
        for i in six.moves.range(1, 5):
            with tf.variable_scope('res_unit0_%d' % i):
                x = self._residual(x, 64, 64, self._stride_arr(1,1), False)
        
        with tf.variable_scope('conv_bn'):
            x = self._batch_norm('conv_bn', x)
            x = tf.tanh(x,name = 'conv_tanh')
            #x = self._relu(x, self.hps.relu_leakiness)
        
        #x = tf.nn.dropout(x, keep_prob = 0.5)
        #x = tf.nn.max_pool(x, [1,1,3,1], [1,1,2,1], padding='SAME')
        
        batch_size = self.hps.batch_size
        in_dim = 64*13
        hiden_dim = 1024
        pad_val = tf.cast((hiden_dim - in_dim)/2, tf.int32)
        x = tf.reshape(x, shape=[batch_size, -1, in_dim]) #with shape batch x seq_length x features
        
        #x = tf.pad(tensor=x, paddings = [[0, 0], [0, 0], [pad_val, pad_val]])
        
        seq_len = self._cal_seq_len(self.seq_len)
        
        sparse_labels, vals_sparse, label_len = self._ctc_label_dense_to_sparse2(labels=self.labels, seq_len = seq_len, label_lengths = self.label_len)
        #labels = self._ctc_label_dense_to_sparse(labels=self.labels, label_lengths=seq_len)
        #vals_sparse, seq_len, cur_len = self._ctc_helper(labels=self.labels, init_len = self.seq_len, label_lengths = seq_len)
        """
        x, _ = self._Bidirectional_Rnn_Multilayer(inputx=x, n_cell_dim=hiden_dim, seq_len=seq_len, name_scope='brnn', num_layers=self.hps.num_layers, is_bn=False)  
        
        """
        #is_resnet = [False, True, True,True]
        #is_bn = [False, True, True, True]
        with tf.variable_scope('rnn_unit_0'):
            x, _ = self._Undirectional_Rnn_Onelayer(inputx=x, n_cell_dim=hiden_dim, seq_len=seq_len, name_scope='brnn', is_pad = True, pad_val = pad_val, is_res =True)
        
        for i in six.moves.range(1, self.hps.num_layers):
            with tf.variable_scope('rnn_unit_%d' % i):
                x, _ = self._Undirectional_Rnn_Onelayer(inputx=x, n_cell_dim=hiden_dim, seq_len=seq_len, name_scope='brnn',  is_pad = False, pad_val = pad_val, is_res = True)    
        

        # Reshaping to apply the same weights over the timesteps
        x = tf.reshape(x, [-1, hiden_dim])
        with tf.variable_scope('logit'):
            logits = self._fully_connected_v2(x, 'logit' , self.hps.num_classes)
            # Reshaping back to the original shape
            logits = tf.reshape(logits, [self.hps.batch_size, -1, self.hps.num_classes])
            # Time major with seq_length x batch x features
            logits = tf.transpose(logits, (1, 0, 2))
        

        loss = tf.nn.ctc_loss(sparse_labels, logits, seq_len) #change to label_len gets bad result, seems that ``seq_len'' is the right choice, belows the same. 
        #loss = warpctc_tensorflow.ctc(activations=logits, flat_labels=vals_sparse, label_lengths=label_len, input_lengths=seq_len, blank_label=self.hps.num_classes-1)

        self.cost = 1.0*tf.reduce_mean(loss)
        #self.cost = tf.cond(self.cost>=5e3, lambda:tf.constant(500.0), lambda:self.cost)
        self.cost += self._decay()
        tf.summary.scalar('cost', self.cost)
        # Option 2: tf.contrib.ctc.ctc_beam_search_decoder
        # (it's slower but you'll get better results)
        self.decoded, self.log_prob = tf.nn.ctc_greedy_decoder(logits, seq_len, merge_repeated=True)
        #self.decoded, self.log_prob = tf.nn.ctc_beam_search_decoder(logits, seq_len)
        # Inaccuracy: label error rate, or: Levenshtein distance
        self.ler = tf.reduce_mean(tf.edit_distance(tf.cast(self.decoded[0], tf.int32), sparse_labels))
        
        if self.mode == 'eval':
            self.predictions = tf.sparse_tensor_to_dense(self.decoded[0])
            self.truth = tf.sparse_tensor_to_dense(sparse_labels)

      
    def _cal_seq_len(self, seq_len):
        seq_len = tf.ceil(tf.to_float(seq_len)/2)
        seq_len = tf.ceil((seq_len)/2)
        #seq_len = tf.ceil((seq_len)/2)
        return tf.to_int32(seq_len)
    
    def _build_train_op(self):
        """Build training specific ops for the graph."""
        self.lrn_rate = tf.constant(self.hps.lrn_rate, tf.float32)
        tf.summary.scalar('learning rate', self.lrn_rate)
        
        trainable_variables = tf.trainable_variables()
        grads = tf.gradients(self.cost, trainable_variables)
        
        grads,_ = tf.clip_by_global_norm(grads, 100.0)
        #grads = tf.clip_by_value(grads, -0.1, 0.1) 
        
        if self.hps.optimizer == 'sgd':
            optimizer = tf.train.GradientDescentOptimizer(self.lrn_rate)
        elif self.hps.optimizer == 'mom':
            optimizer = tf.train.MomentumOptimizer(self.lrn_rate, 0.9)
        elif self.hps.optimizer == 'adam':
            optimizer = tf.train.AdamOptimizer(self.lrn_rate)
        
        apply_op = optimizer.apply_gradients(
            zip(grads, trainable_variables),
            global_step=self.global_step, name='train_step')
        
        train_ops = [apply_op] + self._extra_train_ops
        self.train_op = tf.group(*train_ops)
    
    # TODO(xpan): Consider batch_norm in contrib/layers/python/layers/layers.py
    def _batch_norm(self, name, x):
        """Batch normalization."""
        with tf.variable_scope(name):
            params_shape = [x.get_shape()[-1]]
    
            beta = tf.get_variable(
              'beta', params_shape, tf.float32,
              initializer=tf.constant_initializer(0.0, tf.float32))
            gamma = tf.get_variable(
              'gamma', params_shape, tf.float32,
              initializer=tf.constant_initializer(1.0, tf.float32))
            
            #beta = Qmf_quan(beta, 4, 7)
            #gamma = Qmf_quan(gamma, 4, 7)
            
        if self.mode == 'train':
            mean, variance = tf.nn.moments(x, [0, 1, 2], name='moments')
    
            moving_mean = tf.get_variable(
                'moving_mean', params_shape, tf.float32,
                initializer=tf.constant_initializer(0.0, tf.float32),
                trainable=False)
            moving_variance = tf.get_variable(
                'moving_variance', params_shape, tf.float32,
                initializer=tf.constant_initializer(1.0, tf.float32),
                trainable=False)
            
            #moving_mean = Qmf_quan(moving_mean, 4, 7)
            #moving_variance = Qmf_quan(moving_variance, 4, 7)
            self._extra_train_ops.append(moving_averages.assign_moving_average(
                moving_mean, mean, 0.9))
            self._extra_train_ops.append(moving_averages.assign_moving_average(
                moving_variance, variance, 0.9))
        else:
            mean = tf.get_variable(
                'moving_mean', params_shape, tf.float32,
                initializer=tf.constant_initializer(0.0, tf.float32),
                trainable=False)
            variance = tf.get_variable(
                'moving_variance', params_shape, tf.float32,
                initializer=tf.constant_initializer(1.0, tf.float32),
                trainable=False)
            #mean = Qmf_quan(mean, 4, 7)
            #variance = Qmf_quan(variance, 4, 7)
            tf.summary.histogram(mean.op.name, mean)
            tf.summary.histogram(variance.op.name, variance)
        # elipson used to be 1e-5. Maybe 0.001 solves NaN problem in deeper net.
        y = tf.nn.batch_normalization(
              x, mean, variance, beta, gamma, 0.001)
        y.set_shape(x.get_shape())
        return y
    """
    input is expected to be a two-dimension matrix, not a high dimension tensor.
    """
    def _batch_norm_vec(self, name, x):
        """Batch normalization."""
        with tf.variable_scope(name):
            params_shape = [x.get_shape()[-1]]
    
            beta = tf.get_variable(
              'beta', params_shape, tf.float32,
              initializer=tf.constant_initializer(0.0, tf.float32))
            gamma = tf.get_variable(
              'gamma', params_shape, tf.float32,
              initializer=tf.constant_initializer(1.0, tf.float32))
            
            #beta = Qmf_quan(beta, 4, 7)
            #gamma = Qmf_quan(gamma, 4, 7)
            
        if self.mode == 'train':
            mean, variance = tf.nn.moments(x, [0], name='moments')
    
            moving_mean = tf.get_variable(
                'moving_mean', params_shape, tf.float32,
                initializer=tf.constant_initializer(0.0, tf.float32),
                trainable=False)
            moving_variance = tf.get_variable(
                'moving_variance', params_shape, tf.float32,
                initializer=tf.constant_initializer(1.0, tf.float32),
                trainable=False)
            #moving_mean = Qmf_quan(moving_mean, 4, 7)
            #moving_variance = Qmf_quan(moving_variance, 4, 7)
            self._extra_train_ops.append(moving_averages.assign_moving_average(
                moving_mean, mean, 0.9))
            self._extra_train_ops.append(moving_averages.assign_moving_average(
                moving_variance, variance, 0.9))
        else:
            mean = tf.get_variable(
                'moving_mean', params_shape, tf.float32,
                initializer=tf.constant_initializer(0.0, tf.float32),
                trainable=False)
            variance = tf.get_variable(
                'moving_variance', params_shape, tf.float32,
                initializer=tf.constant_initializer(1.0, tf.float32),
                trainable=False)
            #mean = Qmf_quan(mean, 4, 7)
            #variance = Qmf_quan(variance, 4, 7)
            tf.summary.histogram(mean.op.name, mean)
            tf.summary.histogram(variance.op.name, variance)
        # elipson used to be 1e-5. Maybe 0.001 solves NaN problem in deeper net.
        y = tf.nn.batch_normalization(
              x, mean, variance, beta, gamma, 0.001)
        y.set_shape(x.get_shape())
        return y
  
    def _residual(self, x, in_filter, out_filter, stride,
                activate_before_residual=False):
        """Residual unit with 2 sub layers."""
        if activate_before_residual:
            with tf.variable_scope('shared_activation'):
                x = self._batch_norm('init_bn', x)
                x = self._relu(x, self.hps.relu_leakiness)
                orig_x = x
        else:
            with tf.variable_scope('residual_only_activation'):
                orig_x = x
                x = self._batch_norm('init_bn', x)
                x = self._relu(x, self.hps.relu_leakiness)
    
        with tf.variable_scope('sub1'):
            x = self._conv('conv1', x, 3, in_filter, out_filter, stride)
    
        with tf.variable_scope('sub2'):
            x = self._batch_norm('bn2', x)
            x = self._relu(x, self.hps.relu_leakiness)
            x = self._conv('conv2', x, 3, out_filter, out_filter, [1, 1, 1, 1])
    
        with tf.variable_scope('sub_add'):
            if in_filter != out_filter:
                orig_x = tf.nn.avg_pool(orig_x, stride, stride, 'VALID')
                pad_val = (out_filter-in_filter)//2
                orig_x = tf.pad(
                    orig_x, [[0, 0], [0, 0], [0, 0],
                             [pad_val, out_filter-in_filter - pad_val]])
            x += orig_x
    
        tf.logging.debug('image after unit %s', x.get_shape())
        return x
    
    def _ctc_label_dense_to_sparse2(self, labels, seq_len, label_lengths): #label_lengths, 
        """
        TODO lists:
        1. current approach may not include the 'space'(0 in this case) label ?! While including the 0 label will cause 'No valid path found' problem,
        which is generally because of the 'inf' loss calculated.  --solved, we seperate seq_len and label_len. 2017.4.30
        2. how to prevent tha case that seq_len is less than label lengths. (or, why ''not enough seq_len'' despite that seq_len > label_lengths when wrap-ctc is used)
        """
        label_shape = labels.get_shape().as_list()
        #len_shape = label_lengths.get_shape().as_list()[0]
        batch = label_shape[0]
        #assert(batch == len_shape)
        max_len = tf.reduce_max(label_lengths)

        
        """
        cur_len = self._cal_seq_len(max_len)
        cur_len = tf.tile(tf.expand_dims(cur_len,axis=-1),[batch])
        """
        label_lengths = tf.where(tf.less(label_lengths,seq_len),label_lengths,seq_len)

        mask = tf.cast(tf.sequence_mask(label_lengths,max_len), tf.int32)

        
        labels_add = tf.add(labels,1) 
        labels_add = tf.multiply(labels_add, mask)
        where_val = tf.less(tf.constant(0), labels_add) # since '0' here is used as a label as well as the 'padding' value, we need to distinglish these two cases.
        indices = tf.where(where_val)
        
        
        vals_sparse = tf.gather_nd(labels, indices)
        return tf.SparseTensor(tf.to_int64(indices), vals_sparse, tf.to_int64([batch,tf.reduce_max(label_lengths)])), vals_sparse, label_lengths
    
    def _Bidirectional_Rnn_Onelayer(self, inputx, n_cell_dim, seq_len, name_scope, is_bn = False, is_res = False):
        with tf.name_scope(name_scope):
            
            if is_bn:
                lstm_fw_cell = tf.contrib.rnn.GRUCell(n_cell_dim)
                lstm_bw_cell = tf.contrib.rnn.GRUCell(n_cell_dim)
            else:
                # Forward direction cell:
                lstm_fw_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(n_cell_dim, forget_bias=0.0) #, dropout_keep_prob=0.5
                # Backward direction cell:
                lstm_bw_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(n_cell_dim, forget_bias=0.0) #dropout_keep_prob=0.5, 

            outputs, output_states = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=lstm_fw_cell,
                cell_bw=lstm_bw_cell,
                inputs=inputx, 
                dtype=tf.float32,
                time_major=False,
                sequence_length=seq_len)
            fw_x, bw_x = outputs
            outputs = fw_x + bw_x #sum to form the final output.
            if is_bn:
                outputs = self._batch_norm('bn', outputs)
            if is_res:
                outputs += inputx
            
        return outputs, output_states

    def _Undirectional_Rnn_Onelayer(self, inputx, n_cell_dim, seq_len, name_scope, is_pad, pad_val, is_res = True):
        
        with tf.name_scope(name_scope):

            # Forward direction cell:
            #lstm_fw_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(n_cell_dim, forget_bias=0.0) #, dropout_keep_prob=0.5
            lstm_fw_cell =  tf.contrib.rnn.BasicLSTMCell(n_cell_dim, forget_bias=0.0, state_is_tuple=True) #, dropout_keep_prob=0.5
            
            #lstm_fw_cell =  LogQLSTMCell(n_cell_dim, forget_bias=0.0, state_is_tuple=True)

            outputs, output_states = tf.nn.dynamic_rnn(
                cell=lstm_fw_cell,
                inputs=inputx, 
                dtype=tf.float32,
                time_major=False,
                sequence_length=seq_len)

            if is_res:
                if is_pad:
                    inputx = tf.pad(tensor=inputx, paddings = [[0, 0], [0, 0], [pad_val, pad_val]])
                outputs += inputx
            
        return outputs, output_states


    def _Bidirectional_LSTMRnn_Onelayer(self, inputx, n_cell_dim, seq_len, name_scope, is_pad, pad_val, is_res = True):
        with tf.name_scope(name_scope):
            lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(n_cell_dim, forget_bias=0.0, state_is_tuple=True) #, dropout_keep_prob=0.5
            # Backward direction cell:
            lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(n_cell_dim, forget_bias=0.0, state_is_tuple=True)  #dropout_keep_prob=0.5, 
            
            #ResidualWrapper
            
            outputs, output_states = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=lstm_fw_cell,
                cell_bw=lstm_bw_cell,
                inputs=inputx, 
                dtype=tf.float32,
                time_major=False,
                sequence_length=seq_len)

            if is_res:
                if is_pad:
                    inputx = tf.pad(tensor=inputx, paddings = [[0, 0], [0, 0], [pad_val, pad_val]])
                outputs += inputx
            
        return outputs, output_states

     
    def _Bidirectional_Rnn_Multilayer(self, inputx, n_cell_dim, seq_len, name_scope, num_layers = 3, is_bn = False):
        with tf.name_scope(name_scope):
            
            if is_bn:
                lstm_fw_cell = tf.contrib.rnn.GRUCell(n_cell_dim)
                lstm_bw_cell = tf.contrib.rnn.GRUCell(n_cell_dim)
            else:
                # Forward direction cell:
                lstm_fw_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(n_cell_dim, forget_bias=0.0) #, dropout_keep_prob=0.5
                # Backward direction cell:
                lstm_bw_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(n_cell_dim, forget_bias=0.0) #dropout_keep_prob=0.5, 
            
            fw_stack = tf.contrib.rnn.MultiRNNCell([lstm_fw_cell] * num_layers, state_is_tuple=True)
            bw_stack = tf.contrib.rnn.MultiRNNCell([lstm_bw_cell] * num_layers, state_is_tuple=True)
            
            outputs, output_states = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=fw_stack,
                cell_bw=bw_stack,
                inputs=inputx, 
                dtype=tf.float32,
                time_major=False,
                sequence_length=seq_len)
            
            fw_x, bw_x = outputs
            outputs = fw_x + bw_x #sum to form the final output.
            if is_bn:
                outputs = self._batch_norm('bn', outputs)
            
        return outputs, output_states
    
    def _decay(self):
        """L2 weight decay loss."""
        costs = []
        for var in tf.trainable_variables():
            if var.op.name.find(r'DW')>=0 or var.op.name.find(r'weights')>= 0: #kernel is used in LayerNorm LSTM
                costs.append(tf.nn.l2_loss(var))
                # tf.summary.histogram(var.op.name, var)
        sum_loss = tf.add_n(costs)
        tf.summary.scalar('weight_loss', sum_loss)        
        return tf.multiply(self.hps.weight_decay_rate, sum_loss)

    def _conv(self, name, x, filter_size, in_filters, out_filters, strides):
        """Convolution."""
        with tf.variable_scope(name):
            n = filter_size * filter_size * out_filters
            kernel = tf.get_variable(
              'DW', [filter_size, filter_size, in_filters, out_filters],
              tf.float32, initializer=tf.random_normal_initializer(
                  stddev=np.sqrt(2.0/n)))
        return tf.nn.conv2d(x, kernel, strides, padding='SAME')
    
    def _conv2(self, name, x, filter_size1,filter_size2, in_filters, out_filters, strides):
        """Convolution."""
        with tf.variable_scope(name):
            n = filter_size1 * filter_size2 * out_filters
            kernel = tf.get_variable(
              'DW', [filter_size1, filter_size2, in_filters, out_filters],
              tf.float32, initializer=tf.random_normal_initializer(
                  stddev=np.sqrt(2.0/n)))
            
        return tf.nn.conv2d(x, kernel, strides, padding='SAME')
    def _relu(self, x, leakiness=0.0):
        """Relu, with optional leaky support."""
        #return tf.clip_by_value(x,0.0,20.0)
        return tf.where(tf.less(x, 0.0), leakiness * x, x, name='leaky_relu')
        #x1 = tf.where(tf.less(x, 0.0), leakiness * x, x, name='leaky_relu_l')
        #x2 = tf.where(tf.less(20.0, x1), 20 + leakiness * (x1-20), x1, name='leaky_relu_r')
        #return x2

    def _fully_connected(self, x, out_dim):
        """FullyConnected layer for final output."""
        x = tf.reshape(x, [self.hps.batch_size, -1])
        w = tf.get_variable(
            'DW', [x.get_shape()[1], out_dim],
            initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
        b = tf.get_variable('biases', [out_dim],
                            initializer=tf.constant_initializer())
        
        return tf.nn.xw_plus_b(x, w, b)

    def _fully_connected_v2(self, x, name, out_dim):
        """FullyConnected layer for final output."""
        #x = tf.reshape(x, [self.hps.batch_size, -1])
        with tf.variable_scope(name):
            w = tf.get_variable(
            'DW', [x.get_shape()[1], out_dim],
            initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
            b = tf.get_variable('biases', [out_dim],
                            initializer=tf.constant_initializer())
                
        return tf.nn.xw_plus_b(x, w, b)

    def _global_avg_pool(self, x):
        assert x.get_shape().ndims == 4
        return tf.reduce_mean(x, [1, 2])
    
