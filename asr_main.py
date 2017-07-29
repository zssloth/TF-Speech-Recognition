import warpctc_tensorflow
import time
import six
import sys

import data
import numpy as np
import asr_model
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('dataset', 'librispeech', 'speech corpus.')
tf.app.flags.DEFINE_string('mode', 'train', 'train or eval.')
tf.app.flags.DEFINE_string('set_name', 'train_delta_*', 'train or eval.')
tf.app.flags.DEFINE_string('train_dir', '/home/zswang/training_logs/asr_model/train',
                           'Directory to keep training outputs.')
tf.app.flags.DEFINE_string('eval_dir', '/home/zswang/training_logs/asr_model/test',
                           'Directory to keep eval outputs.')
tf.app.flags.DEFINE_integer('eval_batch_count', 262,
                            'Number of batches to eval.')
tf.app.flags.DEFINE_bool('eval_once', False,
                         'Whether evaluate the model only once.')
tf.app.flags.DEFINE_string('log_root', '/home/zswang/training_logs/asr_model',
                           'Directory to keep the checkpoints. Should be a '
                           'parent directory of FLAGS.train_dir/eval_dir.')
tf.app.flags.DEFINE_integer('num_gpus', 1,
                            'Number of gpus used for training. (0 or 1)')


init_lr_rate = 2e-3
mim_lr = 6e-6
n_step = 891
"""
bp1.tar.gz, test_sort, ~7k steps, loss: 52.333, ler: 0.167, ~50% wer --> 20 mfcc feature with sorted indices and normalization.
    -- without normalization get similar results.
sudo python asr_main.py  --train_dir=/home/datadisk/zswang/training_logs/asr_model/train --eval_dir=/home/datadisk/zswang/training_logs/asr_model/eval \
    --log_root=/home/datadisk/zswang/training_logs/asr_model --set_name=test_sort
test_sort_6.9wer.tar.gz, test_sort, 36 mfcc features with normalization, 6.9% wer !! -->test_sort_6.9wer_training_logs.tar.gz

test_sort_6.9wer_training_logs.tar.gz, 36 mfcc feature with 3convs + 5residual + 3 layernorm_lstm,~18.565k steps (<11 epochs, too slow !!), 36.1% wer on testset.
    -- with greddy decoder: ~40% wer
    -- wer in the training set, only eval 2620 samples:~10.1% wer

sudo python asr_main.py  --train_dir=/home/datadisk/zswang/training_logs/asr_model/train --eval_dir=/home/datadisk/zswang/training_logs/asr_model/eval  --log_root=/home/datadisk/zswang/training_logs/asr_model --set_name=train_sort --mode=eval

"""
def train(hps):
    """Training loop."""
    mfccs, labels, seq_len, label_len = data.build_input3(batch_size=hps.batch_size, set_name=FLAGS.set_name, mode=FLAGS.mode,num_mfcc=13)
    
    #seq_len = [tf.shape(mfccs)[1]]
    model = asr_model.ASRModel(hps, FLAGS.mode, mfccs, labels, seq_len, label_len)
    model.build_graph()
    
    param_stats = tf.contrib.tfprof.model_analyzer.print_model_analysis(
      tf.get_default_graph(),
      tfprof_options=tf.contrib.tfprof.model_analyzer.
          TRAINABLE_VARS_PARAMS_STAT_OPTIONS)
    sys.stdout.write('total_params: %d\n' % param_stats.total_parameters)
    
    tf.contrib.tfprof.model_analyzer.print_model_analysis(
      tf.get_default_graph(),
      tfprof_options=tf.contrib.tfprof.model_analyzer.FLOAT_OPS_OPTIONS)

    precision = model.ler

    summary_hook = tf.train.SummarySaverHook(
      save_steps=n_step,
      output_dir=FLAGS.train_dir,
      summary_op=tf.summary.merge([model.summaries,
                                   tf.summary.scalar('ler', precision)]))

    logging_hook = tf.train.LoggingTensorHook(
      tensors={'step': model.global_step,
               'loss': model.cost,
               'precision': precision},
      every_n_iter=100)

    class _LearningRateSetterHook(tf.train.SessionRunHook):
        """Sets learning_rate based on global step."""
        
        def begin(self):
            self._lrn_rate = init_lr_rate
        
        def before_run(self, run_context):
            return tf.train.SessionRunArgs(
              model.global_step,  # Asks for global step value.
              feed_dict={model.lrn_rate: self._lrn_rate
                         })  # Sets learning rate
        
        def after_run(self, run_context, run_values):
            train_step = run_values.results
            #self._lrn_rate = init_lr_rate*(0.708**(train_step/891)) #1783
            if train_step <= n_step*4:
                self._lrn_rate = init_lr_rate
            elif train_step <= n_step*30:
                self._lrn_rate = init_lr_rate*(0.8**((train_step - n_step*4)/n_step)) #1783   
            else:
                self._lrn_rate = mim_lr
            """
            elif train_step <= 1783*10:
                self._lrn_rate = init_lr_rate*0.2
            elif train_step <= 1783*15:
                self._lrn_rate = init_lr_rate*0.04
            elif train_step <= 1783*20:
                self._lrn_rate = init_lr_rate*0.008
            """
    #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.50)
    with tf.train.MonitoredTrainingSession(
      checkpoint_dir=FLAGS.log_root,
      hooks=[logging_hook, _LearningRateSetterHook()],
      chief_only_hooks=[summary_hook],
      # Since we provide a SummarySaverHook, we need to disable default
      # SummarySaverHook. To do that we set save_summaries_steps to 0.
      save_summaries_steps=0,
      save_checkpoint_secs = 1200*4,
      config=tf.ConfigProto(allow_soft_placement=True)) as mon_sess:
        while not mon_sess.should_stop():
            mon_sess.run(model.train_op)

def evaluate(hps):
    mfccs, labels, seq_len, label_len = data.build_input3(batch_size=hps.batch_size, set_name=FLAGS.set_name, mode=FLAGS.mode, num_mfcc=13)
    model = asr_model.ASRModel(hps, FLAGS.mode, mfccs, labels, seq_len, label_len)
    model.build_graph()
    saver = tf.train.Saver()
    summary_writer = tf.summary.FileWriter(FLAGS.eval_dir)
    #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.50) #, gpu_options=gpu_options
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    tf.train.start_queue_runners(sess)
    
    def _wer(original, result):
        r"""
        The WER is defined as the editing/Levenshtein distance on word level
        divided by the amount of words in the original text.
        In case of the original having more words (N) than the result and both
        being totally different (all N words resulting in 1 edit operation each),
        the WER will always be 1 (N / N = 1).
        """
        # The WER ist calculated on word (and NOT on character) level.
        # Therefore we split the strings into words first:
        original = original.split()
        result = result.split()
        return _levenshtein(original, result) / float(len(original))
    
    def _wers(originals, results):
        count = len(originals)
        rates = []
        mean = 0.0
        assert count == len(results)
        for i in range(count):
            rate = _wer(originals[i], results[i])
            mean = mean + rate
            rates.append(rate)
        return rates, mean / float(count)
    
    # The following code is from: http://hetland.org/coding/python/levenshtein.py
    
    # This is a straightforward implementation of a well-known algorithm, and thus
    # probably shouldn't be covered by copyright to begin with. But in case it is,
    # the author (Magnus Lie Hetland) has, to the extent possible under law,
    # dedicated all copyright and related and neighboring rights to this software
    # to the public domain worldwide, by distributing it under the CC0 license,
    # version 1.0. This software is distributed without any warranty. For more
    # information, see <http://creativecommons.org/publicdomain/zero/1.0>
    
    def _levenshtein(a,b):
        "Calculates the Levenshtein distance between a and b."
        n, m = len(a), len(b)
        if n > m:
            # Make sure n <= m, to use O(min(n,m)) space
            a,b = b,a
            n,m = m,n
    
        current = list(range(n+1))
        for i in range(1,m+1):
            previous, current = current, [i]+[0]*n
            for j in range(1,n+1):
                add, delete = previous[j]+1, current[j-1]+1
                change = previous[j-1]
                if a[j-1] != b[i-1]:
                    change = change + 1
                current[j] = min(add, delete, change)
        return current[n]
        
    best_wer = 1.0
    while True:
        try:
            ckpt_state = tf.train.get_checkpoint_state(FLAGS.log_root)
        except tf.errors.OutOfRangeError as e:
            tf.logging.error('Cannot restore checkpoint: %s', e)
            continue
        if not (ckpt_state and ckpt_state.model_checkpoint_path):
            tf.logging.info('No model to eval yet at %s', FLAGS.log_root)
            continue
        tf.logging.info('Loading checkpoint %s', ckpt_state.model_checkpoint_path)
        saver.restore(sess, ckpt_state.model_checkpoint_path)
    
        total_prediction, wer_sum = 0, 0
        for _ in six.moves.range(FLAGS.eval_batch_count):
            (summaries, loss, decoded, truth, train_step) = sess.run(
            [model.summaries, model.cost, model.predictions, model.truth, model.global_step])
    
            total_prediction += hps.batch_size
            
            decoded_str, truth_str = data.batch_index2str(decoded, truth)
            #print decoded_str[0]
            #print truth_str[0]
            #print '--------------------------------------'
            _, mean_wer = _wers(truth_str,decoded_str)
            wer_sum += mean_wer*hps.batch_size
            #print mean_wer
            #decoded = tf.sparse_to_dense(sparse_indices, output_shape, sparse_values, default_value, validate_indices, name)
        
        
        wer_final = 1.0 * wer_sum / total_prediction
        best_ler = min(wer_final, best_wer)
    
        precision_summ = tf.Summary()
        precision_summ.value.add(
          tag='wer', simple_value=wer_final)
        summary_writer.add_summary(precision_summ, train_step)
        best_precision_summ = tf.Summary()
        best_precision_summ.value.add(
          tag='Best wer', simple_value=best_wer)
        summary_writer.add_summary(best_precision_summ, train_step)
        summary_writer.add_summary(summaries, train_step)
        tf.logging.info('loss: %.3f, wer: %.3f, best wer: %.3f' %
                      (loss, wer_final, best_ler))
        summary_writer.flush()
    
        if FLAGS.eval_once:
            break
    
        time.sleep(30)

def main(_):
    if FLAGS.num_gpus == 0:
        dev = '/cpu:0'
    elif FLAGS.num_gpus == 1:
        dev = '/gpu:0'
    else:
        raise ValueError('Only support 0 or 1 gpu.')
    
    if FLAGS.mode == 'train':
        batch_size = 32
    elif FLAGS.mode == 'eval':
        batch_size = 10
    
    if FLAGS.dataset == 'librispeech':
        num_classes = data.voca_size
    
    hps = asr_model.HParams(batch_size=batch_size,
                               num_classes=num_classes,
                               min_lrn_rate=0.000001,
                               lrn_rate=init_lr_rate,
                               num_layers=5,
                               weight_decay_rate=0.001,#0.0002
                               relu_leakiness=0.1,
                               optimizer='adam')
    
    with tf.device(dev):
        if FLAGS.mode == 'train':
            train(hps)
        elif FLAGS.mode == 'eval':
            evaluate(hps)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
