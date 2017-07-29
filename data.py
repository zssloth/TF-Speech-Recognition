import tensorflow as tf
import sugartensor as stf
import numpy as np
import csv
import string
import tempfile
from preprocess_data import _data_path
import os

__author__ = 'namju.kim@kakaobrain.com'


# default data path
_data_path = '/home/datadisk/zswang/speech-corpus/' 
#_data_path = '/home/zswang/github/WaveNet-ASR/'
#
# vocabulary table
#

# index to byte mapping
index2byte = [' ', 'a', 'b', 'c', 'd', 'e', 'f', 'g',
              'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q',
              'r', 's', 't', 'u', 'v', 'w', 'x', 'y','z','<EMP>']

# byte to index mapping
byte2index = {}
for i, ch in enumerate(index2byte):
    byte2index[ch] = i

# vocabulary size
voca_size = len(index2byte)


# convert sentence to index list
def str2index(str_):

    # clean white space
    str_ = ' '.join(str_.split())
    # remove punctuation and make lower case
    str_ = str_.translate(None, string.punctuation).lower()

    res = []
    for ch in str_:
        try:
            res.append(byte2index[ch])
        except KeyError:
            # drop OOV
            pass
    return res


# convert index list to string
def index2str(index_list):
    # transform label index to character
    str_ = ''
    for ch in index_list:
        if ch < voca_size-1:
            str_ += index2byte[ch]
        elif ch == voca_size-1:  # <EOS>
            print 'end of sequence'
            break
    return str_
def batch_index2str(decoded, truth):
    batch = np.shape(decoded)[0]
    t,d = [],[]
    for i in range(batch):
        t.append(index2str(truth[i,:]))
        d.append(index2str(decoded[i,:]))
    return d,t
# print list of index list
def print_index(indices):
    for index_list in indices:
        print(index2str(index_list))

# real-time wave to mfcc conversion function
@stf.sg_producer_func
def _load_mfcc(src_list):

    # label, wave_file
    label, mfcc_file, lenb = src_list

    # decode string to integer
    label = np.fromstring(label, np.int32)
    lenb = np.fromstring(lenb, np.int32)

    # load mfcc
    mfcc = np.load(mfcc_file, allow_pickle=False)

    # speed perturbation augmenting
    #mfcc = _augment_speech(mfcc)

    return label, mfcc, lenb

def _augment_speech(mfcc):
    
    # random frequency shift ( == speed perturbation effect on MFCC )
    r = np.random.randint(-2, 2)

    # shifting mfcc
    mfcc = np.roll(mfcc, r, axis=1)
    # zero padding
    if r > 0:
        mfcc[:, :r, :] = 0
    elif r < 0:
        mfcc[:, r:, :] = 0

    return mfcc
def generate_csv(set_name='train'):
    csv_f = open(_data_path + 'preprocess/meta/%s_split.csv' % set_name, 'w+')
    writer = csv.writer(csv_f, delimiter=',')

    with open(_data_path + 'preprocess/meta/%s.csv' % set_name) as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        for row in reader:
            label = np.asarray(row[1:], dtype=np.int)
            writer.writerow(label)
    
def build_input2(batch_size=16, set_name='train_sort', mode='train'):

    # load meta file
    label, mfcc_file, len1 = [], [], []

    with open(_data_path + 'preprocess/meta/%s.csv' % set_name) as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        for i, row in enumerate(reader):
            # mfcc file
            mfcc_file.append(_data_path + 'preprocess/mfcc/' + row[0] + '.npy')
            # label info ( convert to string object for variable-length support )
            #print 'shape is:', np.shape(label_temp), np.shape(indices)
            str1 = np.asarray(row[1:], dtype=np.int32)
            len_buf = np.asarray([len(str1)], dtype=np.int32)
            label.append(str1.tostring())
            #print str1, len_buf
            len1.append(len_buf.tostring())
            if i >= 1999:
                break
    print np.shape(len1)
    # to constant tensor
    label_t = tf.convert_to_tensor(label)
    mfcc_file_t = tf.convert_to_tensor(mfcc_file)
    len_t = tf.convert_to_tensor(len1)

    # create queue from constant tensor
    label_q, mfcc_file_q, len_q \
        = tf.train.slice_input_producer([label_t, mfcc_file_t, len_t], shuffle=False)

    # create label, mfcc queue
    label_q, mfcc_q, len_q = _load_mfcc(source=[label_q, mfcc_file_q, len_q],
                                 dtypes=[tf.int32, tf.float32, tf.int32],
                                 capacity=batch_size*16, num_threads=16)
    
    # create batch queue with dynamic pad
    label_, mfcc_, len_ = tf.train.batch([label_q, mfcc_q, len_q], batch_size,
                                 shapes=[(None,), (20, None), (1,)],
                                 num_threads=16, capacity=batch_size*16,
                                 dynamic_pad=True, allow_smaller_final_batch=False)
    
    #label_ = tf.deserialize_many_sparse(label_, tf.int32)
    # batch * time * dim
    mfcc_ = tf.transpose(mfcc_, perm=[0, 2, 1])
    # calc total batch count
    num_batch = len(label) // batch_size
    len_ = tf.reshape(len_,[-1])
    assert label_.get_shape()[0] == batch_size
    #convert to sparse tensor
    

    # print info
    stf.sg_info('%s set loaded.(total data=%d, total batch=%d)'
               % (set_name.upper(), len(label), num_batch))
    return mfcc_, label_, len_

def build_input3(batch_size=16, set_name='train', mode='train', num_mfcc = 40):
    # load meta file
    filename = '/home/datadisk/zswang/speech-corpus/preprocess/Libri_%s' % set_name
    data_files = tf.gfile.Glob(filename)
    filename_queue = tf.train.string_input_producer(data_files, num_epochs = None, shuffle = False)
    
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    

    
    sequence_features = {
        'mfccs': tf.FixedLenFeature([], dtype=tf.string),
        'seq_lens': tf.FixedLenFeature([], dtype=tf.string),
        'labels': tf.FixedLenFeature([], dtype=tf.string),
        'label_lens': tf.FixedLenFeature([], dtype=tf.string)
    }
    
    # Parse the example
    sequence_parsed = tf.parse_single_example(
        serialized_example,
        features=sequence_features
    )
    
    mfccs = tf.reshape(tf.decode_raw(sequence_parsed['mfccs'], tf.float32),[-1, num_mfcc, 3])
    #mfccs = _augment_speech(mfccs) #data perturbation
    
    labels = tf.decode_raw(sequence_parsed['labels'], tf.int32)
    seq_lens = tf.decode_raw(sequence_parsed['seq_lens'], tf.int32)
    label_lens = tf.decode_raw(sequence_parsed['label_lens'], tf.int32)
    # create batch queue with dynamic pad
    label_, mfcc_, seq_len_, label_len_ = tf.train.batch([labels, mfccs, seq_lens, label_lens], batch_size,
                                 shapes=[(None,), (None, num_mfcc, 3), (1,), (1,)],
                                 num_threads=64, capacity=batch_size*64,
                                 dynamic_pad=True, allow_smaller_final_batch=False)
    
    #label_ = tf.deserialize_many_sparse(label_, tf.int32)
    # batch * time * dim
    #label_ = tf.reshape(tf.expand_dims(label_,axis=-1), [batch_size, 20, -1])
    
    #mfcc_ =  tf.reshape(mfcc_, [batch_size, -1, 21])
    #mfcc_ = _augment_speech(mfcc_)
    
    seq_len_ = tf.reshape(seq_len_,[-1])
    label_len_ = tf.reshape(label_len_,[-1])
    label_ = tf.reshape(label_,[batch_size,-1])
    #label_ = tf.cast(label_, tf.int32)
    
    assert label_.get_shape()[0] == batch_size
    #convert to sparse tensor
    return mfcc_, label_, seq_len_,label_len_
#generate_csv('train')
#generate_csv('valid')
#generate_csv('test')
def make_tfrecord_librispeech(set_name='train', is_full = False, num_ex = 2000):
    
    def make_example(mfcc_files, s_lens, labels, l_lens):
        # The object we return
        #ex = tf.train.Example()
        
        def _int64_feature(value):
            return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
        def _float_feature(value):
            return tf.train.Feature(int64_list=tf.train.FloatList(value=[value]))
        def _bytes_feature(value):
            return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
        
        ex = tf.train.Example(features=tf.train.Features(feature={
        'mfccs': _bytes_feature(mfcc_files),
        'seq_lens': _bytes_feature(s_lens),
        'labels': _bytes_feature(labels),
        'label_lens': _bytes_feature(l_lens)}))
        """
        #print len(labels)
        fl_tokens = ex.feature_lists.feature_list["mfccs"]
        fl_labels = ex.feature_lists.feature_list["labels"]
        fl_lens = ex.feature_lists.feature_list["lens"]
        for token, label, len1 in zip(mfcc_files, labels, lens):
            print token
            fl_tokens.feature.add().float_list.value.append(token)
            fl_labels.feature.add().int64_list.value.append(label)
            fl_lens.feature.add().int64_list.value.append(len1)
        """
        return ex
    
    # load meta file
    labels, mfcc_files, seq_len, label_len = [], [], [], []
    with open(_data_path + 'preprocess/meta/%s.csv' % set_name) as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        for i, row in enumerate(reader):
            if is_full:
                # mfcc file
                mfcc = np.load(_data_path + 'preprocess/mfcc/' + row[0] + '.npy', allow_pickle=False)
                seq_len.append(np.asarray([np.shape(mfcc)[0]], dtype=np.int32).tostring())
                
                mfcc_reshape = np.asarray(np.reshape(mfcc,[-1]),dtype=np.float32)
                mfcc_files.append(mfcc_reshape.tostring())
                # label info ( convert to string object for variable-length support )
                #print 'shape is:', np.shape(label_temp), np.shape(indices)
                str2 = np.asarray(row[1:], dtype=np.int32)
                label_len.append(np.asarray([np.shape(str2)[0]], dtype=np.int32).tostring())
                labels.append(str2.tostring())
                
            else:
                if i < num_ex :
                    # mfcc file
                    mfcc = np.load(_data_path + 'preprocess/mfcc/' + row[0] + '.npy', allow_pickle=False)
                    seq_len.append(np.asarray([np.shape(mfcc)[0]], dtype=np.int32).tostring())
                    
                    mfcc_reshape = np.asarray(np.reshape(mfcc,[-1]),dtype=np.float32)
                    mfcc_files.append(mfcc_reshape.tostring())
                    # label info ( convert to string object for variable-length support )
                    #print 'shape is:', np.shape(label_temp), np.shape(indices)
                    str2 = np.asarray(row[1:], dtype=np.int32)
                    label_len.append(np.asarray([np.shape(str2)[0]], dtype=np.int32).tostring())
                    labels.append(str2.tostring())                 
    print 'preprocess step complete, begin generating process with length:', len(label_len)
    #print np.shape(mfcc_files[1000]), np.shape(mfcc_files[200])
    # Write all examples into a TFRecords file
    to_f = 2620
    if set_name == 'train' or set_name == 'train_sort':
        to_f = 28539
    b_n = 2880
          
    """
    with tempfile.NamedTemporaryFile(dir=_data_path + 'preprocess/',delete=False) as fp:
        writer = tf.python_io.TFRecordWriter(fp.name)
        print fp.name
        for mfcc_file, s_len, label, l_len in zip(mfcc_files, seq_len, labels, label_len):
            #print np.shape(mfcc_file)
            ex = make_example(mfcc_file, s_len, label, l_len)
            writer.write(ex.SerializeToString())
        writer.close()
    """
    j = 0
    path_s = _data_path + 'preprocess/'
    if set_name == 'train' or set_name == 'train_sort':
        for i, (mfcc_file, s_len, label, l_len) in enumerate(zip(mfcc_files, seq_len, labels, label_len)):
            ex = make_example(mfcc_file, s_len, label, l_len)
            if i == 0:
                fp = tempfile.NamedTemporaryFile(dir=_data_path + 'preprocess/',delete=False)
                print fp.name
                writer = tf.python_io.TFRecordWriter(fp.name)
            elif i%b_n == 0:
                writer.close()
                os.system('mv %s %sLibri_train_delta_%s-100_Seri' % (fp.name, path_s, j))
                j = j + 1
                fp = tempfile.NamedTemporaryFile(dir=_data_path + 'preprocess/',delete=False)
                print fp.name
                writer = tf.python_io.TFRecordWriter(fp.name)            
            writer.write(ex.SerializeToString())
        writer.close()
        os.system('mv %s %sLibri_train_delta_%s-100_Seri' % (fp.name, path_s, j))
    else:
        with tempfile.NamedTemporaryFile(dir=_data_path + 'preprocess/',delete=False) as fp:
            writer = tf.python_io.TFRecordWriter(fp.name)
            print fp.name
            for mfcc_file, s_len, label, l_len in zip(mfcc_files, seq_len, labels, label_len):
                #print np.shape(mfcc_file)
                ex = make_example(mfcc_file, s_len, label, l_len)
                writer.write(ex.SerializeToString())
            os.system('mv %s %sLibri_test_delta-100_Seri' % (fp.name, path_s))
            writer.close()        

        
def libri_length_test(set_name='train',rat=1.0):
    # load meta file
    i = 0
    total = 0
    with open(_data_path + 'preprocess/meta/%s.csv' % set_name) as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        for row in reader:
            total = total + 1
            
            mfcc = np.load(_data_path + 'preprocess/mfcc/' + row[0] + '.npy', allow_pickle=False)
            len_mfcc = np.shape(mfcc)[0]
            labels = np.asarray(row[1:], dtype=np.int32)
            len_label = np.shape(labels)[0]
            #i += 1.0*len_mfcc/len_label
            if len_mfcc >= rat*len_label:
                i = i + 1
                  
    print 'process step complete, with valid percentage: ' ,1.0*i/total, total

#libri_length_test()

#make_tfrecord_librispeech('test',True)
#make_tfrecord_librispeech('train',True)