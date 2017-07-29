from data import make_tfrecord_librispeech, libri_length_test
from preprocess_data import process_norm, process_sort


#process_norm()
#process_sort()
make_tfrecord_librispeech('train_sort',True)
#make_tfrecord_librispeech('test',True)

#libri_length_test('test',8.0)