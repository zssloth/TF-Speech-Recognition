import numpy as np
import pandas as pd
import glob
import csv
import librosa
import scikits.audiolab
import data
import os
import subprocess
from operator import itemgetter
from python_speech_features import mfcc

__author__ = 'namju.kim@kakaobrain.com'


# data path
_root_path = '/home/datadisk/zswang/speech-corpus/' 
_data_path = _root_path


#
# process VCTK corpus
#

def process_vctk(csv_file):

    # create csv writer
    writer = csv.writer(csv_file, delimiter=',')

    # read label-info
    df = pd.read_table(_data_path + 'VCTK-Corpus/speaker-info.txt', usecols=['ID'],
                       index_col=False, delim_whitespace=True)

    # read file IDs
    file_ids = []
    for d in [_data_path + 'VCTK-Corpus/txt/p%d/' % uid for uid in df.ID.values]:
        file_ids.extend([f[-12:-4] for f in sorted(glob.glob(d + '*.txt'))])

    for i, f in enumerate(file_ids):

        # wave file name
        wave_file = _data_path + 'VCTK-Corpus/wav48/%s/' % f[:4] + f + '.wav'
        fn = wave_file.split('/')[-1]
        target_filename = 'asset/data/preprocess/mfcc/' + fn + '.npy'
        if os.path.exists( target_filename ):
            continue
        # print info
        print("VCTK corpus preprocessing (%d / %d) - '%s']" % (i, len(file_ids), wave_file))

        # load wave file
        wave, sr = librosa.load(wave_file, mono=True, sr=None)

        # re-sample ( 48K -> 16K )
        wave = wave[::3]

        # get mfcc feature
        mfcc = librosa.feature.mfcc(wave, sr=16000)

        # get label index
        label = data.str2index(open(_data_path + 'VCTK-Corpus/txt/%s/' % f[:4] + f + '.txt').read())

        # save result ( exclude small mfcc data to prevent ctc loss )
        if len(label) < mfcc.shape[1]:
            # save meta info
            writer.writerow([fn] + label)
            # save mfcc
            np.save(target_filename, mfcc, allow_pickle=False)


#
# process LibriSpeech corpus
#
def process_libri(csv_file, category):

    parent_path = _data_path + category + '/'
    labels, wave_files = [], []

    # create csv writer
    writer = csv.writer(csv_file, delimiter=',')

    # read directory list by speaker
    speaker_list = glob.glob(parent_path + '*')
    for spk in speaker_list:

        # read directory list by chapter
        chapter_list = glob.glob(spk + '/*/')
        for chap in chapter_list:

            # read label text file list
            txt_list = glob.glob(chap + '/*.txt')
            for txt in txt_list:
                with open(txt, 'rt') as f:
                    records = f.readlines()
                    for record in records:
                        # parsing record
                        field = record.split('-')  # split by '-'
                        speaker = field[0]
                        chapter = field[1]
                        field = field[2].split()  # split field[2] by ' '
                        utterance = field[0]  # first column is utterance id

                        # wave file name
                        wave_file = parent_path + '%s/%s/%s-%s-%s.flac' % \
                                                  (speaker, chapter, speaker, chapter, utterance)
                        wave_files.append(wave_file)

                        # label index
                        labels.append(data.str2index(' '.join(field[1:])))  # last column is text label

    # save results
    for i, (wave_file, label) in enumerate(zip(wave_files, labels)):
        fn = wave_file.split('/')[-1] #extract file name
        target_filename = _root_path + 'preprocess/mfcc/' + fn + '.npy'
        

        if os.path.exists( target_filename ):
            #print 'continue.'
            continue

        
        # print info
        print("LibriSpeech corpus preprocessing (%d / %d) - '%s']" % (i, len(wave_files), wave_file))

        # load flac file
        wave, sr, _ = scikits.audiolab.flacread(wave_file)

        # get mfcc feature, default 20 mfcc features, return np.ndarray [shape=(n_mfcc=20, t)], where t is the number of frames.
        #mfcc = librosa.feature.mfcc(wave, sr=16000)
        
        n_fft = 400 #16000*0.025 #25ms
        hop_length = 160 #16000*0.01
        """
        return np.ndarray [shape=(n_mfcc=20, t)], where t is the number of frames
        40ms per frame(window length) with 10ms stride
        t = sec_of_samples*sample_rate/hop_length
        """
        mfcc_total = [] 
        mfcc = librosa.feature.mfcc(wave, sr=16000, n_fft=n_fft, hop_length=hop_length, n_mfcc=13)
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        #mfcc = librosa.feature.melspectrogram(wave, sr=16000, n_fft=n_fft, hop_length=hop_length, n_mels=128)
        
        
        mfcc_total.append(mfcc)
        mfcc_total.append(mfcc_delta)
        mfcc_total.append(mfcc_delta2)
        
        mfcc = np.asarray(mfcc_total) #size: 3*13*fea_len
        
        mfcc_ = np.transpose(mfcc,axes=[2,1,0]) #size: fea_len*13*3
        #mfcc_ = mfcc_[:,1:]
        
        #do normalization
        """
        """
        mean = np.mean(mfcc_)
        std  = np.std(mfcc_)
        mfcc_ = (mfcc_- mean)/std


        # save result ( exclude small mfcc data to prevent ctc loss )
        if len(label) < mfcc_.shape[0]:
            """if len(label) > mfcc.shape[1], meaning that there has at least two characters in 10ms(hop_length) and we can not separate it."""
            # filename
            # save meta info
            writer.writerow([fn] + label)

            # save mfcc
            np.save(target_filename, mfcc_, allow_pickle=False)

def process_sorted_libri(csv_file, category):

    parent_path = _data_path + category + '/'
    labels, wave_files, len_list = [], [], []

    # create csv writer
    writer = csv.writer(csv_file, delimiter=',')

    # read directory list by speaker
    speaker_list = glob.glob(parent_path + '*')
    for spk in speaker_list:

        # read directory list by chapter
        chapter_list = glob.glob(spk + '/*/')
        for chap in chapter_list:

            # read label text file list
            txt_list = glob.glob(chap + '/*.txt')
            for txt in txt_list:
                with open(txt, 'rt') as f:
                    records = f.readlines()
                    for record in records:
                        # parsing record
                        field = record.split('-')  # split by '-'
                        speaker = field[0]
                        chapter = field[1]
                        field = field[2].split()  # split field[2] by ' '
                        utterance = field[0]  # first column is utterance id

                        # wave file name
                        wave_file = parent_path + '%s/%s/%s-%s-%s.flac' % \
                                                  (speaker, chapter, speaker, chapter, utterance)
                        wave_files.append(wave_file)

                        # label index
                        label_buf = data.str2index(' '.join(field[1:]))
                        labels.append(label_buf)  # last column is text label
                        len_list.append(len(label_buf))
    
    zip_list = zip(wave_files, labels, len_list)
    zip_list.sort(key=itemgetter(2))
    # save results
    for i, (wave_file, label, _) in enumerate(zip_list):
        fn = wave_file.split('/')[-1] #extract file name
        target_filename = _root_path + 'preprocess/mfcc/' + fn + '.npy'
        

        if os.path.exists( target_filename ):
            #print 'continue.'
            continue

        
        # print info
        print("LibriSpeech corpus preprocessing (%d / %d) - '%s']" % (i, len(wave_files), wave_file))

        # load flac file
        wave, sr, _ = scikits.audiolab.flacread(wave_file)

        # get mfcc feature, default 20 mfcc features, return np.ndarray [shape=(n_mfcc=20, t)], where t is the number of frames.
        #mfcc = librosa.feature.mfcc(wave, sr=16000)
        
        n_fft = 400 #16000*0.025 #25ms
        hop_length = 160 #16000*0.01
        """
        return np.ndarray [shape=(n_mfcc=20, t)], where t is the number of frames
        40ms per frame(window length) with 10ms stride
        t = sec_of_samples*sample_rate/hop_length
        """
        mfcc_total = [] 
        mfcc = librosa.feature.mfcc(wave, sr=16000, n_fft=n_fft, hop_length=hop_length, n_mfcc=13)
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        #mfcc = librosa.feature.melspectrogram(wave, sr=16000, n_fft=n_fft, hop_length=hop_length, n_mels=128)
        
        
        mfcc_total.append(mfcc)
        mfcc_total.append(mfcc_delta)
        mfcc_total.append(mfcc_delta2)
        
        mfcc = np.asarray(mfcc_total) #size: 3*13*fea_len
        
        mfcc_ = np.transpose(mfcc,axes=[2,1,0]) #size: fea_len*13*3
        #mfcc_ = mfcc_[:,1:]
        
        #do normalization
        """
        """
        mean = np.mean(mfcc_)
        std  = np.std(mfcc_)
        mfcc_ = (mfcc_- mean)/std


        # save result ( exclude small mfcc data to prevent ctc loss )
        if len(label) < mfcc_.shape[0]:
            """if len(label) > mfcc.shape[1], meaning that there has at least two characters in 10ms(hop_length) and we can not separate it."""
            # filename
            # save meta info
            writer.writerow([fn] + label)

            # save mfcc
            np.save(target_filename, mfcc_, allow_pickle=False)
#
# process TEDLIUM corpus
#
def convert_sph( sph, wav ):
    """Convert an sph file into wav format for further processing"""
    command = [
        'sox','-t','sph', sph, '-b','16','-t','wav', wav
    ]
    subprocess.check_call( command ) # Did you install sox (apt-get install sox)

def process_ted(csv_file, category):

    parent_path = _data_path + 'TEDLIUM_release2/' + category + '/'
    labels, wave_files, offsets, durs = [], [], [], []

    # create csv writer
    writer = csv.writer(csv_file, delimiter=',')

    # read STM file list
    stm_list = glob.glob(parent_path + 'stm/*')
    for stm in stm_list:
        with open(stm, 'rt') as f:
            records = f.readlines()
            for record in records:
                field = record.split()

                # wave file name
                wave_file = parent_path + 'sph/%s.sph.wav' % field[0]
                wave_files.append(wave_file)

                # label index
                labels.append(data.str2index(' '.join(field[6:])))

                # start, end info
                start, end = float(field[3]), float(field[4])
                offsets.append(start)
                durs.append(end - start)

    # save results
    for i, (wave_file, label, offset, dur) in enumerate(zip(wave_files, labels, offsets, durs)):
        fn = "%s-%.2f" % (wave_file.split('/')[-1], offset)
        target_filename = 'asset/data/preprocess/mfcc/' + fn + '.npy'
        if os.path.exists( target_filename ):
            continue
        # print info
        print("TEDLIUM corpus preprocessing (%d / %d) - '%s-%.2f]" % (i, len(wave_files), wave_file, offset))
        # load wave file
        if not os.path.exists( wave_file ):
            sph_file = wave_file.rsplit('.',1)[0]
            if os.path.exists( sph_file ):
                convert_sph( sph_file, wave_file )
            else:
                raise RuntimeError("Missing sph file from TedLium corpus at %s"%(sph_file))
        wave, sr = librosa.load(wave_file, mono=True, sr=None, offset=offset, duration=dur)

        # get mfcc feature
        mfcc = librosa.feature.mfcc(wave, sr=16000)

        # save result ( exclude small mfcc data to prevent ctc loss )
        if len(label) < mfcc.shape[1]:
            # filename

            # save meta info
            writer.writerow([fn] + label)

            # save mfcc
            np.save(target_filename, mfcc, allow_pickle=False)



#
# Create directories
#
if not os.path.exists(_root_path + 'preprocess'):
    os.makedirs('asset/data/preprocess')
if not os.path.exists(_root_path + 'preprocess/meta'):
    os.makedirs(_root_path + 'preprocess/meta')
if not os.path.exists(_root_path + 'preprocess/mfcc'):
    os.makedirs(_root_path + 'preprocess/mfcc')

#
# Run pre-processing for training
#
def process_norm():
    # LibriSpeech corpus for train

    csv_f = open(_root_path + 'preprocess/meta/train.csv', 'w')
    process_libri(csv_f, 'train-clean-100')
    csv_f.close()
    
    #
    # Run pre-processing for validation
    #
    """  
    # LibriSpeech corpus for valid
    csv_f = open(_root_path + 'preprocess/meta/valid.csv', 'w')
    process_libri(csv_f, 'dev-clean')
    csv_f.close()
    """    
    
    #
    # Run pre-processing for testing
    #
    
    # LibriSpeech corpus for test
    csv_f = open(_root_path + 'preprocess/meta/test.csv', 'w')
    process_libri(csv_f, 'test-clean')
    csv_f.close()
    
def process_sort():
    
    # LibriSpeech corpus for train
    csv_f = open(_root_path + 'preprocess/meta/train_sort.csv', 'w')
    process_sorted_libri(csv_f, 'train-clean-100')
    csv_f.close()
    
    #
    # Run pre-processing for validation
    #
    """
    # LibriSpeech corpus for valid
    csv_f = open(_root_path + 'preprocess/meta/valid_sort.csv', 'w')
    process_sorted_libri(csv_f, 'dev-clean')
    csv_f.close()
    
    
    #
    # Run pre-processing for testing
    #
    """
    # LibriSpeech corpus for test
    csv_f = open(_root_path + 'preprocess/meta/test.csv', 'w')
    process_sorted_libri(csv_f, 'test-clean')
    csv_f.close()
    
#process_norm()
