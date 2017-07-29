from python_speech_features import mfcc, delta, fbank, logfbank
import scipy.io.wavfile as wav
import numpy as np
import librosa
file_name = '6147-34607-0011.flac' #'61-70970-0025.flac'
(wave, sr) = librosa.load(file_name)#, mono=True, sr=None
mfcc_feat = mfcc(wave, sr )
mfcc_total=[]
mfccs = librosa.feature.mfcc(wave, sr=sr,n_mfcc=13)
mfcc_delta = librosa.feature.delta(mfccs)
mfcc_delta2 = librosa.feature.delta(mfccs, order=2)
mfcc_total.append(mfccs)
mfcc_total.append(mfcc_delta)
mfcc_total.append(mfcc_delta2)

mfccs = np.asarray(mfcc_total)
#mfccs = librosa.feature.melspectrogram(wave, sr=sr,n_mels=128)
print np.shape(mfccs), np.shape(mfcc_delta), np.shape(mfcc_delta2)
#print mfccs
"""
mean = np.mean(mfcc_feat)
std  = np.std(mfcc_feat)
mfcc_feat = (mfcc_feat-mean)/std
"""

#print mfcc_feat[:,0]
#print mfcc_feat[:,12]
#print np.shape(wave)
#print np.shape(mfcc_feat)


d_mfcc_feat = delta(mfcc_feat, 2)
fbank_feat = logfbank(wave, sr)

#print np.shape(d_mfcc_feat)
#print np.shape(fbank_feat)
#print mfcc_feat
