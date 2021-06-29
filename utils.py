import glob
import os
import librosa
import random
import numpy as np
import tensorflow as tf
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

def load_categories(path):
    '''
    Get the categories from json file
    :param path (str): path to json file
    :return (list): List of categories
    ''' 
    import json
    with open(path) as json_data_file:
        data = json.load(json_data_file)

    return data['categories']

def load_data(path, categories):
    '''
    get the labels from data file names
    :param path (str): the path of data folder
    :categories (list): all the categories of labels
    :return (list), (list): the path of all data files, the labels
    '''
    x = []
    y = []
    for file in os.listdir(path):
        for key in categories.keys():
            if file.startswith(key):
                y.append(categories[key])
                x.append(os.path.join(path, file))
                
    return x, y

def to_categorical(y, num_classes = 6):
    '''
    Converts list of languages into a binary class matrix
    :param y (list): list of languages
    :return (numpy array): binary class matrix
    '''
    # integer encode
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(y)
    integer_encoded.reshape(integer_encoded.shape[0])
    a = tf.keras.utils.to_categorical(integer_encoded, num_classes=num_classes)

    return a

def get_wav(language_num, RATE = 36000):
    '''
    Load wav file from disk and down-samples to RATE
    :param language_num (list): list of file names
    :return (numpy array): Down-sampled wav file
    '''

    y, sr = librosa.load(language_num)
    return (librosa.core.resample(y=y,orig_sr=sr,target_sr=RATE, scale=True))

def to_mfcc(wav, RATE = 36000, N_MFCC = 300):
    '''
    Converts wav file to Mel Frequency Ceptral Coefficients
    :param wav (numpy array): Wav form
    :return (2d numpy array: MFCC
    '''
    return(librosa.feature.mfcc(y=wav, sr=RATE, n_mfcc=N_MFCC))
  
def to_mel(wav, RATE = 36000, N_MELS = 200):
    '''
    Converts wav file to Mel Frequency Ceptral Coefficients
    :param wav (numpy array): Wav form
    :return (2d numpy array: MFCC
    '''
    return(librosa.feature.melspectrogram(y=wav, sr=RATE, n_mels=N_MELS))
  
def remove_silence(wav, thresh=0.04, chunk=5000):
    '''
    Searches wav form for segments of silence. If wav form values are lower than 'thresh' for 'chunk' samples, the values will be removed
    :param wav (np array): Wav array to be filtered
    :return (np array): Wav array with silence removed
    '''

    tf_list = []
    for x in range(int(len(wav) / chunk)):
        if (np.any(wav[chunk * x:chunk * (x + 1)] >= thresh) or np.any(wav[chunk * x:chunk * (x + 1)] <= -thresh)):
            tf_list.extend([True] * chunk)
        else:
            tf_list.extend([False] * chunk)

    tf_list.extend((len(wav) - len(tf_list)) * [False])
    return(wav[tf_list])

def normalize_mfcc(mfcc):
    '''
    Normalize mfcc
    :param mfcc:
    :return:
    '''
    mms = MinMaxScaler()
    return(mms.fit_transform(np.abs(mfcc)))

def get_input_shape(mfccs, COL_SIZE):
    """
    Get the input shape of data
    :param mfccs: list of mfccs
    :param COL_SIZE: args.COL_SIZE
    :return (tuple): input shape
    """
    rows = np.array(mfccs[0]).shape[0]
    columns = COL_SIZE

    return (None, rows, columns, 1)

def make_segments(mfccs,labels, COL_SIZE = 45, OVERLAP_SIZE = 15):
    '''
    Makes segments of mfccs and attaches them to the labels
    :param mfccs: list of mfccs
    :param labels: list of labels
    :return (tuple): Segments with labels
    '''
    segments = []
    seg_labels = []
    for mfcc,label in zip(mfccs,labels):
        for surplus in range(0, COL_SIZE, OVERLAP_SIZE):
            for start in range(0, int(mfcc.shape[1] / COL_SIZE) - 1):
                segments.append(mfcc[:, start * COL_SIZE + surplus : (start + 1) * COL_SIZE + surplus])
                seg_labels.append(label)

            
        if (int(mfcc.shape[1]) < COL_SIZE):
            begin_duration = random.randint(0, COL_SIZE - mfcc.shape[1])
            end_duration = COL_SIZE - mfcc.shape[1] - begin_duration
            mfcc_ = np.concatenate((np.zeros((mfcc.shape[0], begin_duration)), mfcc), axis = 1)
            mfcc_ = np.concatenate((mfcc_,np.zeros((mfcc.shape[0], end_duration))), axis = 1)
            segments.append(mfcc_)
            seg_labels.append(label)

    return(np.array(segments)[..., np.newaxis], seg_labels)

def segment_one(mfcc, label, COL_SIZE = 45, OVERLAP_SIZE = 15):
    '''
    Creates segments from on mfcc image. If last segments is not long enough to be length of columns divided by COL_SIZE
    :param mfcc (numpy array): MFCC array
    :return (numpy array): Segmented MFCC array
    '''
    segments = []
    seg_labels = []
    
    for surplus in range(0, COL_SIZE, OVERLAP_SIZE):
        for start in range(0, int(mfcc.shape[1] / COL_SIZE) - 1):
            segments.append(mfcc[:, start * COL_SIZE + surplus : (start + 1) * COL_SIZE + surplus])
            seg_labels.append(label)

    if (int(mfcc.shape[1]) < COL_SIZE):
        begin_duration = random.randint(0, COL_SIZE - mfcc.shape[1])
        end_duration = COL_SIZE - mfcc.shape[1] - begin_duration
        mfcc_ = np.concatenate((np.zeros((mfcc.shape[0], begin_duration)), mfcc), axis = 1)
        mfcc_ = np.concatenate((mfcc_,np.zeros((mfcc.shape[0], end_duration))), axis = 1)
        segments.append(mfcc_)
        seg_labels.append(label)

    return(np.array(segments)[..., np.newaxis], np.array(seg_labels))

def create_segmented_mfccs(X_train):
    '''
    Creates segmented MFCCs from X_train
    :param X_train: list of MFCCs
    :return: segmented mfccs
    '''
    segmented_mfccs = []
    for mfcc in X_train:
        segmented_mfccs.append(segment_one(mfcc))
    return(segmented_mfccs)
    
def log_specgram(audio, sample_rate, window_size=20, step_size=10, eps=1e-10):
    nperseg = int(round(window_size * sample_rate / 1e3))
    noverlap = int(round(step_size * sample_rate / 1e3))
    freqs, _, spec = signal.spectrogram(audio,
                                        fs=sample_rate,
                                        window='hann', # 'text'
                                        nperseg=nperseg,
                                        noverlap = noverlap,
                                        detrend=False)
    return np.log(spec.T.astype(np.float32) + eps)