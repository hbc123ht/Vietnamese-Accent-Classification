import os
from p_tqdm import p_map
import numpy as np
import tensorflow as tf
import multiprocessing
from dynaconf import settings
settings.load_file(path="config.py")

from utils import (get_wav, to_mfcc, load_categories,
                    normalize_mfcc, make_segments, 
                    segment_one, load_data, get_input_shape)
                            
from model import Model

import logging
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.ERROR)

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

if __name__ == '__main__':

    #labels
    categories = load_categories(os.path.join(settings.DATA_DIR, 'categories/labels.json'))
    
    #load data
    X, y = load_data(os.path.join(settings.DATA_DIR,'data'), categories = categories)


    # Get resampled wav files using multiprocessing
    logging.info('Loading wav files....')

    #load DATA
    X = p_map(get_wav, X)

    # # Convert to MFCC
    logging.info('Converting to MFCC....')
    X = p_map(to_mfcc, X)
    
    # # Data normalization
    logging.info('Nomalizing data....')
    X = p_map(normalize_mfcc,X)

    # # get the input shape 
    input_shape = get_input_shape(X, settings.COL_SIZE)
    print(input_shape)
    # # initiate model
    model = Model(input_shape = input_shape, num_classes = len(categories), lr = settings.LR)

    #load the weights                
    try:
        model.load_weights(settings.LOAD_CHECKPOINT_DIR)
    except:
        logging.error('Unable to load checkpoints')

    for X_,y_ in zip(X, y):
        X_, y_ = segment_one(X_ ,y_, COL_SIZE = settings.COL_SIZE)
        prediction = model.predict(X_)
        prediction = np.argmax(prediction, axis = 1)
        prediction = np.bincount(prediction)
        prediction = np.argmax(prediction)
        prediction = [key for key, value in categories.items() if value == prediction]
        print(prediction)
    