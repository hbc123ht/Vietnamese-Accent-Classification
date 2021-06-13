import os
from p_tqdm import p_map
import numpy as np
import tensorflow as tf
import multiprocessing
import argparse
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

def parser():
    parser = argparse.ArgumentParser(description='Configs for training')
    parser.add_argument("--DEBUG", default = True, type = bool, help = 'Debug mode')
    parser.add_argument("--SILENCE_THRESHOLD", default = .01, type = float, help = 'Debug mode')
    parser.add_argument('--COL_SIZE', default = 30, type = int, help = 'Range of a single segment')
    parser.add_argument('--SAVE_CHECKPOINT_FREQUENCY', default = 5, type = int, help = 'Save checkpoint per number of epochs')
    parser.add_argument('--NUM_EPOCH', default = 100,type = int, help = 'Number of epochs')
    parser.add_argument('--DATA_DIR', type = str, help = 'Dir of data')
    parser.add_argument('--CHECKPOINT_DIR', default = 'checkpoint', type = str, help ='Dir of checkpoint')
    parser.add_argument('--LOG', default='logs', type = str, help = 'Dir of logs')
    parser.add_argument('--BATCH_SIZE', default=32, type = int)
    parser.add_argument('--STEPS_PER_EPOCH', default=128, type = int)
    parser.add_argument('--LOAD_CHECKPOINT_DIR', default=None, type = str)
    parser.add_argument('--CATEGORIES_DIR', default=None, type = str)
    parser.add_argument('--LR', default=0.01, type = float)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    
    # init arparser
    args = parser()

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
        prediction = np.argmax(np.sum(prediction,axis = 0))
        prediction = [key for key, value in categories.items() if value == prediction]
        print(prediction)
    