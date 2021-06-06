import os
from p_tqdm import p_map
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
import numpy as np
from sklearn.model_selection import train_test_split
import argparse

from utils import (get_input_shape, get_wav, load_categories, to_mfcc, 
                    normalize_mfcc, make_segments, 
                    load_data, get_input_shape)

from model import Model

import logging
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

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
    categories = load_categories(args.CATEGORIES_DIR)
    
    #load data
    X, y = load_data(args.DATA_DIR, categories = categories)

    #split and shuffle data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.08, random_state=42)

    # Get resampled wav files using multiprocessing
    logging.info('Loading wav files....')

    X_train = p_map(get_wav, X_train)
    X_test = p_map(get_wav, X_test)

    # # Convert to MFCC
    logging.info('Converting to MFCC....')
    X_train = p_map(to_mfcc, X_train)
    X_test = p_map(to_mfcc, X_test)

    # # Data normalization
    logging.info('Nomalizing data....')
    X_train = p_map(normalize_mfcc,X_train)
    X_test = p_map(normalize_mfcc,X_test)
           
    # Get input shape
    input_shape = get_input_shape(X_train, args.COL_SIZE)

    # # Create segments from MFCCs
    X_train, y_train = make_segments(X_train, y_train, COL_SIZE = args.COL_SIZE)
    X_validation, y_validation = make_segments(X_test, y_test, COL_SIZE = args.COL_SIZE)
    
    # # Train model
    model = Model(input_shape, num_classes = len(categories), lr = args.LR)

    # Creates log file for graphical interpretation using TensorBoard
    tb = TensorBoard(log_dir=args.LOG, histogram_freq=0, batch_size=32, write_graph=True, write_grads=True,
                     write_images=True, embeddings_freq=0, embeddings_layer_names=None,
                     embeddings_metadata=None)

    
    # modelCheckpoint
    cp = ModelCheckpoint(os.path.join(args.CHECKPOINT_DIR, 'model.{epoch:02d}.h5'),
                                              monitor='val_loss',
                                              verbose=1,
                                              mode = 'max',
                                              save_freq = args.SAVE_CHECKPOINT_FREQUENCY * args.STEPS_PER_EPOCH)

    #load the weights                
    if (args.LOAD_CHECKPOINT_DIR != None): 
        model.load_weights(args.LOAD_CHECKPOINT_DIR)

 
    # fit the model
    logging.info('Start training....')
    model.fit(x = np.array(X_train), y = np.array(y_train),
                batch_size = args.BATCH_SIZE,
                steps_per_epoch= len(X_train) / args.BATCH_SIZE,
                epochs=args.NUM_EPOCH,
                callbacks=[tb, cp], 
                validation_data=(np.array(X_validation), np.array(y_validation)))
