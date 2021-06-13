import os
from p_tqdm import p_map
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
import numpy as np
from sklearn.model_selection import train_test_split
from dynaconf import settings
settings.load_file(path="config.py")

from utils import (get_input_shape, get_wav, load_categories, to_mfcc, 
                    normalize_mfcc, make_segments, 
                    load_data, get_input_shape)

from model import Model

import logging
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

if __name__ == '__main__':

    #labels
    categories = load_categories(os.path.join(settings.DATA_DIR, 'categories/labels.json'))
    
    #load data
    X, y = load_data(os.path.join(settings.DATA_DIR,'data'), categories = categories)

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
    input_shape = get_input_shape(X_train, settings.COL_SIZE)

    # # Create segments from MFCCs
    X_train, y_train = make_segments(X_train, y_train, COL_SIZE = settings.COL_SIZE)
    X_validation, y_validation = make_segments(X_test, y_test, COL_SIZE = settings.COL_SIZE)
    
    # # Train model
    model = Model(input_shape, num_classes = len(categories), lr = settings.LR)

    # Creates log file for graphical interpretation using TensorBoard
    tb = TensorBoard(log_dir=settings.LOG, histogram_freq=0, batch_size=32, write_graph=True, write_grads=True,
                     write_images=True, embeddings_freq=0, embeddings_layer_names=None,
                     embeddings_metadata=None)

    
    # modelCheckpoint
    cp = ModelCheckpoint(os.path.join(settings.CHECKPOINT_DIR, 'model.{epoch:02d}.h5'),
                                              monitor='val_loss',
                                              verbose=1,
                                              mode = 'max',
                                              save_freq = settings.SAVE_CHECKPOINT_FREQUENCY * settings.STEPS_PER_EPOCH)

    #load the weights                
    if (settings.LOAD_CHECKPOINT_DIR != None): 
        model.load_weights(settings.LOAD_CHECKPOINT_DIR)

 
    # fit the model
    logging.info('Start training....')
    model.fit(x = np.array(X_train), y = np.array(y_train),
                batch_size = settings.BATCH_SIZE,
                steps_per_epoch= len(X_train) / settings.BATCH_SIZE,
                epochs=settings.NUM_EPOCH,
                callbacks=[tb, cp], 
                validation_data=(np.array(X_validation), np.array(y_validation)))
