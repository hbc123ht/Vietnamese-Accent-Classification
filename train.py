import os
import pandas as pd
from collections import Counter
import sys
import tqdm
from p_tqdm import p_map
sys.path.append('../speech-accent-recognition/src>')

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
import accuracy
import multiprocessing
import numpy as np
from sklearn.model_selection import train_test_split
import argparse

from preprocessing import (to_categorical, get_wav, to_mfcc, 
                            remove_silence, normalize_mfcc, make_segments, 
                            segment_one,create_segmented_mfccs, load_data)
from model import Model


def parser():
    parser = argparse.ArgumentParser(description='Configs for training')
    parser.add_argument("--DEBUG", default = True, type = bool, help = 'Debug mode')
    parser.add_argument("--SILENCE_THRESHOLD", default = .01, type = float, help = 'Debug mode')
    parser.add_argument('--COL_SIZE = 30', default = 30, type = int, help = 'Range of a single segment')
    parser.add_argument('--SAVE_CHECKPOINT_FREQUENCY', default = 5, type = int, help = 'Save checkpoint per number of epochs')
    parser.add_argument('--NUM_EPOCH', default = 100,type = int, help = 'Number of epochs')
    parser.add_argument('--DATA_DIR', type = str, help = 'Dir of data')
    parser.add_argument('--CHECKPOINT_DIR', default = 'checkpoint', type = str, help ='Dir of checkpoint')
    parser.add_argument('--LOG', default='logs', type = str, help = 'Dir of logs')
    parser.add_argument('--BATCH_SIZE', default=128, type = int)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    
    # init arparser
    args = parser()

    #labels
    categories = ['female_central',
                  'male_central',
                  'female_north',
                  'male_north',
                  'female_south',
                  'male_south']

    #load data
    X, y = load_data(args.DATA_DIR, categories)
    
    # To categorical
    y = to_categorical(y)

    #split and shuffle data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    print(np.array(X_train).shape[0], np.array(y_train).shape[0])
    # y_test = to_categorical(y_test)

    # Get resampled wav files using multiprocessing
    if args.DEBUG:
        print('Loading wav files....')
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())


    X_train = p_map(get_wav, X_train)
    X_test = p_map(get_wav, X_test)
    # # Convert to MFCC
    if args.DEBUG:
        print('Converting to MFCC....')
    X_train = p_map(to_mfcc, X_train)
    X_test = p_map(to_mfcc, X_test)

    # # Create segments from MFCCs
    X_train, y_train = make_segments(X_train, y_train)
    X_validation, y_validation = make_segments(X_test, y_test)

    # Get input shape
    input_shape = (X_train[0].shape[0], X_train[0].shape[1], 1)

    X_train = np.asarray(X_train)
    X_validation = np.asarray(X_validation)

    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
    X_validation = X_validation.reshape(X_validation.shape[0], X_validation.shape[1], X_validation.shape[2], 1)
    
    # # Train model
    model = Model(input_shape, num_classes = len(categories))
    # Stops training if accuracy does not change at least 0.005 over 10 epochs
    es = EarlyStopping(monitor='acc', min_delta=.005, patience=10, verbose=1, mode='auto')

    # Creates log file for graphical interpretation using TensorBoard
    tb = TensorBoard(log_dir=args.LOG, histogram_freq=0, batch_size=32, write_graph=True, write_grads=True,
                     write_images=True, embeddings_freq=0, embeddings_layer_names=None,
                     embeddings_metadata=None)
    
    # modelCheckpoint
    cp = ModelCheckpoint(os.path.join(args.CHECKPOINT_DIR, 'model.{epoch:02d}.h5'),
                                              monitor='val_loss',
                                              verbose=1,
                                              save_best_only=True,
                                              save_freq = args.SAVE_CHECKPOINT_FREQUENCY)

    # Image shifting
    datagen = ImageDataGenerator(width_shift_range=0.05)

    # # Compile
    # model.compile(batch_size=args.BATCH_SIZE,
    #             steps_per_epoch=len(X_train) / 32, 
    #             epochs=args.NUM_EPOCH,
    #             callbacks=[es,tb, cp], 
    #             validation_data=(X_validation,y_validation))
    # Fit model using ImageDataGenerator
    model.fit_generator(datagen.flow(np.array(X_train), tf.stack(y_train),batch_size=args.BATCH_SIZE),
                steps_per_epoch=len(X_train) / 32, 
                epochs=args.NUM_EPOCH,
                callbacks=[tb, cp], 
                validation_data=(np.array(X_validation), np.array(y_validation)))

    # # Make predictions on full X_test MFCCs
    # y_predicted = accuracy.predict_class_all(create_segmented_mfccs(X_test), model)

    # # Print statistics
    # print('Training samples:', train_count)
    # print('Testing samples:', test_count)
    # print('Accuracy to beat:', acc_to_beat)
    # print('Confusion matrix of total samples:\n', np.sum(accuracy.confusion_matrix(y_predicted, y_test),axis=1))
    # print('Confusion matrix:\n',accuracy.confusion_matrix(y_predicted, y_test))
    # print('Accuracy:', accuracy.get_accuracy(y_predicted,y_test))

    # # Save model
    # save_model(model, model_filename)
