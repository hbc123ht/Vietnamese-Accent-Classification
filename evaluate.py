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
from model import Model, Model2

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
    parser.add_argument('--LR', default=0.01, type = float)
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
    X, y = load_data(args.DATA_DIR, categories = categories)
    # To categorical
    y = to_categorical(y, num_classes = len(categories))


    # Get resampled wav files using multiprocessing
    if args.DEBUG:
        print('Loading wav files....')
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())


    X = p_map(get_wav, X)
    # # Convert to MFCC
    if args.DEBUG:
        print('Converting to MFCC....')
    X = p_map(to_mfcc, X)

    # # Data normalization
    X = p_map(normalize_mfcc,X)

   
    
    # # Create segments from MFCCs
    X, y = make_segments(X, y, COL_SIZE = args.COL_SIZE)

    # Get input shape
    input_shape = (X[0].shape[0], X[0].shape[1], 1)

     # # Convert to numpy
    X = np.asarray(X)
    

    X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
    
    # # Train model
    model = Model2(input_shape, num_classes = len(categories), lr = args.LR)

    #load the weights                
    if (args.LOAD_CHECKPOINT_DIR != None): 
        model.load_weights(args.LOAD_CHECKPOINT_DIR)

    score = model.evaluate(X, y, batch_size=128)
    print(score[1])
    # # Compile
    # model.compile(batch_size=args.BATCH_SIZE,
    #             steps_per_epoch=len(X_train) / 32, 
    #             epochs=args.NUM_EPOCH,
    #             callbacks=[es,tb, cp], 
    #             validation_data=(X_validation,y_validation))
    # Fit model using ImageDataGenerator

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
