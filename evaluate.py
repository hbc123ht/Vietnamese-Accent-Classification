import os
from p_tqdm import p_map

import tensorflow as tf
import multiprocessing
from dynaconf import settings
settings.load_file(path="config.py")

from utils import (get_wav, to_mfcc, load_categories,
                    normalize_mfcc, make_segments, 
                    segment_one, load_data, get_input_shape)
                            
from model import ResNet18

import logging
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.ERROR)

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

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
    
    # # initiate model
    model = ResNet18(len(categories))
    model.build(input_shape = input_shape)
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adadelta(lr=settings.LR),
              metrics=['accuracy'])

    #load the weights                
    try:
        model.load_weights(settings.LOAD_CHECKPOINT_DIR)
    except:
        logging.error('Unable to load checkpoints')

    #Evaluates
    acc = 0.0

    for X_,y_ in zip(X, y):
        X_, y_ = segment_one(X_ ,y_, COL_SIZE = settings.COL_SIZE, OVERLAP_SIZE = settings.OVERLAP_SIZE)
        try:
            score = model.evaluate(X_, y_)
            acc += (score[1] >= 0.17)
        except:
            pass
        

    print('Accuracy : ',acc / len(y))
    