import os
from p_tqdm import p_map
import numpy as np
import tensorflow as tf
import multiprocessing
from dynaconf import settings
settings.load_file(path="config.py")

from utils import (get_wav, to_mfcc, load_categories, to_mel,
                    normalize_mfcc, make_segments, 
                    segment_one, load_data, get_input_shape, make_segment, add_dim)
                            
from model import Model

import logging
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.ERROR)

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

if __name__ == '__main__':

    #labels
    categories = load_categories(os.path.join(settings.DATA_DIR, 'categories/labels.json'))
    
    #load data
    X, y = load_data(os.path.join(settings.EVALUATE_DIR,'data'), categories = categories)


    model = tf.keras.models.load_model(settings.LOAD_MODEL_DIR)
    sum = 0.
    acc = 0.
    for X_, y_ in zip(X,y):
        print(y_)
        print(X_)
        X_ = get_wav(X_)
        X_ = make_segment(X_, COL_SIZE = settings.COL_SIZE, OVERLAP_SIZE = settings.OVERLAP_SIZE)
        X_ = p_map(to_mfcc, X_)
        X_ = p_map(normalize_mfcc,X_)
        X_ = p_map(add_dim, X_)
        try:
            prediction = model.predict(np.array(X_))
            prediction = np.argmax(prediction, axis = 1)
            prediction = np.bincount(prediction)
            prediction = np.argmax(prediction)
            print(prediction)
            if prediction == y_:
                acc = acc + 1
        except:
            pass
        sum = sum + 1
        print("-----------------------")
    print(acc/sum)