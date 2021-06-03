import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout

def Model(input_shape, num_classes): #64
    '''
    2D convolutional neural network
    :param X_train: Numpy array of mfccs
    :param y_train: Binary matrix based on labels
    :return: model
    '''

    #initiate model

    model = tf.keras.Sequential()

    model.add(Conv2D(32, kernel_size=(3,3), activation='relu',
                     data_format="channels_last",
                     input_shape=input_shape))

    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Conv2D(64,kernel_size=(3,3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])

    return model

def save_model(model, model_filename):
    '''
    Save model to file
    :param model: Trained model to be saved
    :param model_filename: Filename
    :return: None
    '''
    model.save('../models/{}.h5'.format(model_filename))  # creates a HDF5 file 'my_model.h5'
