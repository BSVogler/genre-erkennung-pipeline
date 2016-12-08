#!/usr/bin/env python3

import numpy as np
np.random.seed(1337)  # for reproducibility
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Convolution1D, MaxPooling1D
from keras.layers import LSTM, GRU, Flatten
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import pickle
import json

numGenres=3
def mfcc_model(input_shape):
    nb_filter = 100
    filter_length = 4
    hidden_dims = 250

    pool_length = 4

    # LSTM
    lstm_output_size = 300

    # create model
    model = Sequential()

    model.add(Convolution1D(
                            input_shape=input_shape,
                            nb_filter=nb_filter,
                            filter_length=filter_length,
                            border_mode='valid',
                            subsample_length=1))
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_length=pool_length))
    model.add(Dropout(0.4))
    # #
    # # #
    model.add(Convolution1D(
                            nb_filter=int(nb_filter/5),
                            filter_length=int(filter_length),
                            border_mode='valid',
                            subsample_length=1))
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_length=pool_length))
    model.add(Dropout(0.4))
    # model.add(Flatten())
    # # #
    # # #
    #
    # model.add(Convolution1D(
    #                         nb_filter=int(nb_filter/10),
    #                         filter_length=int(filter_length/20),
    #                         border_mode='valid',
    #                         subsample_length=2))
    # model.add(Activation('relu'))
    # model.add(MaxPooling1D(pool_length=pool_length))
    # model.add(Dropout(0.2))
    #
    model.add(LSTM(lstm_output_size,
                    # input_shape=input_shape,
                    activation='sigmoid',
                    inner_activation='hard_sigmoid'))
    #
    # model.add(Dropout(0.2))



    # model.add(Flatten())
    # model.add(LSTM(lstm_output_size))
    model.add(Dropout(0.4))
    model.add(Dense(numGenres))
    model.add(Dropout(0.2))
    #
    # model.add(Convolution1D(
    #                         nb_filter=int(nb_filter/10),
    #                         filter_length=int(filter_length/5),
    #                         border_mode='valid',
    #                         subsample_length=1))
    # model.add(Activation('relu'))
    # model.add(MaxPooling1D(pool_length=pool_length))
    # model.add(Dropout(0.4))


    # model.add(Lambda(max_1d, output_shape=(nb_filter)))
    # model.add(LSTM(lstm_output_size))
    # model.add(Dropout(0.2))
    # # We add a vanilla hidden layer:
    # model.add(Activation('relu'))
    # model.add(Dense(hidden_dims))
    # model.add(Flatten())
    # model.add(Dense(200))
    # model.add(Activation("sigmoid"))
    # model.add(Dropout(0.2))

    return model

if __name__=="__main__":
    
    if not os.path.exists("model_weights"):
        os.makedirs("model_weights")
        
    #check if can write to file
    f = open('model_weights/mfcc_model_weights.hdf5', 'a')
    f.write('test')
    f.close()
   
    # print(X)
    # load vectorized song features
    #
    batch_size = 20
    nb_epoch = 50
    X = pickle.load(open("pickled_vectors/mfcc_coefficients_training_vector.pickle","rb"))
    y = pickle.load(open("pickled_vectors/mfcc_coefficients_label.pickle","rb"))

    X_test = pickle.load(open("pickled_vectors/mfcc_coefficients_evaluation_training_vector.pickle","rb"))
    y_test = pickle.load(open("pickled_vectors/mfcc_coefficients_evaluation_label.pickle","rb"))

    model = mfcc_model((X.shape[1],X.shape[2]))
    model.add(Dense(numGenres))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy']
                  )

    print("X shape",X.shape)
    print("y shape",y.shape)
    print("X_test", X_test.shape)
    print("y_test", y_test.shape)

    print("Fitting")

    history = model.fit(X, y,
              batch_size=batch_size,
              nb_epoch=nb_epoch,
              validation_data=(X_test, y_test),
              shuffle="batch"
            )

    with open("experimental_results.json","w") as f:
        f.write(json.dumps(history.history, sort_keys=True, indent=4, separators=(',', ': ')))

    for k,v in history.history.items():
        # print(k,v)
        _keys = list(history.history.keys())
        _keys.sort()
        plt.subplot(411+_keys.index(k))
        # x_space = np.linspace(0,1,len(v))
        plt.title(k)

        plt.plot(range(0,len(v)),v,marker="8",linewidth=1.5)
        
    model.save_weights("model_weights/mfcc_model_weights.hdf5",overwrite=True)
    
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    plt.show()
