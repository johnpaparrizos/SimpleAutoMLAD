from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout
from tensorflow.keras.callbacks import EarlyStopping

import tensorflow as tf
import numpy as np
from sklearn.preprocessing import MinMaxScaler

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['CUDA_VISIBLE_DEVICES']='0'
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

class cnn:
    def __init__(self, slidingwindow = 100, predict_time_steps = 1, contamination = 0.1, epochs = 10, patience = 10, verbose=0, num_channel=[32, 32, 64], activation='relu'):
        self.slidingwindow = slidingwindow
        self.predict_time_steps = predict_time_steps
        self.contamination = contamination
        self.epochs = epochs
        self.patience = patience
        self.verbose = verbose
        self.kernel_size = 2
        self.dropout_rate = 0.25
        self.conv_strides = 1
        self.dense_unit = num_channel[-1]
        self.num_filter = num_channel[:-1]
        self.activation = activation
        
    def fit(self, X_clean, X_dirty, ratio = 0.15):

        slidingwindow = self.slidingwindow
        predict_time_steps = self.predict_time_steps
        self.n_test_ = len(X_dirty)

        X_train, Y_train = self.create_dataset(X_clean, slidingwindow, predict_time_steps)
        X_test, Y_test = self.create_dataset(X_dirty, slidingwindow, predict_time_steps)
        
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
        
        # print("X_train: ", X_train.shape)   # (461, 11, 1)
        # print("X_test: ", X_test.shape)         # (4718, 11, 1)
        
        model = Sequential()
        model.add(Conv1D(filters=self.num_filter[0], kernel_size=self.kernel_size, strides=self.conv_strides, padding='valid', activation=self.activation,
                        input_shape=(slidingwindow, 1)))
        model.add(MaxPooling1D(pool_size=2))
        for i, num_filter in enumerate(self.num_filter, 1):
            model.add(Conv1D(filters=num_filter, kernel_size=self.kernel_size, strides=self.conv_strides, padding='valid', activation=self.activation))
            model.add(MaxPooling1D(pool_size=2))       
        model.add(Flatten())
        model.add(Dense(units=self.dense_unit, activation='relu'))
        model.add(Dropout(self.dropout_rate))
        model.add(Dense(units=self.predict_time_steps))

        model.compile(loss='mse', optimizer='adam')
        
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=self.verbose, patience=self.patience)
        
        model.fit(X_train, Y_train, validation_split=ratio,
                  epochs=self.epochs, batch_size=64, verbose=self.verbose, callbacks=[es])
        
        prediction = model.predict(X_test)

        self.Y_test = Y_test
        self.prediction = prediction
        self.estimator = model
        self.n_initial = X_train.shape[0]
        
        return self

    def fit_short(self, X_clean, X_dirty, ratio = 0.15):

        slidingwindow = self.slidingwindow
        predict_time_steps = self.predict_time_steps
        self.n_test_ = len(X_dirty)

        X_train, Y_train = self.create_dataset(X_clean, slidingwindow, predict_time_steps)
        X_test, Y_test = self.create_dataset(X_dirty, slidingwindow, predict_time_steps)
        
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
        
        model = Sequential()
        model.add(Conv1D(filters=self.num_filter[0], kernel_size=self.kernel_size, strides=self.conv_strides, padding='same', activation='relu',
                        input_shape=(slidingwindow, 1)))
        model.add(MaxPooling1D(pool_size=2))
        for i, num_filter in enumerate(self.num_filter[:-1], 1):
            model.add(Conv1D(filters=num_filter, kernel_size=self.kernel_size, strides=self.conv_strides, padding='same', activation='relu'))
            model.add(MaxPooling1D(pool_size=2))
        model.add(Conv1D(filters=num_filter, kernel_size=self.kernel_size, strides=self.conv_strides, padding='same', activation='relu'))
        
        model.add(Flatten())
        model.add(Dense(units=self.dense_unit, activation='relu'))
        model.add(Dropout(self.dropout_rate))
        model.add(Dense(units=self.predict_time_steps))

        model.compile(loss='mse', optimizer='adam')
        
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=self.verbose, patience=self.patience)
        
        model.fit(X_train, Y_train, validation_split=ratio,
                  epochs=self.epochs, batch_size=64, verbose=self.verbose, callbacks=[es])
        
        prediction = model.predict(X_test)

        self.Y_test = Y_test
        self.prediction = prediction
        self.estimator = model
        self.n_initial = X_train.shape[0]
        
        return self

    def create_dataset(self, X, slidingwindow, predict_time_steps=1):
        Xs, ys = [], []
        for i in range(len(X) - slidingwindow - predict_time_steps+1):
            tmp = X[i : i + slidingwindow + predict_time_steps]
            tmp= MinMaxScaler(feature_range=(0,1)).fit_transform(tmp.reshape(-1,1)).ravel()
            x = tmp[:slidingwindow]
            y = tmp[slidingwindow:]
            
            Xs.append(x)
            ys.append(y)
        return np.array(Xs), np.array(ys)
    

    
    def decision_function(self, X= False, measure = None):
        """Derive the decision score based on the given distance measure
        Parameters
        ----------
        X : numpy array of shape (n_samples, )
            The input samples.
        measure : object
            object for given distance measure with methods to derive the score
        Returns
        -------
        self : object
            Fitted estimator.
        """
        if type(X) != bool:
            self.X_train_ = X

        Y_test = self.Y_test

        score = []
        prediction = self.prediction
        # print("prediction: ", prediction.shape)
        # print("Y_test: ", Y_test.shape)
        for i in range(prediction.shape[0]):
            score.append(measure.measure(Y_test[i], prediction[i], 0))
        
        score = np.array(score)
        decision_scores_ = np.zeros(self.n_test_)
        
        decision_scores_[self.slidingwindow : (self.n_test_-self.predict_time_steps+1)]=score
        decision_scores_[: self.slidingwindow] = score[0]
        decision_scores_[self.n_test_-self.predict_time_steps+1:]=score[-1]
        
        self.decision_scores_ = decision_scores_
        return self