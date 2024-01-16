import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
import math


class AutoEncoder:  
    def __init__(self, slidingWindow = 100,  contamination = 0.1, epochs = 10, verbose=0, batch_size=64, hidden_neurons=[64, 32, 32, 64], norm='bn', output_activation='relu'):
        self.slidingWindow = slidingWindow
        self.contamination = contamination
        self.epochs = epochs
        self.verbose = verbose
        self.batch_size = batch_size
        self.hidden_activation = 'relu'
        self.output_activation = output_activation
        self.dropout_rate = 0.2
        self.l2_regularizer = 0.1
        self.hidden_neurons = hidden_neurons
        self.norm = norm    # bn, dropout

    def fit(self, X_clean, X_dirty, ratio = 0.15):

        TIME_STEPS = self.slidingWindow
        epochs = self.epochs
        
        X_train = self.create_dataset(X_clean,TIME_STEPS)
        X_test = self.create_dataset(X_dirty,TIME_STEPS)
        
        X_train = MinMaxScaler().fit_transform(X_train.T).T
        X_test = MinMaxScaler().fit_transform(X_test.T).T

        # X_train = StandardScaler().fit_transform(X_train.T).T
        # X_test = StandardScaler().fit_transform(X_test.T).T

        # print("Input X_train data: ", X_train.shape)
        # print("Input X_test data: ", X_test.shape)
        # print('TIME_STEPS: ', TIME_STEPS)

        model = Sequential()
        # # Input layer
        # model.add(Dense(
        #     self.hidden_neurons[0], activation=self.hidden_activation,
        #     input_shape=(TIME_STEPS,),
        #     activity_regularizer=l2(self.l2_regularizer)))
        # if self.norm== 'bn': model.add(BatchNormalization())
        # if self.norm== 'dropout': model.add(Dropout(self.dropout_rate))
        # # Additional layers
        # for i, hidden_neuron in enumerate(self.hidden_neurons, 1):
        #     model.add(Dense(
        #         hidden_neuron,
        #         activation=self.hidden_activation,
        #         activity_regularizer=l2(self.l2_regularizer)))
        #     if self.norm== 'bn': model.add(BatchNormalization())
        #     if self.norm== 'dropout': model.add(Dropout(self.dropout_rate))
        # # Output layers
        # model.add(Dense(TIME_STEPS, activation=self.output_activation,
        #                 activity_regularizer=l2(self.l2_regularizer)))

        # Input layer
        model.add(Dense(self.hidden_neurons[0], activation=self.hidden_activation))
        if self.norm== 'bn': model.add(BatchNormalization())
        if self.norm== 'dropout': model.add(Dropout(self.dropout_rate))
        # Additional layers
        for i, hidden_neuron in enumerate(self.hidden_neurons[:-1], 1):
            model.add(Dense(hidden_neuron, activation=self.hidden_activation))
            if self.norm== 'bn': model.add(BatchNormalization())
            if self.norm== 'dropout': model.add(Dropout(self.dropout_rate))
        model.add(Dense(self.hidden_neurons[-1], activation=self.hidden_activation))
        # Output layers
        model.add(Dense(TIME_STEPS, activation=self.output_activation))
        
        
        model.compile(optimizer='adam', loss='mse')     
        
        history = model.fit(X_train, X_train,
                        epochs=epochs,
                        batch_size=self.batch_size,
                        shuffle=True,
                        validation_split=0.15,
                        verbose=self.verbose,
                        callbacks=[EarlyStopping(monitor="val_loss", verbose=self.verbose, patience=5, mode="min")])

        test_predict = model.predict(X_test)
        test_mae_loss = np.mean(np.abs(test_predict - X_test), axis=1)
        score = MinMaxScaler().fit_transform(test_mae_loss.reshape(-1,1)).ravel()
        score = np.array([score[0]]*math.ceil((self.slidingWindow-1)/2) + list(score) + [score[-1]]*((self.slidingWindow-1)//2))
        self.decision_scores_ = score

        return self
        
    
    # Generated training sequences for use in the model.
    def create_dataset(self, X, time_steps):
        output = []
        for i in range(len(X) - time_steps + 1):
            output.append(X[i : (i + time_steps)])
        return np.stack(output)