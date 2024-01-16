import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers


class AE_MLP2:  
    def __init__(self, slidingWindow = 100,  contamination = 0.1, epochs = 10, verbose=0, hp_version=0):
        self.slidingWindow = slidingWindow
        self.contamination = contamination
        self.epochs = epochs
        self.verbose = verbose
        self.model_name = 'AE_MLP2'
        self.hp_version = hp_version

    def fit(self, X_clean, X_dirty, ratio = 0.15):

        TIME_STEPS = self.slidingWindow
        epochs = self.epochs
        

        X_train = self.create_dataset(X_clean,TIME_STEPS)
        X_test = self.create_dataset(X_dirty,TIME_STEPS)
        
        X_train = MinMaxScaler().fit_transform(X_train.T).T
        X_test = MinMaxScaler().fit_transform(X_test.T).T

        # print("Input X_train data: ", X_train.shape)
        # print("Input X_test data: ", X_test.shape)

        if self.hp_version == 0:    # Default
            model = Sequential()
            model.add(layers.Dense(32,  activation='relu'))
            model.add(layers.BatchNormalization())
            model.add(layers.Dense(16, activation='relu'))
            model.add(layers.BatchNormalization())
            model.add(layers.Dense(32, activation='relu'))
            model.add(layers.Dense(TIME_STEPS, activation='relu'))
        if self.hp_version == 1:
            model = Sequential()
            model.add(layers.Dense(32,  activation='relu'))
            model.add(layers.Dense(16, activation='relu'))
            model.add(layers.BatchNormalization())
            model.add(layers.Dense(8, activation='relu'))
            model.add(layers.Dense(16, activation='relu'))
            model.add(layers.BatchNormalization())
            model.add(layers.Dense(32, activation='relu'))
            model.add(layers.Dense(TIME_STEPS, activation='relu'))
        if self.hp_version == 2:
            model = Sequential()
            model.add(layers.Dense(32,  activation='relu'))
            model.add(layers.BatchNormalization())
            model.add(layers.Dense(8, activation='relu'))
            model.add(layers.BatchNormalization())
            model.add(layers.Dense(32, activation='relu'))
            model.add(layers.Dense(TIME_STEPS, activation='relu'))

        model.compile(optimizer='adam', loss='mse')     
        
        history = model.fit(X_train, X_train,
                        epochs=epochs,
                        batch_size=64,
                        shuffle=False,
                        validation_split=0.15,verbose=self.verbose,
                        callbacks=[EarlyStopping(monitor="val_loss", verbose=self.verbose, patience=5, mode="min")])

        test_predict = model.predict(X_test)
        test_mae_loss = np.mean(np.abs(test_predict - X_test), axis=1)
        nor_test_mae_loss = MinMaxScaler().fit_transform(test_mae_loss.reshape(-1,1)).ravel()
        score = np.zeros(len(X_dirty))
        score[self.slidingWindow//2:self.slidingWindow//2+len(test_mae_loss)]=nor_test_mae_loss
        score[:self.slidingWindow//2]=nor_test_mae_loss[0]
        score[self.slidingWindow//2+len(test_mae_loss):]=nor_test_mae_loss[-1]
        
        self.decision_scores_ = score
        
        return self
        
    
    # Generated training sequences for use in the model.
    def create_dataset(self, X, time_steps):
        output = []
        for i in range(len(X) - time_steps + 1):
            output.append(X[i : (i + time_steps)])
        return np.stack(output)