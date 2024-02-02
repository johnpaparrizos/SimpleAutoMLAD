import numpy as np
import math
from TSB_UAD.models.distance import Fourier
from TSB_UAD.models.feature import Window
from TSB_UAD.vus.metrics import get_metrics
from TSB_UAD.utils.slidingWindows import find_length, find_length_rank
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from functools import wraps
import time
import os
import logging

# import sys
# sys.path.append('..')
from TSB_UAD.models.iforest import IForest
from TSB_UAD.models.lof import LOF
from TSB_UAD.models.pca import PCA
from TSB_UAD.models.matrix_profile import MatrixProfile
from TSB_UAD.models.poly import POLY
# from TSB_UAD.models.norma import NORMA
from TSB_UAD.models.ocsvm import OCSVM
from TSB_UAD.models.hbos import HBOS
from TSB_UAD.models.autoencoder import AutoEncoder
from TSB_UAD.models.lstm import lstm
from TSB_UAD.models.AE_mlp2 import AE_MLP2
from TSB_UAD.models.cnn import cnn

def run_iforest_dev(data, periodicity, n_estimators):
    slidingWindow = find_length_rank(data, rank=periodicity)
    if slidingWindow == 1:
        clf = IForest(n_estimators=n_estimators)
        clf.fit(data)
        score = clf.decision_scores_
        score = MinMaxScaler(feature_range=(0,1)).fit_transform(score.reshape(-1,1)).ravel()
    else:
        X = Window(window = slidingWindow).convert(data).to_numpy()
        clf = IForest(n_estimators=n_estimators)
        clf.fit(X)
        score = clf.decision_scores_
        score = MinMaxScaler(feature_range=(0,1)).fit_transform(score.reshape(-1,1)).ravel()
        score = np.array([score[0]]*math.ceil((slidingWindow-1)/2) + list(score) + [score[-1]]*((slidingWindow-1)//2))
    return score

def run_lof_dev(data, periodicity, n_neighbors):
    slidingWindow = find_length_rank(data, rank=periodicity)
    X = Window(window = slidingWindow).convert(data).to_numpy()
    clf = LOF(n_neighbors=n_neighbors, n_jobs=-1)
    clf.fit(X)
    score = clf.decision_scores_
    score = MinMaxScaler(feature_range=(0,1)).fit_transform(score.reshape(-1,1)).ravel()
    score = np.array([score[0]]*math.ceil((slidingWindow-1)/2) + list(score) + [score[-1]]*((slidingWindow-1)//2))
    return score

def run_matrix_profile_dev(data, periodicity, cross_correlation):
    slidingWindow = find_length_rank(data, rank=periodicity)
    clf = MatrixProfile(window = slidingWindow, cross_correlation=cross_correlation)
    clf.fit(data)
    score = clf.decision_scores_
    score = MinMaxScaler(feature_range=(0,1)).fit_transform(score.reshape(-1,1)).ravel()
    score = np.array([score[0]]*math.ceil((slidingWindow-1)/2) + list(score) + [score[-1]]*((slidingWindow-1)//2))
    return score

def run_pca_dev(data, periodicity, n_components):
    slidingWindow = find_length_rank(data, rank=periodicity)
    X = Window(window = slidingWindow).convert(data).to_numpy()
    clf = PCA(n_components=n_components)
    clf.fit(X)
    score = clf.decision_scores_
    score = MinMaxScaler(feature_range=(0,1)).fit_transform(score.reshape(-1,1)).ravel()
    score = np.array([score[0]]*math.ceil((slidingWindow-1)/2) + list(score) + [score[-1]]*((slidingWindow-1)//2))      
    return score

def run_norma_dev(data, periodicity, clustering):
    slidingWindow = find_length_rank(data, rank=periodicity)
    clf = NORMA(pattern_length=slidingWindow, nm_size=3*slidingWindow, clustering=clustering)
    clf.fit(data)
    score = clf.decision_scores_
    score = MinMaxScaler(feature_range=(0,1)).fit_transform(score.reshape(-1,1)).ravel()
    score = np.array([score[0]]*math.ceil((slidingWindow-1)/2) + list(score) + [score[-1]]*((slidingWindow-1)//2))
    if len(score) > len(data):
        start = len(score) - len(data)
        score = score[start:]
    return score

def run_hbos_dev(data, periodicity, n_bins):
    slidingWindow = find_length_rank(data, rank=periodicity)
    X = Window(window = slidingWindow).convert(data).to_numpy()
    clf = HBOS(n_bins=n_bins)
    clf.fit(X)
    score = clf.decision_scores_
    score = MinMaxScaler(feature_range=(0,1)).fit_transform(score.reshape(-1,1)).ravel()
    score = np.array([score[0]]*math.ceil((slidingWindow-1)/2) + list(score) + [score[-1]]*((slidingWindow-1)//2))
    return score

def run_poly_dev(data, periodicity, power):
    slidingWindow = find_length_rank(data, rank=periodicity)
    clf = POLY(power=power, window = slidingWindow)
    clf.fit(data)
    measure = Fourier()
    measure.detector = clf
    measure.set_param()
    clf.decision_function(measure=measure)
    score = clf.decision_scores_
    score = MinMaxScaler(feature_range=(0,1)).fit_transform(score.reshape(-1,1)).ravel()
    return score

def run_ocsvm_dev(data, periodicity, kernel):
    slidingWindow = find_length_rank(data, rank=periodicity)
    if int(len(data)) < 2560:
        X_train = Window(window = slidingWindow).convert(data[:int(0.3*len(data)+slidingWindow)]).to_numpy()
    elif int(len(data)) > 20000:
        X_train = Window(window = slidingWindow).convert(data[:int(0.05*len(data)+slidingWindow)]).to_numpy()
    else:
        X_train = Window(window = slidingWindow).convert(data[:int(0.1*len(data)+slidingWindow)]).to_numpy()
    X_test = Window(window = slidingWindow).convert(data).to_numpy()
    X_train_ = MinMaxScaler(feature_range=(0,1)).fit_transform(X_train.T).T
    X_test_ = MinMaxScaler(feature_range=(0,1)).fit_transform(X_test.T).T
    clf = OCSVM(kernel=kernel)
    clf.fit(X_train_, X_test_)
    score = clf.decision_scores_
    score = MinMaxScaler(feature_range=(0,1)).fit_transform(score.reshape(-1,1)).ravel()
    score = np.array([score[0]]*math.ceil((slidingWindow-1)/2) + list(score) + [score[-1]]*((slidingWindow-1)//2))
    return score

# def run_ocsvm_dev(data, periodicity, nu):
#     slidingWindow = find_length_rank(data, rank=periodicity)
#     if int(len(data)) < 2560:
#         X_train = Window(window = slidingWindow).convert(data[:int(0.3*len(data)+slidingWindow)]).to_numpy()
#     else:
#         X_train = Window(window = slidingWindow).convert(data[:int(0.1*len(data)+slidingWindow)]).to_numpy()
#     X_test = Window(window = slidingWindow).convert(data).to_numpy()
#     X_train_ = MinMaxScaler(feature_range=(0,1)).fit_transform(X_train.T).T
#     X_test_ = MinMaxScaler(feature_range=(0,1)).fit_transform(X_test.T).T
#     clf = OCSVM(nu=nu)
#     clf.fit(X_train_, X_test_)
#     score = clf.decision_scores_
#     score = MinMaxScaler(feature_range=(0,1)).fit_transform(score.reshape(-1,1)).ravel()
#     score = np.array([score[0]]*math.ceil((slidingWindow-1)/2) + list(score) + [score[-1]]*((slidingWindow-1)//2))
#     return score

def run_ae_dev(data, periodicity, hidden_neurons, output_activation, norm):
    slidingWindow = find_length_rank(data, rank=periodicity)
    # clf = AE_MLP2(slidingWindow = slidingWindow, epochs=100, verbose=0, hp_version=hidden_neurons)
    try:
        clf = AutoEncoder(slidingWindow = slidingWindow, epochs=100, batch_size=64, hidden_neurons=hidden_neurons, output_activation=output_activation, norm=norm)
        if int(len(data)) < 2560:
            data_train = data[:int(0.3*len(data)+slidingWindow)]
        else:
            data_train = data[:int(0.1*len(data)+slidingWindow)]
        clf.fit(data_train, data)
        score = clf.decision_scores_
    except:
        print('Decrease batch size...')
        clf = AutoEncoder(slidingWindow = slidingWindow, epochs=100, batch_size=16, hidden_neurons=hidden_neurons, output_activation=output_activation, norm=norm)
        if int(len(data)) < 2560:
            data_train = data[:int(0.3*len(data)+slidingWindow)]
        else:
            data_train = data[:int(0.1*len(data)+slidingWindow)]
        clf.fit(data_train, data)
        score = clf.decision_scores_
    score = MinMaxScaler(feature_range=(0,1)).fit_transform(score.reshape(-1,1)).ravel()
    return score

# def run_ae_dev(data, hp_version):
#     slidingWindow = find_length_rank(data, rank=1)
#     clf = AE_MLP2(slidingWindow = slidingWindow, epochs=100, verbose=0, hp_version=hp_version)
#     # clf = AutoEncoder(slidingWindow = slidingWindow, epochs=100, verbose=0, output_activation=output_activation, norm=norm)
#     if int(len(data)) < 2560:
#         data_train = data[:int(0.3*len(data)+slidingWindow)]
#     else:
#         data_train = data[:int(0.1*len(data)+slidingWindow)]
#     clf.fit(data_train, data)
#     score = clf.decision_scores_
#     score = MinMaxScaler(feature_range=(0,1)).fit_transform(score.reshape(-1,1)).ravel()
#     return score

def run_cnn_dev(data, periodicity, num_channel, activation):
    slidingWindow = find_length_rank(data, rank=periodicity)
    clf = cnn(slidingwindow = slidingWindow, epochs=100, patience=5, verbose=0, num_channel=num_channel, activation=activation)
    if int(len(data)) < 2560:
        data_train = data[:int(0.3*len(data)+slidingWindow)]
    else:
        data_train = data[:int(0.1*len(data)+slidingWindow)]
    try:
        clf.fit(data_train, data)
    except:
        clf.fit_short(data_train, data)
    measure = Fourier()
    measure.detector = clf
    measure.set_param()
    clf.decision_function(measure=measure)
    score = clf.decision_scores_
    score = MinMaxScaler(feature_range=(0,1)).fit_transform(score.reshape(-1,1)).ravel()
    return score

def run_lstm_dev(data, periodicity, hidden_dim, activation):
    slidingWindow = find_length_rank(data, rank=periodicity)
    clf = lstm(slidingwindow = slidingWindow, epochs = 50, patience = 5, verbose=0, hidden_dim=hidden_dim, activation=activation)
    if int(len(data)) < 2560:
        data_train = data[:int(0.3*len(data)+slidingWindow)]
    if int(len(data)) > 20000:
        data_train = data[:int(0.05*len(data)+slidingWindow)]
    else:
        data_train = data[:int(0.1*len(data)+slidingWindow)]
    clf.fit(data_train, data)
    measure = Fourier()
    measure.detector = clf
    measure.set_param()
    clf.decision_function(measure=measure)
    score = clf.decision_scores_
    score = MinMaxScaler(feature_range=(0,1)).fit_transform(score.reshape(-1,1)).ravel()
    return score
