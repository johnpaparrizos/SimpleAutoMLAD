# -*- coding: utf-8 -*-
import numpy as np
import warnings, sys, argparse, time
import pandas as pd
sys.path.append('../..')
from TSB_UAD.utils.slidingWindows import find_length_rank
from TSB_UAD.TSB_run_det import *
from sklearn.preprocessing import StandardScaler

# (1) Det set
Det_Pool_Avg = ['IForest_3_200', 'IForest_1_100', 'IForest_0_200', 'LOF_3_60', 'LOF_1_30', 'MP_2_False', 'PCA_3_None', 'PCA_1_0.5',
            'HBOS_3_20', 'HBOS_1_40', 'POLY_3_5', 'POLY_2_1', 'OCSVM_1_rbf', 'OCSVM_3_poly',
            'AE_1_1_bn', 'AE_2_0_dropout', 'CNN_2_0_relu', 'CNN_3_1_sigmoid', 'LSTM_1_1_relu', 'LSTM_3_1_sigmoid']
# (2) Omit the DL models
Det_Pool_Avg_wo_DL = ['IForest_3_200', 'IForest_1_100', 'IForest_0_200', 'LOF_3_60', 'LOF_1_30', 'MP_2_False', 'PCA_3_None', 'PCA_1_0.5',
            'HBOS_3_20', 'HBOS_1_40', 'POLY_3_5', 'POLY_2_1', 'OCSVM_1_rbf', 'OCSVM_3_poly']
# (3) Top 10 Det set
Top_10_Det = ['MP_2_False', 'LOF_1_30', 'AE_1_1_bn', 'LOF_3_60', 'IForest_1_100', 'PCA_1_0.5', 'IForest_3_200', 'PCA_3_None', 'AE_2_0_dropout', 'POLY_2_1']

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='AutoML Solution for TSAD')
    parser.add_argument('-a', '--AutoML_Solution_Pool', nargs='*',
                        default='Avg_ens', required=True,
                        help='Avg_ens Max_ens Avg_of_Max')
    args = parser.parse_args()

    # Step 1 TODO: Needs to be modified
    filepath = 'example_ts/toy_ts.out'       
    df = pd.read_csv(filepath, header=None).dropna().to_numpy()
    data = df[:,0].astype(float)
    label = df[:,1].astype(int)
    slidingWindow = find_length_rank(data, rank=1)

    print('data: ', data.shape)

    # Step 2: Generate the anomaly scores
    unique_score = []
    for det in Det_Pool_Avg_wo_DL:
        det_name = det.split('_')[0]
        det_hp = det.split('_')[1:]
        if det_name == 'IForest':
            score = run_iforest_dev(data, periodicity=int(det_hp[0]), n_estimators=int(det_hp[1]))
        elif det_name == 'LOF':
            score = run_lof_dev(data, periodicity=int(det_hp[0]), n_neighbors=int(det_hp[1]))
        elif det_name == 'MP':
            score = run_matrix_profile_dev(data, periodicity=int(det_hp[0]), cross_correlation=bool(det_hp[1]))
        elif det_name == 'PCA':
            det_hp[1]=None if det_hp[1] == 'None' else float(det_hp[1])
            score = run_pca_dev(data, periodicity=int(det_hp[0]), n_components=det_hp[1])
        elif det_name == 'NORMA':
            score = run_norma_dev(data, periodicity=int(det_hp[0]), clustering=det_hp[1])
        elif det_name == 'HBOS':
            score = run_hbos_dev(data, periodicity=int(det_hp[0]), n_bins=int(det_hp[1]))
        elif det_name == 'POLY':
            score = run_poly_dev(data, periodicity=int(det_hp[0]), power=int(det_hp[1]))
        elif det_name == 'OCSVM':
            score = run_ocsvm_dev(data, periodicity=int(det_hp[0]), kernel=det_hp[1])
        elif det_name == 'AE':
            hidden_neurons_list = [[64, 32, 32, 64], [32, 16, 32]]
            score = run_ae_dev(data, periodicity=int(det_hp[0]), hidden_neurons=hidden_neurons_list[int(det_hp[1])], output_activation='relu', norm=det_hp[2])
        elif det_name == 'CNN':
            num_channel_list = [[32, 32, 40], [32, 64, 64]]
            score = run_cnn_dev(data, periodicity=int(det_hp[0]), num_channel=num_channel_list[int(det_hp[1])], activation=det_hp[2])
        elif det_name == 'LSTM':
            hidden_dim_list = [32, 64]
            score = run_lstm_dev(data, periodicity=int(det_hp[0]), hidden_dim=hidden_dim_list[int(det_hp[1])], activation=det_hp[2])
        unique_score.append(score)
    det_scores = np.array(unique_score).T

    scaler = StandardScaler()
    scaler.fit(det_scores)
    standardized_det_scores = scaler.transform(det_scores)      # (ts, num_of_det)

    print('standardized_det_scores: ', standardized_det_scores.shape)

    if 'Avg_ens' in args.AutoML_Solution_Pool:
        print('--- Using Avg_ens ---')
        avg_ens_scores = np.mean(standardized_det_scores, axis=1)
        Avg_ens_result = get_metrics(avg_ens_scores, label, metric='all', slidingWindow=slidingWindow)
        print('VUS-PR: ', Avg_ens_result['VUS_PR'])

    if 'Max_ens' in args.AutoML_Solution_Pool:
        print('--- Using Avg_of_Max ---')
        max_ens_scores = np.max(standardized_det_scores, axis=1)
        Max_ens_result = get_metrics(max_ens_scores, label, metric='all', slidingWindow=slidingWindow)
        print('VUS-PR: ', Max_ens_result['VUS_PR'])

    if 'Avg_of_Max' in args.AutoML_Solution_Pool:
        print('--- Using Avg_of_Max ---')
        max_ens_scores_list = []
        for i in range(20):
            indices = np.random.choice(standardized_det_scores.shape[1], 5, replace=False)
            max_ens_scores_list.append(np.max(standardized_det_scores[:,indices], axis=1))
        Avg_of_Max_scores = np.mean(np.array(max_ens_scores_list), axis=0)
        Avg_of_Max_result = get_metrics(Avg_of_Max_scores, label, metric='all', slidingWindow=slidingWindow)
        print('VUS-PR: ', Avg_of_Max_result['VUS_PR'])