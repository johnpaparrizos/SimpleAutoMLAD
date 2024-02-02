# -*- coding: utf-8 -*-
import numpy as np
import warnings, sys, argparse, time
import pandas as pd
sys.path.append('../..')
from TSB_UAD.utils.slidingWindows import find_length_rank
from TSB_UAD.TSB_run_det import *
from sklearn.preprocessing import StandardScaler

# Det_Pool_Avg = ['IForest_3_200', 'IForest_1_100', 'IForest_0_200', 'LOF_3_60', 'LOF_1_30', 'MP_2_False', 'PCA_3_None', 'PCA_1_0.5',
#             'HBOS_3_20', 'HBOS_1_40', 'POLY_3_5', 'POLY_2_1', 'OCSVM_1_rbf', 'OCSVM_3_poly',
#             'AE_1_1_bn', 'AE_2_0_dropout', 'CNN_2_0_relu', 'CNN_3_1_sigmoid', 'LSTM_1_1_relu', 'LSTM_3_1_sigmoid']

# Omit the DL models
Det_Pool_Avg = ['IForest_3_200', 'IForest_1_100', 'IForest_0_200', 'LOF_3_60', 'LOF_1_30', 'MP_2_False', 'PCA_3_None', 'PCA_1_0.5',
            'HBOS_3_20', 'HBOS_1_40', 'POLY_3_5', 'POLY_2_1', 'OCSVM_1_rbf', 'OCSVM_3_poly']

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='AutoML Solution for TSAD')
    parser.add_argument('-a', '--AutoML_Solution_Pool', nargs='*',
                        default='Avg_ens', required=True,
                        help='Avg_ens Pointwise_Preds Pointwise_Preds_Ens Pairwise_Preds Pairwise_Preds_Ens Listwise_Preds Listwise_Preds_Ens')
    args = parser.parse_args()

    # TODO: Needs to be modified
    filepath = '/content/sample_data/california_housing_test.csv'       
    df = pd.read_csv(filepath, header=None).dropna().to_numpy()
    data = df[1:,0].astype(float)

    print('data: ', data.shape)

    if 'Avg_ens' in args.AutoML_Solution_Pool:
        print('--- Using Avg_ens ---')
        ens_scores_list = []
        for det in Det_Pool_Avg:
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
            ens_scores_list.append(score)

        det_scores = np.array(ens_scores_list).T
        scaler = StandardScaler()
        scaler.fit(det_scores)
        standardized_det_scores = scaler.transform(det_scores)

        avg_ens_scores = np.mean(standardized_det_scores, axis=1)
        
        print('Output Avg Ens Anomaly Score: ', avg_ens_scores.shape)






