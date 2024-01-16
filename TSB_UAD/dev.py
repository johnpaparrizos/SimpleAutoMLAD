import pandas as pd
from TSB_UAD.TSB_run_det import *

# Candidate anomaly detecotrs avaiable in TSB-UAD benchmark
Det_pool = ['IForest_1_100', 'IForest_0_200', 'LOF_3_60', 'LOF_1_30', 'MP_2_False', 'MP_1_True', 'PCA_3_None', 'PCA_1_0.5',
            'NORMA_1_hierarchical', 'NORMA_3_kshape', 'HBOS_3_20', 'HBOS_1_40', 'POLY_3_5', 'POLY_2_1', 'OCSVM_1_rbf', 'OCSVM_3_poly',
            'AE_1_1_bn', 'AE_2_0_dropout', 'CNN_2_0_relu', 'CNN_3_1_sigmoid', 'LSTM_1_1_relu', 'LSTM_3_1_sigmoid']

filepath = ''    # Iput file path (Univariate by default)
df = pd.read_csv(filepath, header=None).dropna().to_numpy()
data = df[:,0].astype(float)    # shape (ts_len,)
label = df[:,1].astype(int)

for det in Det_pool:
    det_name = det.split('_')[0]
    det_hp = det.split('_')[1:]
    if det_name == 'IForest':
        IForest_score = run_iforest_dev(data, periodicity=int(det_hp[0]), n_estimators=int(det_hp[1]))   # shape (ts_len,)
    elif det_name == 'LOF':
        LOF_score = run_lof_dev(data, periodicity=int(det_hp[0]), n_neighbors=int(det_hp[1]))
    elif det_name == 'MP':
        MP_score = run_matrix_profile_dev(data, periodicity=int(det_hp[0]), cross_correlation=bool(det_hp[1]))
    elif det_name == 'PCA':
        det_hp[1]=None if det_hp[1] == 'None' else float(det_hp[1])
        PCA_score = run_pca_dev(data, periodicity=int(det_hp[0]), n_components=det_hp[1])
    elif det_name == 'NORMA':
        NORMA_score = run_norma_dev(data, periodicity=int(det_hp[0]), clustering=det_hp[1])
    elif det_name == 'HBOS':
        HBOS_score = run_hbos_dev(data, periodicity=int(det_hp[0]), n_bins=int(det_hp[1]))
    elif det_name == 'POLY':
        POLY_score = run_poly_dev(data, periodicity=int(det_hp[0]), power=int(det_hp[1]))
    elif det_name == 'OCSVM':
        OCSVM_score = run_ocsvm_dev(data, periodicity=int(det_hp[0]), kernel=det_hp[1])
    elif det_name == 'AE':
        hidden_neurons_list = [[64, 32, 32, 64], [32, 16, 32]]
        AE_score = run_ae_dev(data, periodicity=int(det_hp[0]), hidden_neurons=hidden_neurons_list[int(det_hp[1])], output_activation='relu', norm=det_hp[2])
    elif det_name == 'CNN':
        num_channel_list = [[32, 32, 40], [32, 64, 64]]
        CNN_score = run_cnn_dev(data, periodicity=int(det_hp[0]), num_channel=num_channel_list[int(det_hp[1])], activation=det_hp[2])
    elif det_name == 'LSTM':
        hidden_dim_list = [32, 64]
        LSTM_score = run_lstm_dev(data, periodicity=int(det_hp[0]), hidden_dim=hidden_dim_list[int(det_hp[1])], activation=det_hp[2])