# -*- coding: utf-8 -*-
import numpy as np
import warnings, sys, argparse, time
import pandas as pd
from collections import Counter
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from utils.config import *
from utils.train_deep_model_utils import json_file
sys.path.append('../..')
from TSB_UAD.utils.slidingWindows import find_length_rank
from TSB_UAD.TSB_run_det import *

Det_Pool_Avg = ['IForest_3_200', 'IForest_1_100', 'IForest_0_200', 'LOF_3_60', 'LOF_1_30', 'MP_2_False', 'MP_1_True', 'PCA_3_None', 'PCA_1_0.5',
            'HBOS_3_20', 'HBOS_1_40', 'POLY_3_5', 'POLY_2_1', 'OCSVM_1_rbf', 'OCSVM_3_poly',
            'AE_1_1_bn', 'AE_2_0_dropout', 'CNN_2_0_relu', 'CNN_3_1_sigmoid', 'LSTM_1_1_relu', 'LSTM_3_1_sigmoid']

class TimeSeriesDataset_Demo(Dataset):
    def __init__(self, data, window_size=1024):
        self.data = data
        self.window_size = window_size
        self.n_windows = len(data) // window_size

    def __len__(self):
        return self.n_windows

    def __getitem__(self, idx):
        start_idx = idx * self.window_size
        end_idx = start_idx + self.window_size
        return self.data[start_idx:end_idx]

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='AutoML Solution for TSAD')
    parser.add_argument('-a', '--AutoML_Solution_Pool', nargs='*',
                        default='Avg_ens', required=True,
                        help='Avg_ens Pointwise_Preds Pointwise_Preds_Ens Pairwise_Preds Pairwise_Preds_Ens Listwise_Preds Listwise_Preds_Ens')
    parser.add_argument('--ens_top_selection', type=int, default=5)
    args = parser.parse_args()

    Pointwise_preds_model_path = '/data/liuqinghua/code/ts/AutoTSAD/weights/Pointwise_preds'
    Pairwise_preds_model_path = '/data/liuqinghua/code/ts/AutoTSAD/weights/Pairwise_preds'
    Listwise_preds_model_path = '/data/liuqinghua/code/ts/AutoTSAD/weights/Listwise_preds'
    model_parameters = json_file('models/configuration/resnet_default.json')

    # TODO: Needs to be modified
    filepath = '/data/liuqinghua/code/ts/TSB-UAD/data/public/NASA-MSL/T-13.test.out'       
    df = pd.read_csv(filepath, header=None).dropna().to_numpy()
    data = df[:,0].astype(float)

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
        avg_ens_scores = np.mean(np.array(ens_scores_list), axis=1)

    if 'Pointwise_Preds' in args.AutoML_Solution_Pool:
        print('--- Using Pointwise_Preds ---')

        test_data =  TimeSeriesDataset_Demo(data)
        # Load model
        t1 = time.time()
        model = deep_models['resnet'](**model_parameters)
        model.load_state_dict(torch.load(Pointwise_preds_model_path))
        model.eval()
        model.to('cuda')
        all_preds = []
    
        # Timeserie to batches
        test_loader = DataLoader(test_data, batch_size=256, shuffle=False)
        for inputs in test_loader:
            inputs = inputs.to('cuda')
            inputs = torch.unsqueeze(inputs, 1)
            # Make predictions
            outputs = model(inputs.float())     # (N, 23)
            # Compute topk acc
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.tolist())
        counter = Counter(all_preds)
        most_voted = counter.most_common(1)     # [(most_voted_id, freq)]
        running_time = time.time() - t1
        
        selected_model = Det_Pool[most_voted[0][0]]
        print('selected_model: ', selected_model)

    if 'Pointwise_Preds_Ens' in args.AutoML_Solution_Pool:
        print('--- Using Pointwise_Preds_Ens ---')

        test_data =  TimeSeriesDataset_Demo(data)
        # Load model
        t1 = time.time()
        model = deep_models['resnet'](**model_parameters)
        model.load_state_dict(torch.load(Pointwise_preds_model_path))
        model.eval()
        model.to('cuda')
        all_ranks = []
    
        # Timeserie to batches
        test_loader = DataLoader(test_data, batch_size=256, shuffle=False)
        for inputs in test_loader:
            inputs = inputs.to('cuda')
            inputs = torch.unsqueeze(inputs, 1)
            # Make predictions
            outputs = model(inputs.float())     # (N, 23)
            sorted, preds_id = outputs.sort(dim=1)
            _, preds_id = preds_id.sort(dim=1)
            preds_agg = np.sum(preds_id.detach().cpu().numpy(), axis=0)
            all_ranks.extend(preds_agg.tolist())
        preds = np.argsort(np.sum(np.array(all_ranks).reshape(-1,23), axis=0), axis=0)[::-1][:args.ens_top_selection]     # Top 1 
        running_time = time.time() - t1
        selected_models = []
        score_list = []
        for i in preds:
            selected_models.append(Det_Pool[i])

        print('selected_models: ', selected_models)

    if 'Pairwise_Preds' in args.AutoML_Solution_Pool:
        print('--- Using Pairwise_Preds ---')

        test_data =  TimeSeriesDataset_Demo(data)
        # Load model
        t1 = time.time()
        model = deep_models['resnet'](**model_parameters)
        model.load_state_dict(torch.load(Pairwise_preds_model_path))
        model.eval()
        model.to('cuda')
        all_ranks = []
    
        # Timeserie to batches
        test_loader = DataLoader(test_data, batch_size=256, shuffle=False)
        for inputs in test_loader:
            inputs = inputs.to('cuda')
            inputs = torch.unsqueeze(inputs, 1)
            # Make predictions
            outputs = model(inputs.float())     # (N, 23)
            sorted, preds_id = outputs.sort(dim=1)
            _, preds_id = preds_id.sort(dim=1)
            preds_agg = np.sum(preds_id.detach().cpu().numpy(), axis=0)
            all_ranks.extend(preds_agg.tolist())
        preds = np.argsort(np.sum(np.array(all_ranks).reshape(-1,23), axis=0), axis=0)[::-1][:1]     # Top 1 
        running_time = time.time() - t1

        selected_model = Det_Pool[preds[0]]
        print('selected_model: ', selected_model)

    if 'Pairwise_Preds_Ens' in args.AutoML_Solution_Pool:
        print('--- Using Pairwise_Preds_Ens ---')

        test_data =  TimeSeriesDataset_Demo(data)
        # Load model
        t1 = time.time()
        model = deep_models['resnet'](**model_parameters)
        model.load_state_dict(torch.load(Pairwise_preds_model_path))
        model.eval()
        model.to('cuda')
        all_ranks = []
    
        # Timeserie to batches
        test_loader = DataLoader(test_data, batch_size=256, shuffle=False)
        for inputs in test_loader:
            inputs = inputs.to('cuda')
            inputs = torch.unsqueeze(inputs, 1)
            # Make predictions
            outputs = model(inputs.float())     # (N, 23)
            sorted, preds_id = outputs.sort(dim=1)
            _, preds_id = preds_id.sort(dim=1)
            preds_agg = np.sum(preds_id.detach().cpu().numpy(), axis=0)
            all_ranks.extend(preds_agg.tolist())
        preds = np.argsort(np.sum(np.array(all_ranks).reshape(-1,23), axis=0), axis=0)[::-1][:args.ens_top_selection]     # Top 1 
        running_time = time.time() - t1
        selected_models = []
        score_list = []
        for i in preds:
            selected_models.append(Det_Pool[i])

        print('selected_model: ', selected_models)

    if 'Listwise_Preds' in args.AutoML_Solution_Pool:
        print('--- Using Listwise_Preds ---')

        test_data =  TimeSeriesDataset_Demo(data)
        # Load model
        t1 = time.time()
        model = deep_models['resnet'](**model_parameters)
        model.load_state_dict(torch.load(Listwise_preds_model_path))
        model.eval()
        model.to('cuda')
        all_ranks = []
    
        # Timeserie to batches
        test_loader = DataLoader(test_data, batch_size=256, shuffle=False)
        for inputs in test_loader:
            inputs = inputs.to('cuda')
            inputs = torch.unsqueeze(inputs, 1)
            # Make predictions
            outputs = model(inputs.float())     # (N, 23)
            sorted, preds_id = outputs.sort(dim=1)
            _, preds_id = preds_id.sort(dim=1)
            preds_agg = np.sum(preds_id.detach().cpu().numpy(), axis=0)
            all_ranks.extend(preds_agg.tolist())
        preds = np.argsort(np.sum(np.array(all_ranks).reshape(-1,23), axis=0), axis=0)[::-1][:1]     # Top 1 
        running_time = time.time() - t1

        selected_model = Det_Pool[preds[0]]
        print('selected_model: ', selected_model)

    if 'Listwise_Preds_Ens' in args.AutoML_Solution_Pool:
        print('--- Using Listwise_Preds_Ens ---')

        test_data =  TimeSeriesDataset_Demo(data)
        # Load model
        t1 = time.time()
        model = deep_models['resnet'](**model_parameters)
        model.load_state_dict(torch.load(Listwise_preds_model_path))
        model.eval()
        model.to('cuda')
        all_ranks = []
    
        # Timeserie to batches
        test_loader = DataLoader(test_data, batch_size=256, shuffle=False)
        for inputs in test_loader:
            inputs = inputs.to('cuda')
            inputs = torch.unsqueeze(inputs, 1)
            # Make predictions
            outputs = model(inputs.float())     # (N, 23)
            sorted, preds_id = outputs.sort(dim=1)
            _, preds_id = preds_id.sort(dim=1)
            preds_agg = np.sum(preds_id.detach().cpu().numpy(), axis=0)
            all_ranks.extend(preds_agg.tolist())
        preds = np.argsort(np.sum(np.array(all_ranks).reshape(-1,23), axis=0), axis=0)[::-1][:args.ens_top_selection]
        running_time = time.time() - t1
        selected_models = []
        for i in preds:
            selected_models.append(Det_Pool[i])

        print('selected_model: ', selected_models)
