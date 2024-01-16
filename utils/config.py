########################################################################
#
# @author : Emmanouil Sylligardos
# @when : Winter Semester 2022/2023
# @where : LIPADE internship Paris
# @title : MSAD (Model Selection Anomaly Detection)
# @component: utils
# @file : config
#
########################################################################

from models.model.convnet import ConvNet
from models.model.inception_time import InceptionModel
from models.model.resnet import ResNetBaseline
from models.model.sit import SignalTransformer


# Important paths
TSB_data_path = "/data/liuqinghua/code/ts/TSB-UAD/data/public/"
TSB_metrics_path = "data/metrics/"
TSB_scores_path = "data/TSB/scores/"

save_done_training = 'results/done_training/'	# when a model is done training a csv with training info is saved here
save_done_training_ranking = 'results/done_training_ranking/'
path_save_results = 'results/pointwise_predictions'	# when evaluating a model, the predictions will be saved here
path_save_results_ranking = 'results/listwise_predictions'

# Detector
detector_names = ['IForest_3_200', 'IForest_1_100', 'IForest_0_200', 'LOF_3_60', 'LOF_1_30', 'MP_2_False', 'MP_1_True', 'PCA_3_None', 'PCA_1_0.5',
            'NORMA_1_hierarchical', 'NORMA_3_kshape', 'HBOS_3_20', 'HBOS_1_40', 'POLY_3_5', 'POLY_2_1', 'OCSVM_1_rbf', 'OCSVM_3_poly',
            'AE_1_1_bn', 'AE_2_0_dropout', 'CNN_2_0_relu', 'CNN_3_1_sigmoid', 'LSTM_1_1_relu', 'LSTM_3_1_sigmoid']


# Dict of model names to Constructors
deep_models = {
    'convnet':ConvNet,
    'inception_time':InceptionModel,
    'inception':InceptionModel,
    'resnet':ResNetBaseline,
    'sit':SignalTransformer,
}
