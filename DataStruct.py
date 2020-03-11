import numpy as np
from Definitions import *
from Util import *


class DataStruct:

    def __init__(self, root):
        # Loading Dataset
        self.train_data = np.loadtxt(root + '/trainData.csv', dtype=np.float32, delimiter=',')
        self.train_labels = np.loadtxt(root + '/trainLabels.csv', dtype=np.int32, delimiter=',')
        self.test_data = np.loadtxt(root + '/testData.csv', dtype=np.float32, delimiter=',')
        self.test_labels = np.loadtxt(root + '/testLabels.csv', dtype=np.int32, delimiter=',')
        self.class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

        self.tr_samples_size, _ = self.train_data.shape
        self.tr_samples_size, self.feature_size = self.train_data.shape
        self.te_samples_size, _ = self.test_data.shape

    def normalize(self):
        # train data mean and std
        tr_std = np.std(self.train_data, axis=0)
        tr_mean = np.mean(self.train_data, axis=0)

        # prevent zero std
        tr_std += 0.00000001

        # preprocessing train data
        self.train_data -= tr_mean
        self.train_data /= tr_std

        # preprocessing test data
        self.test_data -= tr_mean
        self.test_data /= tr_std


class TrainingInfo:

    def __init__(self, train_type):
        if train_type == TR_BGD:
            self.max_epoch = VAL_MAX_BGD_EPOCH_COUNT
            self.opt_epoch = VAL_OPT_BGD_EPOCH_COUNT
            self.epoch_step = VAL_BGD_EPOCH_STEP
            self.batch_size = VAL_BGD_BATCH_SIZE
        else:
            self.max_epoch = VAL_MAX_SGD_EPOCH_COUNT
            self.opt_epoch = VAL_OPT_SGD_EPOCH_COUNT
            self.epoch_step = VAL_SGD_EPOCH_STEP
            self.batch_size = VAL_SGD_BATCH_SIZE


class ExperimentInfo:

    def __init__(self, train_type, test_type,
                 actv_func, loss_func, reg_factor,
                 preprocess, init_type, learning_rate,
                 hidden_size):

        self.train_type = train_type
        self.test_type = parse_input_test_type(test_type)
        self.actv_func = parse_input_actv_func_list(actv_func)
        self.actv_func_str = actv_func
        self.loss_func = parse_input_loss_func(loss_func)
        self.loss_func_str = loss_func
        self.reg_factor = reg_factor
        self.preprocess = preprocess
        self.init_type = init_type
        self.learning_rate = learning_rate
        self.hidden_size = hidden_size

    def print_info(self):
        print('\n')
        print('**********************************************')
        print('Experiment Info')
        print('**********************************************')
        print('expriment type: ', self.test_type)
        print('training type: ', self.train_type)
        print('regularization factor: ', self.reg_factor)
        print('normalization enable: ', self.preprocess)
        print('init type: ', self.init_type)
        print('learning rate: ', self.learning_rate)
        print('hidden size: ', self.hidden_size)
        print('activation funcs: ', self.actv_func_str)
        print('loss funcs: ', self.loss_func_str)
        print('**********************************************')
        print('\n')
