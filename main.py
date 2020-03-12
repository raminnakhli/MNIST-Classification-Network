from ActivationFunctions import *
import Expriments
from DataStruct import DataStruct
from DataStruct import ExperimentInfo
from Util import *
import argparse


# parse input argument
parser = argparse.ArgumentParser(description='This is a script used to run the tests')
parser.add_argument('-dataset', '--dataset-path', default='MNIST', help='path on dataset relative to this file')
parser.add_argument('-tr', '--train-type', default=TR_BGD, help='bgd/sgd')
parser.add_argument('-ex', '--test-type', default='con', help='con/nc/reg/norm/init/lr/custom')
parser.add_argument('-rf', '--regularization-factor', default=0, help='number')
parser.add_argument('-lr', '--learning-rate', default=0.001, help='number')
parser.add_argument('-hs', '--hidden-size', default=20, help='number')
parser.add_argument('-ne', '--normalization-enable', action='store_true', help='normalization enable')
parser.add_argument('-it', '--init-type', default=IT_RANDOM, help='random/xavier')
parser.add_argument('-lf', '--loss-function', default=LF_SOFTMAX, help='softmax/svm')
parser.add_argument('-af', '--activation-function', nargs='+', help='tanh/relu/lrelu/linear', required = True)
args = parser.parse_args()

# load data set
original_data = DataStruct(args.dataset_path)

# check activation function list len
if not len(args.activation_function) == 1:
    print('you should enter just one activation function')
    exit()
else:
    actv_func_list = args.activation_function
    actv_func_list.append(AF_LINEAR)

# set parameters value
expriment_info = ExperimentInfo(test_type= args.test_type,
                                actv_func=actv_func_list,
                                loss_func=args.loss_function,
                                train_type= args.train_type,
                                reg_factor=float(args.regularization_factor),
                                preprocess=args.normalization_enable,
                                init_type=args.init_type,
                                learning_rate=float(args.learning_rate),
                                hidden_size=int(args.hidden_size))

# print parameters value
expriment_info.print_info()

# normalize data if needed
if expriment_info.preprocess:
    original_data.normalize()

# run the test
if expriment_info.test_type == TT_EPOCH_CONVERGENCE:

    Expriments.epoch_convergence(expriment_info.train_type, original_data, expriment_info.hidden_size,
                                 expriment_info.actv_func, expriment_info.loss_func, expriment_info.reg_factor,
                                 expriment_info.learning_rate, expriment_info.init_type)

elif expriment_info.test_type == TT_NEURON_COUNT:

    Expriments.neuron_count(expriment_info.train_type, original_data, expriment_info.actv_func,
                            expriment_info.loss_func, expriment_info.reg_factor, expriment_info.learning_rate,
                            expriment_info.init_type)

elif expriment_info.test_type == TT_REGULARIZATION:

    Expriments.regularization(expriment_info.train_type, original_data, expriment_info.hidden_size,
                              expriment_info.actv_func, expriment_info.loss_func, expriment_info.learning_rate,
                              expriment_info.init_type)

elif expriment_info.test_type == TT_NORMALIZATION:

    Expriments.normalization(expriment_info.train_type, original_data, expriment_info.hidden_size,
                             expriment_info.actv_func, expriment_info.loss_func, expriment_info.reg_factor,
                             expriment_info.learning_rate, expriment_info.init_type)

elif expriment_info.test_type == TT_INITIALIZATION:

    Expriments.initialization(expriment_info.train_type, original_data, expriment_info.hidden_size,
                              expriment_info.actv_func, expriment_info.loss_func, expriment_info.reg_factor,
                              expriment_info.learning_rate)

elif expriment_info.test_type == TT_LEARNING_RATE:

    Expriments.learning_rate_test(expriment_info.train_type, original_data, expriment_info.hidden_size,
                                  expriment_info.actv_func, expriment_info.loss_func, expriment_info.reg_factor,
                                  expriment_info.init_type)

elif expriment_info.test_type == TT_CUSTOM:

    Expriments.performance_test(expriment_info.train_type, original_data, expriment_info.hidden_size,
                                expriment_info.actv_func, expriment_info.loss_func, expriment_info.reg_factor,
                                expriment_info.learning_rate, expriment_info.init_type)
