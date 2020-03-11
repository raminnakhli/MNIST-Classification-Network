from Definitions import *
from ActivationFunctions import *


def CCR(test_labels, classifier_labels):
    CCR_Val = 0
    for label_idx in range(len(test_labels)):
        if test_labels[label_idx] == classifier_labels[label_idx]:
            CCR_Val += 1

    return 100.0 * CCR_Val / len(test_labels)


def parse_input_test_type(test_type):
    if test_type == 'nc':
        return TT_NEURON_COUNT
    elif test_type == 'reg':
        return TT_REGULARIZATION
    elif test_type == 'norm':
        return TT_NORMALIZATION
    elif test_type == 'init':
        return TT_INITIALIZATION
    elif test_type == 'lr':
        return TT_LEARNING_RATE
    elif test_type == 'custom':
        return TT_CUSTOM
    else:
        return TT_EPOCH_CONVERGENCE


def parse_input_actv_func_list(input_list):
    output_list = list()

    for func in input_list:
        if func == AF_TANH:
            output_list.append(Tanh())
        elif func == AF_LINEAR:
            output_list.append(Linear())
        elif func == AF_LRELU:
            output_list.append(LeakyRelu())
        elif func == AF_RELU:
            output_list.append(Relu())

    return output_list


def parse_input_loss_func(input):
    if input == LF_SOFTMAX:
        return Softmax()
    elif input == LF_SVM:
        return SVM()
    else:
        return None
