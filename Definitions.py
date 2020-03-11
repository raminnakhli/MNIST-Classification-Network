######################################################################
# test types
######################################################################
TT_EPOCH_CONVERGENCE = 'epoch_convergence'
TT_NEURON_COUNT = 'neuron_count'
TT_REGULARIZATION = 'regularization'
TT_NORMALIZATION = 'normalization'
TT_INITIALIZATION = 'initialization'
TT_LEARNING_RATE = 'learning_rate'
TT_CUSTOM = 'custom_test'

######################################################################
# activation functions
######################################################################
AF_TANH = 'tanh'
AF_RELU = 'relu'
AF_LRELU = 'lrelu'
AF_LINEAR = 'linear'

######################################################################
# loss functions
######################################################################
LF_SOFTMAX = 'softmax'
LF_SVM = 'svm'

######################################################################
# init types
######################################################################
IT_XAVIER = 'xavier'
IT_RANDOM = 'random'

######################################################################
# train type
######################################################################
TR_BGD = 'bgd'
TR_SGD = 'sgd'

######################################################################
# variables
######################################################################
# batch size
VAL_BGD_BATCH_SIZE = 100
VAL_SGD_BATCH_SIZE = 100

# learning rate
VAL_LEARNING_RATE = 0.001

# max epoch count
VAL_MAX_BGD_EPOCH_COUNT = 20
VAL_MAX_SGD_EPOCH_COUNT = 3000

# epoch step
VAL_BGD_EPOCH_STEP = 2
VAL_SGD_EPOCH_STEP = 100

# optimal epoch count
VAL_OPT_BGD_EPOCH_COUNT = 7
VAL_OPT_SGD_EPOCH_COUNT = 2000
