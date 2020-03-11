from network import NeuralNetwork
from Definitions import *
from plot_conf_mat import *
from DataStruct import TrainingInfo


def epoch_convergence(train_type, data_set, hidden_layers_neuron,
                      layers_active_funcs, loss_function, reg, lr, init_type):
    print('>> Running epoch count convergence\n')

    # calculate fan in and fan out
    fan_in = data_set.train_data.shape[1]
    fan_out = len(data_set.class_names)

    # loss lists
    test_loss_array = list()
    train_loss_array = list()
    test_acc_array = list()

    # specify train info
    t_info = TrainingInfo(train_type)

    # calculate the epoch range
    epoch_range = range(1, t_info.max_epoch, t_info.epoch_step)

    # define the network
    network = NeuralNetwork(layers_count=2,
                            layers_neuron_count=[fan_in, hidden_layers_neuron, fan_out],
                            layers_active_funcs=layers_active_funcs,
                            layers_init_type=init_type,
                            loss_function=loss_function,
                            regularization_factor=reg)

    # do training for different epochs
    for epoch in epoch_range:
        # train network
        tr_loss = network.train(train_type, data_set.train_data, data_set.train_labels,
                                epoch, t_info.batch_size, lr)

        # test network
        ccr, loss = network.test(data_set.test_data, data_set.test_labels)

        # store losses
        test_loss_array.append(loss)
        train_loss_array.append(tr_loss)
        test_acc_array.append(ccr)
        print('train loss : {}, test loss : {}, test acc : {}'.format(tr_loss, loss, ccr))

    plt.figure()

    # plot loss
    plt.subplot(211)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.plot(epoch_range, train_loss_array, label='train', color='red')
    plt.plot(epoch_range, test_loss_array, label='test', color='blue')
    plt.legend()

    # Plot Acc
    plt.subplot(212)
    plt.xlabel('epoch')
    plt.ylabel('acc')
    plt.plot(epoch_range, test_acc_array, label='test acc', color='blue')
    plt.legend()

    plt.show()


def neuron_count(train_type, data_set, layers_active_funcs, loss_function, reg, lr, init_type):
    print('>> Running neuron count test\n')

    # calculate fan in and fan out
    fan_in = data_set.train_data.shape[1]
    fan_out = len(data_set.class_names)

    # loss lists
    test_loss_array = list()
    train_loss_array = list()
    test_acc_array = list()

    # specify min and max and step for neuron count
    layers_min_neuron = 9
    layers_max_neuron = 50
    step = 3

    # specify train info
    t_info = TrainingInfo(train_type)

    # specify different neuron count to test
    neuron_range = range(layers_min_neuron, layers_max_neuron, step)

    # define network for each neuron count
    for neuron in neuron_range:
        network = NeuralNetwork(layers_count=2,
                                layers_neuron_count=[fan_in, neuron, fan_out],
                                layers_active_funcs=layers_active_funcs,
                                layers_init_type=init_type,
                                loss_function=loss_function,
                                regularization_factor=reg)

        # train network
        tr_loss = network.train(train_type, data_set.train_data, data_set.train_labels,
                                t_info.opt_epoch, t_info.batch_size, lr)

        # test network
        ccr, loss = network.test(data_set.test_data, data_set.test_labels)

        # store losses
        test_loss_array.append(loss)
        test_acc_array.append(ccr)
        train_loss_array.append(tr_loss)

        print('train loss : {}, test loss : {}, test acc : {}'.format(tr_loss, loss, ccr))

    plt.figure()

    # plot Loss
    plt.subplot(211)
    plt.xlabel('hidden neuron')
    plt.ylabel('loss')
    plt.plot(neuron_range, train_loss_array, label='train', color='blue')
    plt.plot(neuron_range, test_loss_array, label='test', color='red')
    plt.legend()

    # plot accuracy
    plt.subplot(212)
    plt.xlabel('hidden neuron')
    plt.ylabel('test acc')
    plt.plot(neuron_range, test_acc_array, label='test', color='red')
    plt.legend()

    plt.show()


def regularization(train_type, data_set, hidden_layers_neuron, layers_active_funcs,
                   loss_function, lr, init_type):
    print('>> Running regularization test\n')

    # calculate fan in and fan out
    fan_in = data_set.train_data.shape[1]
    fan_out = len(data_set.class_names)

    # regularization factors to be tested
    regularization_range = [0, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1]

    # loss lists
    test_loss_array = list()
    train_loss_array = list()
    test_acc_array = list()

    # specify train info
    t_info = TrainingInfo(train_type)

    # do regularization test for each factor
    for reg in regularization_range:

        # define network
        network = NeuralNetwork(layers_count=2,
                                layers_neuron_count=[fan_in, hidden_layers_neuron, fan_out],
                                layers_active_funcs=layers_active_funcs,
                                layers_init_type=init_type,
                                loss_function=loss_function,
                                regularization_factor=reg)

        # train network
        tr_loss = network.train(train_type, data_set.train_data, data_set.train_labels,
                                t_info.opt_epoch, t_info.batch_size, lr)

        # test info
        ccr, loss = network.test(data_set.test_data, data_set.test_labels)

        # store losses
        test_loss_array.append(loss)
        test_acc_array.append(ccr)
        train_loss_array.append(tr_loss)
        print('train loss : {}, test loss : {}, test acc : {}'.format(tr_loss, loss, ccr))

    plt.figure()

    # plot loss
    plt.subplot(211)
    plt.xlabel('regularization factor')
    plt.ylabel('loss')
    plt.plot(regularization_range, train_loss_array, label='train', color='red')
    plt.plot(regularization_range, test_loss_array, label='test', color='blue')
    plt.legend()

    # plot accuracy
    plt.subplot(212)
    plt.xlabel('regularization factor')
    plt.ylabel('acc')
    plt.plot(regularization_range, test_acc_array, label='test acc', color='blue')
    plt.legend()

    plt.show()


def normalization(train_type, data_set, hidden_layers_neuron, layers_active_funcs,
                  loss_function, reg, lr, init_type):
    print('>> Running normalization test\n')

    # calculate fan in and fan out
    fan_in = data_set.train_data.shape[1]
    fan_out = len(data_set.class_names)

    plt.figure()

    # set color for plots

    # specify train info
    t_info = TrainingInfo(train_type)

    # specify epoch range
    epoch_range = range(1, t_info.max_epoch, t_info.epoch_step)

    # do the test for normalization and no normalization
    for norm_en in [False, True]:

        if norm_en:
            label = 'normalized'
            data_set.normalize()
            color = ['red', 'green']
        else:
            label = ''
            color = ['blue', 'yellow']

        # loss lists
        test_loss_array = list()
        train_loss_array = list()
        test_acc_array = list()

        # define the network
        network = NeuralNetwork(layers_count=2,
                                layers_neuron_count=[fan_in, hidden_layers_neuron, fan_out],
                                layers_active_funcs=layers_active_funcs,
                                layers_init_type=init_type,
                                loss_function=loss_function,
                                regularization_factor=reg)

        # try for some epochs
        for epoch in epoch_range:
            # train network
            tr_loss = network.train(train_type, data_set.train_data, data_set.train_labels,
                                    epoch, t_info.batch_size, lr)

            # test network
            ccr, loss = network.test(data_set.test_data, data_set.test_labels)

            # store loss
            test_loss_array.append(loss)
            test_acc_array.append(ccr)
            train_loss_array.append(tr_loss)
            print('train loss : {}, test loss : {}, test acc : {}'.format(tr_loss, loss, ccr))

        # plot loss
        plt.subplot(211)
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.plot(epoch_range, train_loss_array, label='train {}'.format(label), color=color[0])
        plt.plot(epoch_range, test_loss_array, label='test {}'.format(label), color=color[1])
        plt.legend()

        # plot accuracy
        plt.subplot(212)
        plt.xlabel('epochs')
        plt.ylabel('acc')
        plt.plot(epoch_range, test_acc_array, label='test acc {}'.format(label), color=color[0])
        plt.legend()

    plt.show()


def initialization(train_type, data_set, hidden_layers_neuron, layers_active_funcs,
                   loss_function, reg, lr):
    print('>> Running initialization test\n')

    # calculate fan in and fan out
    fan_in = data_set.train_data.shape[1]
    fan_out = len(data_set.class_names)

    # specify test info
    t_info = TrainingInfo(train_type)

    # specify epoch range
    epoch_range = range(1, t_info.max_epoch, t_info.epoch_step)

    plt.figure()

    # try test for two different init type
    for init_type in [IT_XAVIER, IT_RANDOM]:

        # loss lists
        test_loss_array = list()
        train_loss_array = list()
        test_acc_array = list()

        # specify color for plots
        if init_type == IT_RANDOM:
            color = ['red', 'green']
        else:
            color = ['blue', 'yellow']

        # define network
        network = NeuralNetwork(layers_count=2,
                                layers_neuron_count=[fan_in, hidden_layers_neuron, fan_out],
                                layers_active_funcs=layers_active_funcs,
                                layers_init_type=init_type,
                                loss_function=loss_function,
                                regularization_factor=reg)

        # try test for some epochs
        for epoch in epoch_range:
            # train network
            tr_loss = network.train(train_type, data_set.train_data, data_set.train_labels,
                                    epoch, t_info.batch_size, lr)

            # test network
            ccr, loss = network.test(data_set.test_data, data_set.test_labels)

            # store loss
            test_loss_array.append(loss)
            test_acc_array.append(ccr)
            train_loss_array.append(tr_loss)
            print('train loss : {}, test loss : {}, test acc : {}'.format(tr_loss, loss, ccr))

        # plot loss
        plt.subplot(211)
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.plot(epoch_range, train_loss_array, label='train, {}'.format(init_type), color=color[0])
        plt.plot(epoch_range, test_loss_array, label='test, {}'.format(init_type), color=color[1])
        plt.legend()

        # plot accuracy
        plt.subplot(212)
        plt.xlabel('epochs')
        plt.ylabel('acc')
        plt.plot(epoch_range, test_acc_array, label='test acc, {}'.format(init_type), color=color[0])
        plt.legend()

    plt.show()


def learning_rate_test(train_type, data_set, hidden_layers_neuron, layers_active_funcs,
                       loss_function, reg, init_type):
    print('>> Running learning rate test\n')

    # calculate fan in and fan out
    fan_in = data_set.train_data.shape[1]
    fan_out = len(data_set.class_names)

    plt.figure()

    # specify train info
    t_info = TrainingInfo(train_type)

    # specify epoch range
    epoch_range = range(1, t_info.max_epoch, t_info.epoch_step)

    # set color array for plots
    color = ['red', 'blue', 'green', 'yellow', 'cyan', 'black', 'orange', 'grey']

    # learning rates to be tested
    learning_rate_list = [0.0001, 0.001, 0.01, 0.02]

    # test each one of the learning rates
    for idx, learning_rate in enumerate(learning_rate_list):

        # loss lists
        test_loss_array = list()
        train_loss_array = list()
        test_acc_array = list()

        # define network
        network = NeuralNetwork(layers_count=2,
                                layers_neuron_count=[fan_in, hidden_layers_neuron, fan_out],
                                layers_active_funcs=layers_active_funcs,
                                layers_init_type=init_type,
                                loss_function=loss_function,
                                regularization_factor=reg)

        # try for some epochs
        for epoch in epoch_range:
            # train network
            tr_loss = network.train(train_type, data_set.train_data, data_set.train_labels,
                                    epoch, t_info.batch_size, learning_rate)

            # test network
            ccr, loss = network.test(data_set.test_data, data_set.test_labels)

            # store loss
            test_loss_array.append(loss)
            test_acc_array.append(ccr)
            train_loss_array.append(tr_loss)
            print('train loss : {}, test loss : {}, test acc : {}'.format(tr_loss, loss, ccr))

        # plot loss
        plt.subplot(211)
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.plot(epoch_range, train_loss_array, label='train, {}'.format(learning_rate), color=color[2 * idx])
        plt.plot(epoch_range, test_loss_array, label='test, {}'.format(learning_rate), color=color[2 * idx + 1])
        plt.legend()

        # plot accuracy
        plt.subplot(212)
        plt.xlabel('epochs')
        plt.ylabel('acc')
        plt.plot(epoch_range, test_acc_array, label='test acc, {}'.format(learning_rate), color=color[idx])
        plt.legend()

    plt.show()


def performance_test(train_type, data_set, hidden_layers_neuron,
                     layers_active_funcs, loss_function, reg, lr, init_type):
    print('>> Running custom test\n')

    # specify fan in and fan out
    fan_in = data_set.train_data.shape[1]
    fan_out = len(data_set.class_names)

    # loss lists
    test_loss_array = list()
    train_loss_array = list()
    test_acc_array = list()

    # specify train info
    t_info = TrainingInfo(train_type)

    # specify epoch range
    epoch_range = range(1, t_info.opt_epoch, t_info.epoch_step)

    # define network
    network = NeuralNetwork(layers_count=2,
                            layers_neuron_count=[fan_in, hidden_layers_neuron, fan_out],
                            layers_active_funcs=layers_active_funcs,
                            layers_init_type=init_type,
                            loss_function=loss_function,
                            regularization_factor=reg)

    # try some epochs
    for epoch in epoch_range:

        # train network
        tr_loss = network.train(train_type, data_set.train_data, data_set.train_labels,
                                epoch, t_info.batch_size, lr)

        # test network
        ccr, loss = network.test(data_set.test_data, data_set.test_labels)

        # store loss
        test_loss_array.append(loss)
        train_loss_array.append(tr_loss)
        test_acc_array.append(ccr)
        print('train loss : {}, test loss : {}, test acc : {}'.format(tr_loss, loss, ccr))

    # predict labels
    predicted_labels = network.predict(data_set.test_data)

    # plot confusion matrix
    plot_confusion_matrix(data_set.test_labels, predicted_labels, data_set.class_names,
                          title='Confusion matrix')

    plt.figure()

    # plot loss
    plt.subplot(211)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.plot(epoch_range, train_loss_array, label='train', color='red')
    plt.plot(epoch_range, test_loss_array, label='test', color='blue')
    plt.legend()

    # plot accuracy
    plt.subplot(212)
    plt.xlabel('epoch')
    plt.ylabel('acc')
    plt.plot(epoch_range, test_acc_array, label='test acc', color='blue')
    plt.legend()

    plt.show()
