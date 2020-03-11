from layer import *
from ActivationFunctions import *
import Util


class NeuralNetwork:

    def __init__(self, layers_count, layers_neuron_count, layers_active_funcs,
                 layers_init_type, regularization_factor, loss_function):

        self.layer_neuron_count = layers_neuron_count
        self.layers = list()
        self.loss_func = loss_function
        self.reg_factor = regularization_factor

        for layer_index in range(layers_count):
            input_neuron_count = layers_neuron_count[layer_index]
            output_neuron_count = layers_neuron_count[layer_index + 1]

            self.layers.append(Layer(feature_size=input_neuron_count,
                                     layer_size=output_neuron_count,
                                     layer_init_type=layers_init_type,
                                     layer_active_func=layers_active_funcs[layer_index]))

    def train(self, train_type, data, label, epoch_count, batch_size, rou):

        if train_type == 'bgd':
            return self.bgd_train(data, label, epoch_count, batch_size, rou)
        else:
            return self.sgd_train(data, label, epoch_count, batch_size, rou)

    def bgd_train(self, data, label, epoch_count, batch_size, rou):

        for epoch in range(epoch_count):

            tr_loss = 0
            for batch in range(len(label) // batch_size):

                start_idx = batch * batch_size
                end_idx = (batch + 1) * batch_size

                reg = 0

                # forward
                forward_data = np.copy(data[start_idx:end_idx, :])
                forward_label = np.copy(label[start_idx:end_idx])

                for layer_idx in range(len(self.layers)):
                    c_layer = self.layers[layer_idx]
                    reg += self.reg_factor * np.sum(c_layer.weight)
                    forward_data = c_layer.forward(forward_data)

                # loss function
                loss = self.loss_func.forward(forward_data, forward_label, reg)

                # backward
                backward_data = self.loss_func.backward()

                for layer_idx in range(len(self.layers) - 1, -1, -1):
                    c_layer = self.layers[layer_idx]
                    backward_data, dw, db = c_layer.backward(backward_data)

                    # regularization L2
                    dw += 2 * self.reg_factor * c_layer.weight
                    c_layer.update(dw, db, rou)

                tr_loss += loss

        return tr_loss / (len(label) // batch_size)

    def sgd_train(self, data, label, epoch_count, batch_size, rou):

        for epoch in range(epoch_count):

            reg = 0
            rand_idx = np.random.randint(low=0, high=data.shape[0], size=batch_size)

            # forward
            forward_data = np.copy(data[rand_idx, :])
            forward_label = np.copy(label[rand_idx])

            for layer_idx in range(len(self.layers)):
                c_layer = self.layers[layer_idx]
                reg += self.reg_factor * np.sum(c_layer.weight)
                forward_data = c_layer.forward(forward_data)

            # loss function
            loss = self.loss_func.forward(forward_data, forward_label, reg)

            # backward
            backward_data = self.loss_func.backward()

            for layer_idx in range(len(self.layers) - 1, -1, -1):
                c_layer = self.layers[layer_idx]
                backward_data, dw, db = c_layer.backward(backward_data)

                # regularization L2
                dw += 2 * self.reg_factor * c_layer.weight
                c_layer.update(dw, db, rou)

        return loss

    def predict(self, data):

        # forward
        forward_data = np.copy(data)

        for layer_idx in range(len(self.layers)):
            forward_data = self.layers[layer_idx].forward(forward_data)

        predicted_label = np.argmax(forward_data, axis=1)
        return predicted_label

    def test(self, data, label):

        # forward
        forward_data = np.copy(data)

        for layer_idx in range(len(self.layers)):
            forward_data = self.layers[layer_idx].forward(forward_data)

        predicted_label = np.argmax(forward_data, axis=1)
        acc = Util.CCR(label, predicted_label)
        loss = self.loss_func.forward(forward_data, label, reg=0)

        return acc, loss
