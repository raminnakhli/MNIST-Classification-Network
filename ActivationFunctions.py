from Gates import *


######################################################################
# Activation Functions
######################################################################
class Tanh:

    def forward(self, x):
        self.x = x
        return np.tanh(x)

    def backward(self, dz):
        return dz * (1 - np.tanh(self.x) ** 2)


class LeakyRelu:

    def __init__(self):
        self.x = 0

    def forward(self, x):
        self.x = x
        return np.maximum(0.1 * x, x)

    def backward(self, dz):
        comp_mat = self.x > 0.1 * self.x
        comp_mat = comp_mat.astype(int) * 0.9
        div_mat = comp_mat + 0.1 * np.ones(shape=self.x.shape)
        return div_mat * dz


class Relu:

    def __init__(self):
        self.x = 0

    def forward(self, x):
        self.x = x
        return np.maximum(x, 0)

    def backward(self, dz):
        max_mat = self.x > 0
        max_mat.astype(int)
        return max_mat * dz


class Linear:

    def forward(self, x):
        return x

    def backward(self, dz):
        return dz


######################################################################
# Loss Functions
######################################################################
class Softmax:

    def __init__(self):
        self.x = 0
        self.label = 0

    def forward(self, x, label, reg):
        self.x = x
        self.label = np.copy(label)

        max_mat = np.max(x, axis=1).reshape(-1, 1)
        norm_mat = (x - max_mat)
        loss_mat = np.exp(norm_mat) / np.sum(np.exp(norm_mat), axis=1, keepdims=True) + 1e-8
        loss = np.sum(-np.log(loss_mat[range(x.shape[0]), label]))

        return loss / x.shape[0]

    def backward(self):
        mask = np.zeros(shape=self.x.shape)
        mask[range(self.x.shape[0]), self.label] = -1

        max_mat = np.max(self.x, axis=1).reshape(-1, 1)
        norm_mat = np.exp(self.x - max_mat)

        sum_exp = np.sum(norm_mat, axis=1)
        return mask + norm_mat / np.repeat(sum_exp.reshape((-1, 1)), self.x.shape[1], axis=1)


class SVM:

    def __init__(self):
        self.label = 0
        self.diff_mat = 0

    def forward(self, x, label, reg):
        self.label = np.copy(label)
        true_label = np.repeat(x[range(x.shape[0]), label].reshape(-1, 1), x.shape[1], axis=1)
        self.diff_mat = x - true_label + np.ones(shape=x.shape)
        max_mat = np.maximum(self.diff_mat, 0)
        max_mat[range(x.shape[0]), label] = 0
        return np.sum(max_mat) / x.shape[0] + reg

    def backward(self):
        init_mat = np.ones(shape=self.diff_mat.shape)
        init_mat[range(self.diff_mat.shape[0]), self.label] = 0
        max_mat = self.diff_mat > 0
        max_mat.astype(int)
        return max_mat * init_mat
######################################################################
