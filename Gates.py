import numpy as np


class MultiplyGate:

    def __init__(self):
        self.y = 0
        self.x = 0

    def forward(self, x, y):
        self.y = np.copy(y)
        self.x = np.copy(x)

        return x.dot(y)

    def backward(self, dz):
        dx = dz.dot(self.y.transpose())
        dy = self.x.transpose().dot(dz)

        return [dx, dy]


class SummationGate:

    def forward(self, x, y):
        return x + y

    def backward(self, dz):
        return [np.copy(dz), np.copy(dz)]


# class MaximizationGate:
#
#     def forward(self, x, y):
#         self.x = x
#         self.y = y
#
#         return np.copy(max(x, y))
#
#     def backward(self, dz):
#         if self.x > self.y:
#             dx = dz
#             dy = 0
#         else:
#             dx = 0
#             dy = dz
#
#         return [dx, dy]


class ExponentialGate:

    def forward(self, x):
        self.x = np.copy(x)
        return np.exp(x)

    def backward(self, dz):
        return np.exp(self.x).dot(dz)

# class LogarithmGate:
#
#     def forward(self, x):
#         self.x = np.copy(x)
#         return np.log(x)
#
#     def backward(self, dz):
#

class ConstantGate:

    def __init__(self, constant):
        self.constant = constant

    def forward(self, x):
        return self.constant * x

    def backward(self, dz):
        return np.copy(dz) if self.constant > 0 else -1 * np.copy(dz)


# class PolynomialGate:
#
#     def __init__(self, power):
#         self.power = power
#
#     def forward(self, x):
#         self.x = x
#         return x ** self.power
#
#     def backward(self, dz):
#         dd = self.power * (self.x ** (self.power - 1))
#         return dd * dz