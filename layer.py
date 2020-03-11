from Definitions import *
from Gates import *


class Layer:

    def __init__(self, feature_size, layer_size, layer_init_type, layer_active_func):
        # Sizes
        self.feature_size = feature_size
        self.layer_size = layer_size
        self.weight = np.zeros(shape=(feature_size, layer_size))
        self.bias = np.zeros(shape=layer_size)
        self.active_func = layer_active_func

        # Gates
        self.multiplyGate = MultiplyGate()
        self.summationGate = SummationGate()

        # Initialization
        self.initialize_weight(layer_init_type)

    def initialize_weight(self, layer_init_type):
        fan_in = self.feature_size
        fan_out = self.layer_size

        if layer_init_type == IT_XAVIER:
            coef = 1 / np.sqrt(fan_in)
        else:
            coef = 0.01

        self.weight = np.random.randn(fan_in, fan_out) * coef
        self.bias = np.random.randn(self.layer_size)

    def forward(self, x):
        mult_res = self.multiplyGate.forward(x, self.weight)
        sum_res = self.summationGate.forward(mult_res, self.bias)
        result = self.active_func.forward(sum_res)
        return result

    def backward(self, dz):
        result = self.active_func.backward(dz)
        db = np.sum(result, axis=0)
        dx, dw = self.multiplyGate.backward(result)
        return [dx, dw, db]

    def update(self, dw, db, rou):
        self.bias -= rou * db
        self.weight -= rou * dw
