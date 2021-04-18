from module import Module
import functional as F

class ReLU(Module):
    def __init__(self, name='relu'):
        super(ReLU, self).__init__()
        self.name = name
        
    def _activation_function(self, x):
        return F.relu(x)
        
    def _activation_gradient(self, x):
        return F.drelu(x)


class Tanh(Module):
    def __init__(self, name='tanh'):
        super(Tanh, self).__init__()
        self.name = name
        
    def _activation_function(self, x):
        return F.tanh(x)

    def _activation_gradient(self, x):
        return F.dtanh(x)
