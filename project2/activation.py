from module import Module
import functional as F

class ReLU(Module):
    def __init__(self, name=''):
        super(ReLU, self).__init__()
        self.name = name
        
    def _activation_function(self, x):
        return F.relu(x)
        
    def _activation_gradient(self, x):
        return F.drelu(x)


class Tanh(Module):
    def _activation_function(self, x):
        return F.tanh(x)

    def _activation_gradient(self, x):
        return F.dtanh(x)
