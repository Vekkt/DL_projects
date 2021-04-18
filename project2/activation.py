from module import Module
import functional as F

class ReLU(Module):
    def __init__(self, name='relu'):
        super(ReLU, self).__init__()
        self.name = name
        
    def forward(self, input):
        return F.relu(input)
        
    def backward(self, gradwrtoutput):
        return F.drelu(gradwrtoutput)


class Tanh(Module):
    def __init__(self, name='tanh'):
        super(Tanh, self).__init__()
        self.name = name
        
    def forward(self, input):
        return F.tanh(input)

    def backward(self, gradwrtoutput):
        return F.dtanh(gradwrtoutput)
