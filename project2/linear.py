from module import Module
from torch import empty
import functional as F
import init
import math

class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = empty(out_features, in_features)
        self.weight_grad = empty(out_features, in_features)

        if bias:
            self.bias = empty(out_features)
            self.bias_grad = empty(out_features)
            self._parameters.append(self.bias)
            
        self.init_parameters()
        self.register_parameters()

    def init_parameters(self): 
        init.xavier_normal_(self.weight)
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            self.bias.uniform_(-bound, bound)

    def _activation_function(self, input):
        return F.linear(input, self.weight, self.bias)

    def _activation_gradient(self, gradwrtoutput):
        if self.bias is not None:
            self.bias_grad = gradwrtoutput.sum(0)
        self.weight_grad = gradwrtoutput.t().mm(self._input)
        return gradwrtoutput.mm(self.weight)

    def register_parameters(self):
        if self.bias is not None:
            self._parameters = [
                (self.weight, self.weight_grad),
                (self.bias, self.bias_grad)
            ]
        else:
            self._parameters = [(self.weight, self.weight_grad)]



    

