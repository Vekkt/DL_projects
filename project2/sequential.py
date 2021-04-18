from module import Module
''' Sequential module class. Represents a sequence
    of modules executed in series.
'''
class Sequential(Module):
    def __init__(self, *args):
        super(Sequential, self).__init__()
        for module in args:
            self._modules.append(module)
        self.register_parameters()
        self.name = 'sequential'
    
    def forward(self, input):
        for module in self._modules:
            input = module(input)
        return input

    def backward(self, gradwrtoutput):
        grad = gradwrtoutput
        for module in reversed(self._modules):
            grad = module.backward(grad)
        return grad

    def register_parameters(self):
        for module in self._modules:
            for p in module.parameters():
                self._parameters.append(p)
