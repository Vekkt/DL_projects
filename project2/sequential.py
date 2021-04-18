from module import Module

class Sequential(Module):
    def __init__(self, *args):
        super(Sequential, self).__init__()
        for module in args:
            self._modules.append(module)
            for p in module.parameters():
                self._parameters.append(p)
    
    def _activation_function(self, input):
        for module in self._modules:
            input = module(input)
        return input

    def _activation_gradient(self, gradwrtoutput):
        for module in reversed(self._modules):
            grad = module._activation_gradient(gradwrtoutput)
        return grad
