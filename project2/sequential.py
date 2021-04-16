from module import Module

class Sequential(Module):
    def __init__(self, *args):
        for module in args:
            self._modules.append(module)
            for p in module.param():
                self._parameters.append(p)
    
    def _activation_function(self, input):
        for module in self.modules:
            input = module(input)
        return input

    def _activation_gradient(self, gradwrtoutput):
        for module in reversed(self.modules):
            grad = module._activation_gradient(gradwrtoutput)
        return grad
