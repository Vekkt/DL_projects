from module import Module

class MSELoss(Module):
    def forward(self, input, target):
        self._input = input
        self._target = target
        return 

    def backward(self, input, target):
        return 

    def __call__(self, input, target):
        return self.forward(input, target)