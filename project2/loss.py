from module import Module
import functional as F

class MSELoss(Module):
    def forward(self, input, target):
        self._input = input
        self._target = target
        return F.mse(input, target)

    def backward(self, input, target):
        return F.dmse(input, target)

    def __call__(self, input, target):
        return self.forward(input, target)
