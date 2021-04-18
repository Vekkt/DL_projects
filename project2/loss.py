from module import Module
import functional as F
from torch import set_grad_enabled

class MSELoss(Module):
    def __init__(self, model):
        self._model = model

    def forward(self, input, target):
        self._input = input
        self._target = target
        return F.mse(input, target)

    def backward(self):
        return self._model.backward(F.dmse(self._input, self._target))

    def __call__(self, input, target):
        return self.forward(input, target)
