from module import Module
import functional as F
from torch import set_grad_enabled

class MSELoss(Module):
    def __init__(self, model):
        self._model = model

    def forward(self, input, target):
        self._input = input
        self._target = target
        self._loss = F.mse(input, target)
        return self

    def backward(self):
        return self._model.backward(F.dmse(self._input, self._target))

    def __call__(self, input, target):
        return self.forward(input, target)

    def __add__(self, other):
        return other + self._loss


class CrossEntropyLoss(Module):
    def __init__(self, model):
        self._model = model

    def forward(self, input, target):
        self._input = input
        self._target = target
        self._loss = F.cross_entropy(input, target)
        return self

    def backward(self):
        return self._model.backward(F.dcross_entropy(self._input, self._target))

    def __call__(self, input, target):
        return self.forward(input, target)

    def __add__(self, other):
        return other + self._loss
