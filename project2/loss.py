from module import Module
import functional as F

class Loss(Module):
    def __init__(self, model, foward_fun, backward_fun):
        self._model = model
        self._forward = foward_fun
        self._backward = backward_fun

    def forward(self, input, target):
        self._input = input
        self._target = target
        self._loss = self._forward(input, target)
        return self

    def backward(self):
        return self._model.backward(self._backward(self._input, self._target))

    def __call__(self, input, target):
        return self.forward(input, target)

    def __add__(self, other):
        return other + self._loss

    def __repr__(self):
        return str(self._loss)

class MSELoss(Loss):
    def __init__(self, model):
        super(MSELoss, self).__init__(
            model, F.mse, F.dmse)

class CrossEntropyLoss(Loss):
    def __init__(self, model):
        super(CrossEntropyLoss, self).__init__(
            model, F.cross_entropy, F.dcross_entropy)
