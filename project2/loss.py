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

class MSELoss(Loss):
    def __init__(self, model):
        forward_fun = F.mse
        backward_fun = F.dmse
        super(MSELoss, self).__init__(model, forward_fun, backward_fun)

class CrossEntropyLoss(Loss):
    def __init__(self, model):
        forward_fun = F.cross_entropy
        backward_fun = F.dcross_entropy
        super(CrossEntropyLoss, self).__init__(model, forward_fun, backward_fun)