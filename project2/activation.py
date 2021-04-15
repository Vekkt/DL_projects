from module import Module

class ReLU(Module):
    def _activation_function(self, x):
        return x[x > 0]

    def _activation_gradient(self, x):
        return (x > 0).float()


class Tanh(Module):
    def _activation_function(self, x):
        return (x.exp() - x.mul(-1).exp()) / (x.exp() + x.mul(-1).exp())

    def _activation_gradient(self, x):
        return 4 * (x.exp() + x.mul(-1).exp()).pow(-2)
