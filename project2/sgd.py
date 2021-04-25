from torch import empty

''' Stochastic gradient descent algorithm representative 
    class. Given a model and a learning rate, it allows
    to update the parameters of the model according to
    the gradient values for each parameter.
'''
class SGD():
    def __init__(self, params, lr=1e-1, momentum=0):
        assert(lr>0)
        self._parameters = params
        self.lr = lr
        self.momentum = momentum
        self.momentum_buffer = []
        for _, grad in self._parameters:
            self.momentum_buffer.append(empty(grad.size()).zero_())

    ''' Makes one step for the gradient descent algorithm
        For each parameter, update its value according to 
        its gradient. 
        The gradient has been computed during the backward
        pass.
        Every operation is executed in place, so that the
        values update everywhere.
    '''
    def step(self):
        for (p, grad), m in zip(self._parameters, self.momentum_buffer):
            m.mul_(self.momentum).add_(grad, alpha=self.lr)
            p.add_(m, alpha=-1.)
