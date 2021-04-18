''' Stochastic gradient descent algorithm representative 
    class. Given a model and a learning rate, it allows
    to update the parameters of the model according to
    the gradient values for each parameter.
'''
class SGD():
    def __init__(self, model, lr=1e-1):
        assert(lr>0)
        self._model = model
        self._lr = lr

    ''' Makes one step for the gradient descent algorithm
        For each parameter, update its value according to 
        its gradient. 
        The gradient has been computed during the backward
        pass.
        Every operation is executed in place, so that the
        values update everywhere.
    '''
    def step(self):
        for p, grad in self._model.parameters():
            p.add_(grad, alpha=-self._lr)
