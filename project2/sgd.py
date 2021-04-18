class SGD():
    def __init__(self, params, lr=1e-1):
        self._parameters = params
        self._lr = lr

    def step(self):
        for p, grad in self._parameters:
            p._add(grad, alpha=-self._lr)