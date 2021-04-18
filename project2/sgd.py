class SGD():
    def __init__(self, model, lr=1e-1):
        self._model = model
        self._lr = lr

    def step(self):
        for p, grad in self._model.parameters():
            p.add_(grad, alpha=-self._lr)
