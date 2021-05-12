
def linear(x, weight, bias=None):
    output = x.matmul(weight.t())
    if bias is not None:
        output += bias
    return output

def relu(x):
    return x * (x > 0)

def drelu(x):
    return 1. * (x > 0)

def tanh(x):
    return (x.exp() - x.mul(-1).exp()) / (x.exp() + x.mul(-1).exp())

def dtanh(x):
    return 4 * (x.exp() + x.mul(-1).exp()).pow(-2)

def mse(x, y):
    return (x-y).pow(2).mean()

def dmse(x, y):
    return (x-y).mul(2).div(x.numel())

def softmax(x):
    # Exponential with normalization trick
    e = x.sub(x.max(dim=1)[0].view(-1,1)).exp()
    return e.div(e.sum(axis=1).view(-1,1))

def log_softmax(x):
    return softmax(x).log()

def cross_entropy(x, y):
    return y.mul(log_softmax(x)).mean(axis=0).sum().mul(-1)

def dcross_entropy(x, y):
    return softmax(x).sub(y)
