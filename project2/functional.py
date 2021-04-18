from torch import empty

def linear(input, weight, bias=None):
    output = input.matmul(weight.t())
    if bias is not None:
        output += bias
    return output

def relu(input):
    return input * (input > 0)

def drelu(input):
    return 1. * (input > 0)

def tanh(x):
    return (x.exp() - x.mul(-1).exp()) / (x.exp() + x.mul(-1).exp())

def dtanh(x):
    return 4 * (x.exp() + x.mul(-1).exp()).pow(-2)

def mse(x, y):
    n = x.size()
    if not n:
        n = 1
    else:
        n = n[0]
    return (x-y).pow(2).sum().mul(1./n)

def dmse(x, y):
    n = x.size()
    if not n:
        n = 1
    else:
        n = n[0]
    return (x-y).mul(2./n)