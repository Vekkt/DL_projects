from torch import empty

def linear(input, weight, bias=None):
    output = input.matmul(weight.t())
    if bias is not None:
        output += bias
    return output
