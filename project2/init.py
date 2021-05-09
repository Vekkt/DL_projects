from math import sqrt

def xavier_normal_(tensor, gain=1.):
    if tensor.dim() > 2:
        raise ValueError("Unsupported tensor format")
        
    n_in = tensor.size(1)
    n_out = tensor.size(0)
    std = gain * sqrt(6.0 / float(n_in + n_out))

    return tensor.normal_(0., std)
