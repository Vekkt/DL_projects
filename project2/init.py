from math import sqrt

def _calculate_fan_in_and_fan_out(tensor):
    num_input_fmaps = tensor.size(1)
    num_output_fmaps = tensor.size(0)
    receptive_field_size = 1
    
    if tensor.dim() > 2:
        receptive_field_size = tensor[0][0].numel()
        
    fan_in = num_input_fmaps * receptive_field_size
    fan_out = num_output_fmaps * receptive_field_size

    return fan_in, fan_out

def xavier_normal_(tensor, gain=1.):
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * sqrt(2.0 / float(fan_in + fan_out))

    return tensor.normal_(0., std)
