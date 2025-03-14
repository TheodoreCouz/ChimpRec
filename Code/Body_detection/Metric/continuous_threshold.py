import math

def t(init_t, bbox_size):
    weight = 0.4 * math.tanh(5*bbox_size - 1) + 0.6
    return init_t*weight