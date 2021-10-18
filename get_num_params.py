from functools import reduce
from operator import mul
import tensorflow as tf


def get_num_params(vars):
    num_params = 0
    for variable in vars:
        shape = variable.get_shape()
        num_params += reduce(mul, [dim.value for dim in shape], 1)
    return num_params
