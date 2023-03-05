import numpy as np


def fin():
    print("I have been called")

def shrink(epsilon, x):
    """
    @Original Author: Prof. Randy
    @Modified by: Chong Zhou
    update to python3: 03/15/2019
    Args:
        epsilon: the shrinkage parameter (either a scalar or a vector)
        x: the vector to shrink on

    Returns:
        The shrunk vector
    """
    output = np.array(x*0.)
    count = 0

    print(enumerate(x))
    #input(x)
    for idx, ele in enumerate(x):
        print("Enumerate Shrink: " + str(count) + " / " + str(enumerate(x)))
        if ele > epsilon:
            output[idx] = ele - epsilon
        elif ele < -epsilon:
            output[idx] = ele + epsilon
        else:
            output[idx] = 0.
        count += 1
    return output