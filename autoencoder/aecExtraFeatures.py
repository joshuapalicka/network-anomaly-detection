import numpy as np
from numpy import dot
from numpy.linalg import norm


def get_Zed(orig_sample, reconstr_sample):
    return np.linalg.norm(orig_sample - reconstr_sample)


def get_Zcs(orig_sample, reconstr_sample):
    return dot(orig_sample, reconstr_sample) / (norm(orig_sample) * norm(reconstr_sample))


def getZc(encoded_sample):  # TODO: Implement when we have MemAE implemented
    return encoded_sample
