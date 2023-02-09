import numpy as np
from numpy import dot
from numpy.linalg import norm


def get_Zed(orig_sample, reconstr_sample):
    return np.linalg.norm(orig_sample - reconstr_sample)


def get_Zcs(orig_sample, reconstr_sample):
    return dot(orig_sample[0], reconstr_sample[0]) / (norm(orig_sample[0]) * norm(reconstr_sample[0]))


def getZc(orig_sample):  # TODO: Implement when we have MemAE implemented
    return 0.0


def getZVector(orig_sample, reconstr_sample):
    # does not return zc yet, as this is unable to be implemented yet
    return [float(get_Zed(orig_sample, reconstr_sample)), float(get_Zcs(orig_sample, reconstr_sample))]
