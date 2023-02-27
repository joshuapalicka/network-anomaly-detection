import numpy as np
from numpy import dot
from numpy.linalg import norm


def get_Zed(orig_sample, reconstr_sample):
    return np.linalg.norm(orig_sample - reconstr_sample)


def get_Zcs(orig_sample, reconstr_sample):
    return dot(orig_sample[0], reconstr_sample[0]) / (norm(orig_sample[0]) * norm(reconstr_sample[0]))


def getZc(encoded_sample):  # TODO: Implement when we have MemAE implemented
    return encoded_sample


def getZVector(orig_sample, reconstr_sample, encoded_sample):
    zVector = []
    #print("Encoded")
    #print(encoded_sample)
    for i in range(len(encoded_sample)):
        zVector.append(float(encoded_sample[0][i]))
    #zVector = [float(item) for item in encoded_sample]
    zVector.append(float(get_Zed(orig_sample, reconstr_sample)))
    zVector.append(float(get_Zcs(orig_sample, reconstr_sample)))
    #print()
    #print("Vector")
    #print(zVector)
    return zVector
