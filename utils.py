import numpy as np

def softmax(vec):
    """Compute softmax values for each element."""
    return np.exp(vec) / np.sum(np.exp(vec), axis=0)
    # return (vec) / np.sum((vec), axis=0)

def distance(v1, v2):
    return np.linalg.norm(np.array(v1) - np.array(v2))
