import numpy as np

def softmax(vec):
    """Compute softmax values for each element."""
    return np.exp(vec) / np.sum(np.exp(vec), axis=0)
