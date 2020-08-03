import numpy as np


def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    delta = 1e-7
    batch_size = y.shape[0]
    if np.count_nonzero(t == 0, axis=1) == 1:
        return -np.sum(np.log(y[np.arange(batch_size), t] + delta)) / batch_size
    else:
        return -np.sum(t * np.log(y + delta)) / batch_size


y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
print(cross_entropy_error(np.array(y), np.array(t)))
