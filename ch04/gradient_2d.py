import numpy as np
import matplotlib.pylab as plt


def numerical_diff(f, x):
    h = 1e-4
    return ((f(x + h) - f(x - h)) / (2 * h))


def _numerical_gradient_no_batch(f, x):
    h = 1e-4
    grad = np.zeros_like(x)

    for idx in range(x.size):
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x)

        x[idx] = float(tmp_val) - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2 * h)
        x[idx] = tmp_val
    return grad


def numerical_gradient(f, X):
    if X.ndim == 1:
        return _numerical_gradient_no_batch(f, X)
    else:
        grad = np.zeros_like(X)

        for idx, x in enumerate(X):
            grad[idx] = _numerical_gradient_no_batch(f, x)
        return grad


def function_tmp1(x0):
    return x0 ** 2 + 4.0 ** 2


def function_tmp2(x1):
    return 3.0 ** 2 + x1 ** 2


def function_2(x):
    return np.sum(x ** 2)


if __name__ == "__main__":
    # print(numerical_diff(function_tmp1, 3.0))
    # print(numerical_diff(function_tmp2, 4.0))
    # print(numerical_gradient(function_2, np.array([3.0, 4.0])))
    # print(numerical_gradient(function_2, np.array([0.0, 2.0])))
    # print(numerical_gradient(function_2, np.array([3.0, 0.0])))
    x0 = np.arange(-2, 2.5, 0.25)
    x1 = np.arange(-2, 2.5, 0.25)
    X, Y = np.meshgrid(x0, x1)

    X = X.flatten()
    Y = Y.flatten()

    grad = numerical_gradient(function_2, np.array([X, Y]).T).T

    plt.figure()
    plt.quiver(X, Y, -grad[0], -grad[1], angles="xy", color="#666666")
    plt.xlim([-2, 2])
    plt.ylim([-2, 2])
    plt.xlabel("x0")
    plt.ylabel("x1")
    plt.grid()
    plt.draw()
    plt.show()
