import numpy as np


def numerical_diff(f, x):
    h = 1e-4
    return ((f(x + h) - f(x - h)) / (2 * h))


def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)

    for idx in range(x.size):
        tmp_val = x[idx]
        x[idx] = tmp_val + h
        fxh1 = f(x)

        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2 * h)
        x[idx] = tmp_val
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
    print(numerical_gradient(function_2, np.array([3.0, 4.0])))
    print(numerical_gradient(function_2, np.array([0.0, 2.0])))
    print(numerical_gradient(function_2, np.array([3.0, 0.0])))
