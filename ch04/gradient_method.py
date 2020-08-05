import numpy as np
from gradient_2d import numerical_gradient


def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x

    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad
    return x


def function_2(x):
    return x[0] ** 2 + x[1] ** 2


if __name__ == "__main__":
    init_x = np.array([-3.0, 4.0])
    print(gradient_descent(function_2, init_x=init_x, lr=0.1, step_num=100))

    # 学習率が大きすぎる例：lr=10
    init_x = np.array([-3.0, 4.0])
    print(gradient_descent(function_2, init_x, lr=10.0, step_num=100))

    # 学習率が小さすぎる例：lr=1e-10
    init_x = np.array([-3.0, 4.0])
    print(gradient_descent(function_2, init_x, lr=1e-10, step_num=100))
