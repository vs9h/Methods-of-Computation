import math
import numpy as np
import scipy as sp
import scipy.special
import scipy.misc
import scipy.integrate
from numpy.linalg import norm
import matplotlib.pyplot as plt


def dim(matrix):
    return matrix.shape[0]


class F:
    def __init__(self, function, name):
        self.f = function
        self.name = name
        self.df = lambda t: sp.misc.derivative(self.f, t)
        self.ddf = lambda t: sp.misc.derivative(self.df, t)


def get_test_data():
    test_data = []
    segment = [-1, 1]
    test_data.append([
        [
            lambda x: -(4 + x) / (5 + 2 * x),
            lambda x: x / 2 - 1,
            lambda x: 1 + math.exp(x / 2),
            lambda x: 2 + x
        ],
        segment
    ])
    test_data.append([
        [
            lambda x: -(4 - x) / (5 - 2 * x), lambda x: (1 - x) / 2,
            lambda x: math.log(3 + x) / 2, lambda x: 1 + x / 3
        ],
        segment
    ])
    test_data.append([
        [
            lambda x: -(6 + x) / (7 + 3 * x), lambda x: -(1 - x / 2),
            lambda x: 1 + math.cos(x) / 2, lambda x: 1 - x / 3
        ],
        segment
    ])
    return test_data


def jacobi_polynomial(n, k):
    return F(lambda t: (1-t**2) * sp.special.eval_jacobi(n, k, k, t), "Jacobi polynomial")


def A_i(functions, phi_i):
    [k, p, q, *_] = functions
    return lambda x: k(x) * phi_i.ddf(x) + p(x) * phi_i.df(x) + q(x) * phi_i.f(x)


def galerkin_method(data_from, N):
    [functions, segment] = data_from
    f = functions[3]
    [a, b] = segment
    phi = [jacobi_polynomial(i, 1) for i in range(N)]
    A = np.array([A_i(functions, phi[i]) for i in range(N)])
    B = np.array([sp.integrate.quad(lambda t: phi[i].f(t) * A[j](t), a, b)[0]
                  for i in range(N) for j in range(N)]).reshape((N, N))
    C = np.array([sp.integrate.quad(lambda t: f(t) * phi[i].f(t), a, b)[0] for i in range(N)])

    alpha = np.linalg.solve(B, C)
    return lambda t: sum([alpha[i] * phi[i].f(t) for i in range(N)])


def get_graph_data(data_from, config):
    [N, h] = config
    u = galerkin_method(data_from, N)
    a, b = data_from[1]
    n = round((b - a) / h)
    x = [a + i * h for i in range(n + 1)]
    y = [u(x[i]) for i in range(n + 1)]
    title = f"N = {N}, h = {h}"
    return x, y, title


def draw(plt_data):
    figure, axis = plt.subplots(2, 2, figsize=(16, 8))

    [i, j] = [0, 0]

    def set_cur_grid():
        [xs, ys, title] = plt_data[2 * i + j]
        axis[i, j].plot(xs, ys, ".-", color='orange', label="Errors", linewidth=0.5)
        axis[i, j].set_title(title)
        axis[i, j].legend()

    set_cur_grid()
    j += 1

    set_cur_grid()
    [i, j] = [1, 0]

    set_cur_grid()
    j += 1
    set_cur_grid()

    figure.tight_layout()
    plt.show()


def process_test(data_from):
    graphs = []
    test_cases = [
        [1, 0.05],
        [3, 0.04],
        [5, 0.03],
        [8, 0.01]
    ]

    for test_config in test_cases:
        graphs.append(get_graph_data(data_from, test_config))
    draw(graphs)

    test_cases = [
        [3, 0.05],
        [5, 0.04],
        [8, 0.03],
        [8, 0.01],
    ]
    graphs = []
    for test_config in test_cases:
        graphs.append(get_graph_data(data_from, test_config))
    draw(graphs)


if __name__ == '__main__':
    data = get_test_data()
    max_test_number = len(data)

    while 0 == 0:
        print(f"Введите номера тестов [1..{max_test_number}]")
        test_numbers = list(map(int, input().split()))

        for k in test_numbers:
            if (k < 1) or (k > max_test_number):
                continue
            print("*" * 30)
            print(f"Тест #{k}")
            process_test(data[k - 1])
            print("*" * 30, '\n')
