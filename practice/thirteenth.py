import math
from random import random

import numpy as np
import scipy as sp
import scipy.misc
import scipy.integrate
import matplotlib.pyplot as plt


class F:
    def __init__(self, function, name):
        self.f = function
        self.name = name
        self.df = lambda t: sp.misc.derivative(self.f, t)
        self.ddf = lambda t: sp.misc.derivative(self.df, t)


class Segment:
    def __init__(self, a, b):
        self.a = a
        self.b = b


class TestConfig:
    def __init__(self, g, segment, dens_fs, Ns):
        self.g = g
        self.segment = segment
        self.density_functions = dens_fs
        self.Ns = Ns



def get_test_data():
    segment = Segment(0, math.pi/2)
    test_data = [
        TestConfig(
            g=F(lambda x: math.cos(x), "cos(x)"),
            segment=segment,
            dens_fs=[
                F(lambda x: 1 / (segment.b - segment.a), "1/b-a = 2/pi"),
                F(lambda x: 2/math.pi, "2/pi"),
                F(lambda x: (4/math.pi)*(1-2*x/math.pi), "4/pi * (1-2x/pi) = 4/pi-8x/pi^2"),
            ],
            Ns=range(250, 10001, 250)
        ),
        TestConfig(
            g=F(lambda x: math.cos(x), "cos(x)"),
            segment=segment,
            dens_fs=[
                F(lambda x: 1 / (segment.b - segment.a), "1/b-a = 2/pi"),
                F(lambda x: 2 / math.pi, "2/pi"),
                F(lambda x: (4 / math.pi) * (1 - 2 * x / math.pi), "4/pi * (1-2x/pi) = 4/pi-8x/pi^2"),
            ],
            Ns=range(100, 1001, 50)
        )
    ]
    return test_data


def get_data_for_plt(f, segment, n):
    h = (segment.b - segment.a)/n
    x = [segment.a + i * h for i in range(n + 1)]
    y = [f.f(x[i]) for i in range(n + 1)]
    label = f"{f.name}"
    return x, y, label


def draw_multiple_plt(data):
    figure, axis = plt.subplots(2, 2, figsize=(16, 8))

    [i, j] = [0, 0]

    def set_cur_grid():
        [xs, ys, label] = data[2 * i + j]
        axis[i, j].plot(xs, ys, ".-", color='orange', label=label, linewidth=0.5)
        axis[i, j].set_title(label)
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

def draw_single_plt(data, title, precise="undefined"):
    plt.title(title)

    colors = ['aqua', 'green', 'orange', 'r', 'grey']

    for i in range(len(data)):
        [xs, ys, label] = data[i]
        plt.plot(xs, ys, "-", color=colors[i], label=label)
        plt.legend()

    if precise != "undefined":
        plt.hlines(y=precise, xmin=data[0][0][0], xmax=data[0][0][len(data[0][0]) - 1], colors='black', linestyles='--', lw=1,
               label=f"precise value = {precise:.2e}")
    plt.show()


def draw_single_plt_diff(data, precise):
    figure, axis = plt.subplots(1, 2, figsize=(16, 8))

    colors = ['aqua', 'green', 'orange', 'r', 'grey']

    for i in range(len(data)):
        [xs, ys, label] = data[i]
        axis[0].plot(xs, ys, "-", color=colors[i], label=label)
        axis[0].set_title("Values")
        axis[0].legend()
    if precise != "undefined":
        axis[0].axhline(y=precise, color='black', linestyle='--', lw=1,
                    label=f"precise value = {precise:.2e}")

    for i in range(len(data)):
        [xs, ys, label] = data[i]
        ys = [np.abs(y - precise) for y in ys]
        axis[1].plot(xs, ys, "-", color=colors[i], label=label)
        axis[1].set_title("Diff between values")
        axis[1].legend()
    if precise != "undefined":
        axis[1].axhline(y=0, color='black', linestyle='--', lw=1,
                    label=f"precise value = {precise:.2e}")

    plt.show()



def find_function_max_in_interval(f, segment):
    n = 1000
    h = (segment.b - segment.a) / n
    return max([f.f(segment.a + i * h) for i in range(n+1)])


def monte_carlo_area(data_from, N):
    k = 0
    X = []
    segment = data_from.segment
    a=segment.a
    b=segment.b
    density_max = find_function_max_in_interval(data_from.g, segment)

    for i in range(N):
        X.append((random() * (b - a) + a, random() * density_max))
    for i in range(N):
        if X[i][1] <= data_from.g.f(X[i][0]):
            k += 1
    return k * density_max * (b-a) / N


def process_test(data_from):
    n = 100
    segment = data_from.segment
    plt_data = [get_data_for_plt(function, segment, n) for function in data_from.density_functions]
    plt_data.append(get_data_for_plt(data_from.g, segment, n))
    draw_single_plt(plt_data, "Density functions")
    max_g = data_from.g.f(data_from.segment.b)
    print(f"Max g(x) = {max_g}")
    precise_value = sp.integrate.quad(data_from.g.f, segment.a, segment.b)[0]
    print(f"Precise value = {precise_value}")

    data_for_different_n = []
    for i in range(len(data_from.density_functions) + 1):
        data_for_different_n.append([])

    l=len(data_from.density_functions)
    for N in data_from.Ns:
        data_for_different_n[l].append(monte_carlo_area(data_from,N))
        for j in range(len(data_from.density_functions)):
            p = data_from.density_functions[j]
            approx_value = 0
            i = N
            density_max = find_function_max_in_interval(p, segment)
            while i > 0:
                x = random() * (segment.b - segment.a) + segment.a
                y = random() * density_max
                if (y < 0) or (y > p.f(x)):
                    continue
                i -= 1
                approx_value += data_from.g.f(x)/p.f(x)
            approx_value /= N
            data_for_different_n[j].append(approx_value)
    plt_data = [(data_from.Ns, data_for_different_n[i], f"f={data_from.density_functions[i].name}") for i in range(len(data_from.density_functions))]
    plt_data.append((data_from.Ns, data_for_different_n[l], "Area"))
    draw_single_plt_diff(plt_data, precise_value)


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
