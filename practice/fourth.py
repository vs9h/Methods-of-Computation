import numpy as np
from numpy.linalg import norm
from numpy.linalg import solve
import matplotlib.pyplot as plt
from enum import Enum, auto

LIMIT = 500

def hilbert(n, precision=16):
    return np.fromfunction(lambda i, j: np.round(1 / (i + j + 1), precision), (n, n))


def dim(A):
    return A.shape[0]

def check_conditions(b):
    condition_nums = [0, 0, 0]
    n = dim(b)
    for j in range(n):
        row_max = 0
        column_max = 0
        for i in range(n):
            row_max += norm(b[i][j])
            column_max += norm(b[j][i])
        condition_nums[0] = max(condition_nums[0], row_max)
        condition_nums[1] = max(condition_nums[1], column_max)
    for j in range(n):
        for i in range(n):
            condition_nums[2] += b[i][j]
    print(condition_nums)
    return condition_nums


def solveWithSimpleIterationMethod(matrix, b_init, eps, n):
    a = np.zeros((n, n))
    b = np.zeros((n, 1))
    for i in range(n):
        for j in range(n):
            if i!=j:
                a[i][j] = -matrix[i][j] / matrix[i][i]
        b[i] = b_init[i] / matrix[i][i]
    amount = 1
    x_cur = a @ b_init + b
    x_prev = b_init
    while (amount < LIMIT) and (norm(x_cur - x_prev) > eps):
        x_prev = x_cur
        x_cur = a @ x_prev + b
        amount += 1

    return x_cur, amount


def solveWithSeidelMethod(matrix, b_init, eps, n):
    R = np.zeros((n, n))
    L = np.zeros((n, n))
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i > j:
                L[i][j] = matrix[i][j]
            elif i < j:
                R[i][j] = matrix[i][j]
            else:
                D[i][j] = matrix[i][j]

    beta = np.linalg.inv(D + L)
    a = np.multiply(beta, -1) @ R
    b = beta @ b_init
    amount = 1
    x_cur = a @ b_init + b
    x_prev = b_init
    while (amount < LIMIT) and (norm(x_cur - x_prev) > eps):
        x_prev = x_cur
        x_cur = a @ x_prev + b
        amount += 1

    return x_cur, amount


class DrawProperty(Enum):
    ONLY_AMOUNT = auto()
    ONLY_ERROR = auto()


def draw_test(matrix, b, from_eps_degree, to_eps_degree, prop, title, logy="false"):
    epsilons = np.logspace(from_eps_degree, to_eps_degree, 300)
    errors = []
    errors_seidel = []
    amounts = []
    amounts_seidel = []
    n = dim(matrix)
    precise_solution = solve(matrix, b)

    for eps in epsilons:
        solution, amount = solveWithSimpleIterationMethod(matrix, b, eps, n)
        solution_seidel, amount_seidel = solveWithSeidelMethod(matrix, b, eps, n)
        errors.append(norm(solution - precise_solution))
        errors_seidel.append(norm(solution_seidel - precise_solution))
        amounts.append(amount)
        amounts_seidel.append(amount_seidel)

    plt.xscale('log')

    plt.title(title)
    if prop == DrawProperty.ONLY_ERROR:
        if (logy != "false"):
            plt.yscale('log')
        plt.plot(epsilons, errors, color='orange', label="Погрешность (метод простой итерации)")
        plt.plot(epsilons, errors_seidel, color='aqua', label="Погрешность (метод Зеделя)")
    else:
        plt.plot(epsilons, amounts, color='green', label="Кол-во итераций (метод простой итерации)")
        plt.plot(epsilons, amounts_seidel, color='blue', label="Кол-во итераций (метод Зеделя)")

    plt.ylabel("Порядок велчин")
    plt.xlabel("Epsilon")

    plt.legend()
    plt.show()

def draw_graphics(matrix, vector, title, logy, from_eps = -12, to_eps = -1):
    draw_test(matrix, vector, from_eps, to_eps, DrawProperty.ONLY_ERROR, title, logy)
    draw_test(matrix, vector, from_eps, to_eps, DrawProperty.ONLY_AMOUNT, title)

def test():

    diagonal = np.array([
        [-400.6, 0, 0],
        [0, -600.4, 0],
        [0, 0, 200.2]])

    vector = np.array([[0.408388], [-0.872359], [-1.06988]])
    # draw_graphics(diagonal, vector, "Диагональная матрица", "false")
    upper_triangle = np.array(
        [[-198.1, 389.9, 123.2],
         [0, 202.4, 249.3],
         [0, 0, 489.2]]
    )
    vector = np.array([[-0.862254], [0.0249838], [-2.32304]])
    # draw_graphics(upper_triangle, vector, "Верхняя треугольная матрица", "false")

    tridiagonal = np.array(
        [[2, -1, 0, 0, 0],
         [-3, 8, -1, 0, 0],
         [0, -5, 12, 2, 0],
         [0, 0, -6, 18, -4],
         [0, 0, 0, -5, 10]]
    )
    vector = np.array([[0.92884], [0.630153], [0.580092], [-0.200029], [-0.200029]])
    # draw_graphics(tridiagonal, vector, "Трёхдиагональная матрица", "true")

    simple_first = np.array([[1, 0.99], [0.99, 0.98]])
    vector = np.array([[0.715274], [-1.17753]])
    # draw_graphics(simple_first, vector, "Двумерная матрица", "false")

    simple_second = np.array([[-401.98, 200.34], [1202.04, -602.32]])
    # draw_graphics(simple_second, vector, "Двумерная (2) матрица", "false")

    h = hilbert(4)
    vector = np.array([[1.4514], [-1.99799], [1.83011], [-0.471568]])
    draw_graphics(h, vector, "Матрица гильберта 4 порядка", "false")

    h = hilbert(5)
    vector = np.array([[-0.943337], [-1.25822], [-1.39549], [0.738115], [-0.0660465]])
    draw_graphics(h, vector, "Матрица гильберта 5 порядка", "false")

    h = hilbert(6)
    vector = np.array([[1.07713],[0.232377],[2.22464],[-1.12486],[1.80719], [-0.113347]])
    # draw_graphics(h, vector, "Матрица гильберта 6 порядка", "false")



if __name__ == '__main__':
    test()
