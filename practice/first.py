from math import sqrt
import numpy as np
from numpy.linalg import norm
from numpy.linalg import inv
from numpy.linalg import det
import matplotlib.pyplot as plt


# spectral conditionality criterion
def cond_s(A):
    return norm(A) * norm(inv(A))


# volumetric criterion
def cond_v(A):
    dim = A.shape[0]
    numerator = 0
    for n in range(0, dim):
        row_sum = 0
        for m in range(0, dim):
            row_sum += A.item(n, m) ** 2
        numerator += sqrt(row_sum)
    return numerator / abs(det(A))


# angular criterion
def cond_a(A):
    dim = A.shape[0]
    C = A.I
    cond = 0
    for i in range(0, dim):
        cond = max(cond, norm(A[i, :]) * norm([C[:, i]]))
    return cond


def hilbert(n, precision=10):
    H = np.empty((n, n))
    for i in range(n):
        for j in range(n):
            H[i][j] = round(1 / ((i + 1) + (j + 1) - 1), precision)
    return H


def smooth_around(a):
    result = []
    for i in range(len(a)):
        i_from = max(i - 3, 0)
        i_to = min(len(a), i + 3)
        avg = 0
        for j in range(i_from, i_to):
            avg += a[j] / (i_to - i_from)
        result.append(avg)
    return result


def DrawCriteriesForMatrix(A, fromErrorDegree, toErrorDegree, smooth, label):
    n = A.shape[0]
    b = A @ np.ones((n, 1))

    plt.ylabel("Порядок велчин")
    plt.xlabel("Шум")

    alphas = []

    x = np.linalg.solve(A, b)

    errors = []
    conds_s = []
    conds_v = []
    conds_a = []
    print(f"cond_s(A) = {cond_s(A)}, cond_v(A) = {cond_v(A)}, cond_a(A) = {cond_a(A)}")

    for alpha in np.logspace(fromErrorDegree, toErrorDegree, 1000, endpoint=False):
        alphas.append(alpha)
        b_error = b + alpha * np.ones((n, 1))
        A_tilda = A + alpha * np.random.rand(n, n)
        x_tilda = np.linalg.solve(A_tilda, b_error)
        errors.append(np.linalg.norm(x_tilda - x))
        conds_s.append(cond_s(A_tilda))
        conds_v.append(cond_v(A_tilda))
        conds_a.append(cond_a(A_tilda))

    plt.xscale('log')
    plt.yscale('log')

    if smooth:
        errors = smooth_around(errors)
        conds_s = smooth_around(conds_s)
        conds_a = smooth_around(conds_a)
        conds_v = smooth_around(conds_v)

    plt.plot(alphas, errors, color='b', label="Погрешность")
    plt.plot(alphas, conds_s, color='y', label="Спектральный критерий")
    plt.plot(alphas, conds_v, color='r', label="Объёмный критерий")
    plt.plot(alphas, conds_a, color='orange', label="Угловой критерий")
    plt.title(label)
    plt.legend()
    plt.grid(linestyle='--')
    plt.hlines(y=1, xmin=alphas[0], xmax=alphas[len(alphas) - 1], colors='purple', linestyles='--', lw=1, label='y=1')

    plt.show()


def DrawCriteriesForMatrixFixedB(A, b, fromError, toError, smooth, label):
    n = A.shape[0]

    plt.ylabel("Порядок велчин")
    plt.xlabel("Шум")

    alphas = []

    x = np.linalg.solve(A, b)

    errors = []
    conds_s = []
    conds_v = []
    conds_a = []
    print(f"cond_s(A) = {cond_s(A)}, cond_v(A) = {cond_v(A)}, cond_a(A) = {cond_a(A)}")

    for alpha in np.logspace(fromError, toError, 1000, endpoint=False):
        alphas.append(alpha)
        # b_error = b + alpha * np.ones((n, 1))
        b_error = b + alpha * np.random.rand(n, 1)
        x_tilda = np.linalg.solve(A, b_error)
        errors.append(np.linalg.norm(x_tilda - x))
        conds_s.append(cond_s(A))
        conds_v.append(cond_v(A))
        conds_a.append(cond_a(A))

    plt.xscale('log')
    plt.yscale('log')

    if smooth:
        errors = smooth_around(errors)
        conds_s = smooth_around(conds_s)
        conds_a = smooth_around(conds_a)
        conds_v = smooth_around(conds_v)

    plt.plot(alphas, errors, color='b', label="Погрешность")
    plt.plot(alphas, conds_s, color='y', label="Спектральный критерий")
    plt.plot(alphas, conds_v, color='r', label="Объёмный критерий")
    plt.plot(alphas, conds_a, color='orange', label="Угловой критерий")
    plt.title(label)
    plt.legend()
    plt.grid(linestyle='--')
    plt.hlines(y=1, xmin=alphas[0], xmax=alphas[len(alphas) - 1], colors='purple', linestyles='--', lw=1, label='y=1')
    plt.show()


if __name__ == '__main__':
    # # Different hilbert matrix (vary arity, max precision)
    for i in range(3, 8):
        DrawCriteriesForMatrix(np.matrix(hilbert(i, 16)), -10, 0, "false", f"Матрица Гильберта (n = {i}, precision = {16})")

    # Different hilbert matrix (fix arity, vary precision)
    # for i in range(1, 7):
    #     DrawCriteriesForMatrix(np.matrix(hilbert(4, i)), -10, 0, "false", f"Матрица Гильберта (n = {5}, precision = {i})")

    # matrices = [
    #     # { -0.2, 0.6 } -> ~ { 0.6, 2.2 },
    #     np.matrix([[-400.6, 199.8], [1198.8, -600.4]]),
    #     np.matrix([[-401.52, 200.16], [1200.96, -601.68]]),
    #     np.matrix([[-401.43, 200.19], [1201.14, -601.62]])
    # ]
    # b = np.matrix([[200], [-600]])
    # index_from = -10
    # index_to = 0
    # for matrix in matrices:
    #     DrawCriteriesForMatrixFixedB(matrix, b, index_from, index_to, "false", "Система из методички (Фиксированный b)")
    #     DrawCriteriesForMatrix(matrix, index_from, index_to, "false", "Система из методички")

    tridiagonal_matrix = np.matrix([[40, 9, 0, 0], [10, 50, 2, 0], [0, 4, 64, -4], [0, 0, 7, 80]])
    DrawCriteriesForMatrix(tridiagonal_matrix, -10, 3, "false", f"Трёхдиагональная матрица с диагональным преобл.")

