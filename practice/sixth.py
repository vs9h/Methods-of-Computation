import math
import numpy as np
from numpy.linalg import norm
from numpy import abs
from numpy import linalg as LA
import matplotlib.pyplot as plt
from enum import Enum, auto


def dim(matrix):
    return matrix.shape[0]


def find_max_element_indices(matrix):
    indices = [0, 1]
    for i in range(dim(matrix)):
        for j in range(i + 1, dim(matrix)):
            if np.abs(matrix[indices[0]][indices[1]]) < np.abs(matrix[i][j]):
                indices = [i, j]
    return indices


def in_gershgorin_circle(matrix, eigen_value):
    circles = []
    for i in range(dim(matrix)):
        sum = 0
        for j in range(dim(matrix)):
            sum += np.abs(matrix[i][j])
        circles.append([matrix[i][i], sum - np.abs(matrix[i][i])])

    for [element, value] in circles:
        if np.abs(element - eigen_value) <= value:
            return True
    return False


def hilbert(n, precision=16):
    return np.fromfunction(lambda i, j: np.round(1 / (i + j + 1), precision), (n, n))


class Strategy(Enum):
    MAX_MODULE_VALUE = auto()
    ZEROING_IN_ORDER = auto()


def get_test_data():
    test_data = []

    test_data.append([np.array(
        [[-0.81417, -0.01937, 0.41372],
         [-0.01937, 0.54414, 0.00590],
         [0.41372, 0.00590, -0.81445]]),
        "(1-ая) Матрица размерности 3"
    ])

    test_data.append([np.array(
        [[-1.51898, -0.19907, 0.95855],
         [-0.19907, 1.17742, 0.06992],
         [0.95855, 0.06992, -1.57151]]),
        "(2-ая) Матрица размерности 3"
    ])

    test_data.append([hilbert(4), "Матрица гильберта 4 порядка"])
    test_data.append([hilbert(7), "Матрица гильберта 7 порядка"])
    test_data.append([hilbert(10), "Матрица гильберта 10 порядка"])
    test_data.append([hilbert(15), "Матрица гильберта 15 порядка"])

    return test_data


# Jacobi method
def calculate_eigen_values(matrix, eps, strategy):
    [i, j, amount] = [0, 0, 0]
    cur_matrix = matrix.copy()
    size = dim(cur_matrix)

    while True:
        H = np.array(np.diag(np.full(size, 1, float)))

        if strategy == Strategy.MAX_MODULE_VALUE:
            [i, j] = find_max_element_indices(cur_matrix)
        elif strategy == Strategy.ZEROING_IN_ORDER:
            if (j < size - 1) and (j + 1 != i):
                j += 1
            elif j == size - 1:
                [i, j] = [i + 1, 0]
            else:
                j += 2
        if np.abs(cur_matrix[i][j]) < eps:
            return [cur_matrix.diagonal(), amount]
        amount += 1
        phi = math.atan(2 * cur_matrix[i][j] / (cur_matrix[i][i] - cur_matrix[j][j])) / 2
        value = math.cos(phi)
        H[j, j] = H[i, i] = value
        H[j, i] = math.sin(phi)
        H[i, j] = -H[j, i]
        cur_matrix = H.T @ cur_matrix @ H


def get_diff_between_eigen_values(lhs_sorted, rhs):
    return norm(lhs_sorted - sorted(rhs, key=lambda l: abs(l)))


def process_test(test_element):
    [matrix, title] = test_element

    amounts_mm = []
    amounts_zio = []
    errors_mm = []
    errors_zio = []
    is_in_gershgorin_circle = True

    eigen_values = np.array(sorted(LA.eigvals(matrix), key=lambda l: abs(l)))

    from_eps_degree = -5
    to_eps_degree = -2
    epsilons = np.logspace(from_eps_degree, to_eps_degree, 300)
    for eps in epsilons:
        [mm_eigen_values, mm_amount] = calculate_eigen_values(matrix, eps, Strategy.MAX_MODULE_VALUE)
        [zio_eigen_values, zio_amount] = calculate_eigen_values(matrix, eps, Strategy.ZEROING_IN_ORDER)
        amounts_mm.append(mm_amount)
        amounts_zio.append(zio_amount)
        errors_mm.append(get_diff_between_eigen_values(eigen_values, mm_eigen_values))
        errors_zio.append(get_diff_between_eigen_values(eigen_values, zio_eigen_values))
        for value in mm_eigen_values:
            if not in_gershgorin_circle(matrix, value):
                is_in_gershgorin_circle = False

    figure, axis = plt.subplots(1, 2, figsize=(16, 8))

    axis[0].plot(epsilons, errors_mm, color='orange', label="Max module")
    axis[0].plot(epsilons, errors_zio, color='aqua', label="Zero in order")
    axis[0].set_ylabel("Погрешность")
    axis[0].set_xlabel("Epsilon")
    axis[0].set_title(title)
    axis[0].set_xscale('log')
    axis[0].legend()

    axis[1].plot(epsilons, amounts_mm, color='orange', label="Max module")
    axis[1].plot(epsilons, amounts_zio, color='aqua', label="Zero in order")
    axis[1].set_ylabel("Кол-во итераций")
    axis[1].set_title(title)
    axis[1].set_xscale('log')
    axis[1].set_xlabel("Epsilon")
    min_y = min(min(amounts_mm), min(amounts_zio))
    max_y = max(max(amounts_mm), max(amounts_zio))
    axis[1].set_yticks(range(max_y, min_y, -math.ceil((max_y - min_y) / 5)))
    axis[1].legend()

    plt.title(title)
    plt.show()
    print(f"Is in gershgorin circle: {is_in_gershgorin_circle}")

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
            print(f"Тест #{k} -- {data[k-1][1]}")
            process_test(data[k - 1])
            print("*" * 30, '\n')
