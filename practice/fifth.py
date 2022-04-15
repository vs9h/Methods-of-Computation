import numpy as np
from numpy.linalg import norm
from numpy import linalg as LA
import matplotlib.pyplot as plt
from enum import Enum, auto

LIMIT = 500


def dim(matrix):
    return matrix.shape[0]


def hilbert(n, precision=16):
    return np.fromfunction(lambda i, j: np.round(1 / (i + j + 1), precision), (n, n))

class DrawProperty(Enum):
    ONLY_AMOUNT = auto()
    ONLY_ERROR = auto()


def get_test_data():
    test_data = []

    test_data.append([np.array(
        [[-198.1, 389.9, 123.2],
         [0, 202.4, 249.3],
         [0, 0, 489.2]]),
        np.array([[-0.862254], [0.0249838], [-2.32304]]),
        "Верхняя треугольная матрица"])
    test_data.append([np.array(
        [[2, -1, 0, 0, 0],
         [-3, 8, -1, 0, 0],
         [0, -5, 12, 2, 0],
         [0, 0, -6, 18, -4],
         [0, 0, 0, -5, 10]]),
        np.array([[0.92884], [0.630153], [0.580092], [-0.200029], [-0.200029]]),
        "Трёхдиагональная матрица"])
    test_data.append(
        [np.array([[1, 0.99], [0.99, 0.98]]), np.array([[0.715274], [-1.17753]]), "(1-ая) Двумерная матрица"])
    test_data.append([np.array([[-401.98, 200.34], [1202.04, -602.32]]), np.array([[0.715274], [-1.17753]]),
                      "(2-ая) Двумерная матрица"])
    test_data.append(
        [hilbert(4), np.array([[1.4514], [-1.99799], [1.83011], [-0.471568]]), "Матрица гильберта 4 порядка"])
    test_data.append([hilbert(5), np.array([[-0.943337], [-1.25822], [-1.39549], [0.738115], [-0.0660465]]),
                      "Матрица гильберта 5 порядка"])
    test_data.append([hilbert(6), np.array([[1.07713], [0.232377], [2.22464], [-1.12486], [1.80719], [-0.113347]]),
                      "Матрица гильберта 6 порядка"])

    test_data.append([hilbert(10), np.random.rand(10, 1),
                      "Матрица гильберта 10 порядка"])

    test_data.append([hilbert(16), np.random.rand(16, 1),
                      "Матрица гильберта 16 порядка"])

    return test_data


def CalculateMaxEigenValueWithPowerMethod(matrix, x, eps):
    x_cur = matrix @ x
    x_prev = x
    amount = 1
    value_prev = x_cur[0][0] / x_prev[0][0]
    x_prev = x_cur
    x_cur = matrix @ x_cur
    value_cur = x_cur[0][0] / x_prev[0][0]
    while (amount < LIMIT) and (norm(value_cur - value_prev) > eps):
        value_prev = value_cur
        x_prev = x_cur
        x_cur = matrix @ x_cur
        value_cur = x_cur[0][0] / x_prev[0][0]
        amount += 1

    return np.abs(value_cur), amount


def CalculateMaxEigenvalueWithScalarProductsMethod(matrix, x, eps):
    x_current = matrix @ x
    x_previous = x
    y_current = matrix.T @ x_previous
    value_previous = (x_current.T @ y_current)[0][0] / (x_previous.T @ x)[0][0]
    amount = 1

    x_previous = x_current
    x_current = matrix @ x_current
    y_current = matrix.T @ y_current

    value_current = (x_current.T @ y_current)[0][0] / (x_previous.T @ y_current)[0][0]
    while (amount < LIMIT) and (np.abs(value_current - value_previous) > eps):
        value_previous = value_current
        x_previous = x_current
        x_current = matrix @ x_current
        y_current = matrix.T @ y_current

        value_current = (x_current.T @ y_current)[0][0] / (x_previous.T @ y_current)[0][0]
        amount += 1

    return np.abs(value_current), amount


def process_test(test_element, prop):
    [matrix, x, title] = test_element

    errors_power = []
    errors_scalar = []
    amounts_power = []
    amounts_scalar = []

    eigenValues = LA.eigvals(matrix)
    eigenValue = eigenValues[0]
    for i in range(len(eigenValues)):
        if norm(abs(eigenValues[i]) > abs(eigenValue)):
            eigenValue = eigenValues[i]

    eigenValue = eigenValue.real

    from_eps_degree = -5
    to_eps_degree = -2
    epsilons = np.logspace(from_eps_degree, to_eps_degree, 300)
    for eps in epsilons:
        [new_eigen_value_power, amount_power] = CalculateMaxEigenValueWithPowerMethod(matrix, x, eps)
        [new_eigen_value_scalar, amount_scalar] = CalculateMaxEigenvalueWithScalarProductsMethod(matrix, x, eps)
        errors_power.append(norm(np.abs(eigenValue) - np.abs(new_eigen_value_power)))
        errors_scalar.append(norm(np.abs(eigenValue) - np.abs(new_eigen_value_scalar)))
        amounts_power.append(amount_power)
        amounts_scalar.append(amount_scalar)

    plt.xscale('log')

    plt.title(title)
    if prop == DrawProperty.ONLY_ERROR:
        plt.plot(epsilons, errors_power, color='orange', label="Погрешность (Power)")
        plt.plot(epsilons, errors_scalar, color='aqua', label="Погрешность (Scalar)")
    else:
        plt.plot(epsilons, amounts_power, color='green', label="Кол-во итераций (Power)")
        plt.plot(epsilons, amounts_scalar, color='blue', label="Кол-во итераций (Scalar)")

    plt.ylabel("Порядок велчин")
    plt.xlabel("Epsilon")

    plt.legend()
    plt.show()


if __name__ == '__main__':
    data = get_test_data()
    max_test_number = len(data)

    while 0 == 0:
        print(f"Введите номера тестов [1..{max_test_number}]")
        test_numbers = list(map(int, input().split()))

        for i in test_numbers:
            if (i < 1) or (i > max_test_number):
                continue
            print("*" * 30)
            print(f"Тест #{i}")
            print(data[i-1][2])
            process_test(data[i - 1], DrawProperty.ONLY_ERROR)
            process_test(data[i - 1], DrawProperty.ONLY_AMOUNT)
            print("*" * 30, '\n')
