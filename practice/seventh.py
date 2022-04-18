import math
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt


def dim(matrix):
    return matrix.shape[0]


def calculate_coeffs(data_from, step):
    [functions, conds, segment] = data_from
    [k, p, q, f] = functions
    [alpha_0, alpha_1, beta_0, beta_1, Ac, Bc] = conds
    [a, b] = segment
    n = round((b - a) / step)
    x = []
    for i in range(n + 1):
        x.append(a + i * step)
    A, B, C, D = [0], [step * alpha_0 - alpha_1], [alpha_1], [step * Ac]
    for i in range(1, n):
        A.append(2 * k(x[i]) - step * p(x[i]))
        B.append(-4 * k(x[i]) + 2 * step ** 2 * q(x[i]))
        C.append(2 * k(x[i]) + step * p(x[i]))
        D.append(2 * step ** 2 * f(x[i]))
    A.append(-beta_1)
    B.append(step * beta_0 + beta_1)
    C.append(0)
    D.append(step * Bc)
    return A, B, C, D


# Test data element: [functions, conds, element]

# functions = [k, p, q, f]
# cond = [alpha_0, alpha_1, beta_0, beta_1, Ac, Bc]
# segment = [a,b]

# alpha_0 * u(a) + alpha_1 * du(a) = Ac
#  beta_0 * u(b) +  beta_1 * du(b) = B
def get_test_data():
    test_data = []
    segment = [-1, 1]

    test_data.append([
        [
            lambda x: -(4 - x) / (5 - 2 * x), lambda x: (1 - x) / 2,
            lambda x: math.log(3 + x) / 2, lambda x: 1 + x / 3],
        [1, 0, 1, 0, 0, 0],
        segment
    ])
    test_data.append([
        [
            lambda x: -(6 + x) / (7 + 3 * x), lambda x: -(1 - x/2),
            lambda x: 1 + math.cos(x) / 2, lambda x: 1 - x / 3],
        [-2, 1, 0, 1, 0, 0],
        segment
    ])
    test_data.append([
        [
            lambda x: -(5 - x) / (7 - 3 * x), lambda x: -(1 - x) / 2,
            lambda x: 1 + math.sin(x) / 2, lambda x: 1 / 2 + x / 2],
        [0, 1, 3, 2, 0, 0],
        segment
    ])

    return test_data


def solve(coeffs):
    [A, B, C, D] = coeffs
    [s, t, u] = [[-C[0]/B[0]], [D[0]/B[0]], [0]]
    n = len(A) - 1
    for i in range(1, len(A)):
        s.append(-C[i] / (A[i] * s[i - 1] + B[i]))
        t.append((D[i] - A[i] * t[i - 1]) / (A[i] * s[i - 1] + B[i]))
        u.append(0)
    u[n] = t[n]
    for i in range(n-1, -1, -1):
        u[i] = s[i] * u[i+1] + t[i]
    return u


def grid(data_from, step, eps):
    coeff = 2
    k = 0
    p = 1
    v2 = solve(calculate_coeffs(data_from, step))
    while True:
        k += 1
        v1 = v2.copy()
        v2 = solve(calculate_coeffs(data_from, step/coeff ** k))
        errors = []
        for i in range(len(v1)):
            errors.append((v2[2 * i] - v1[i]) / (coeff ** p - 1))
        if norm(np.array(errors), 2) < eps:
            for i in range(len(errors)):
                if i % 2 == 0:
                    v2[2 * i] += errors[i]
                else:
                    v2[i] += (errors[i - 1] + errors[i + 1]) / 2
            x = []
            a = data_from[2][0]
            for i in range(len(v2)):
                x.append(a + i * step / (coeff ** k))
            title = f"Eps: {eps}, Grid step: {step / coeff ** k}, Thickening steps: {k}"
            return x, v2, title


def draw(tuples):
    figure, axis = plt.subplots(2, 2, figsize=(16, 8))

    [i, j] = [0, 0]

    def set_cur_grid():
        [xs, ys, title] = tuples[2 * i + j]
        axis[i, j].plot(xs, ys, color='orange', label="Errors", linewidth=0.5)
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
    for i in range(1, 5):
        graphs.append(grid(data_from, step=0.125, eps=10 ** -i))
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


def explicit_scheme(config):
    # (u_n_next - u_n) / tau = k/h^2 (u_n-1 - 2u_n + u_n+1) + f(x_n,t)
    U = np.zeros((config.N + 1, config.M + 1), dtype=float)

    for i in range(0, config.N + 1):
        U[i, 0] = config.mu.subs(x, config.x_grid[i])
    for j in range(0, config.M + 1):
        U[0, j] = config.mu_1.subs(t, config.time_grid[j])
        U[config.N, j] = config.mu_2.subs(t, config.time_grid[j])
    #
    # for i in range(1, config.M + 1):
    #     for j in range(1, config.N):
    #         rhs = (config.kappa/(config.h**2) * (U[j - 1, i - 1] - 2 * U[j, i - 1] + 2 * U[j + 1, i - 1])
    #                + config.f.subs([(x, config.x_grid[j]), (t, config.time_grid[i - 1])]))
    #         U[j, i] = rhs * config.tau + U[j, i - 1]
    for k in range(1, config.M + 1):
        for i in range(0, config.N + 1):
            U[i, 0] = config.mu.subs(x, config.x_grid[i])
        for i in range(1, config.N):
            U[i, k] = config.tau * config.kappa * (U[i + 1, k - 1] - 2 * U[i, k - 1] + U[i - 1, k - 1]) / (config.h ** 2) + \
                        config.tau * config.f.subs([(x, config.x_grid[i]), (t, config.time_grid[k - 1])]) + U[i, k - 1]
        # u_p[0, k] = mu1.subs(t, t_p[k])
        # u_p[N, k] = mu2.subs(t, t_p[k])
    (X_grid, Y_grid) = np.meshgrid(config.x_grid, config.time_grid, indexing="ij")
    return X_grid, Y_grid, U
