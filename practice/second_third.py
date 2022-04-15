from math import sqrt
import numpy as np
from numpy.linalg import norm
from numpy.linalg import inv
from numpy.linalg import det
from numpy.linalg import solve
import matplotlib.pyplot as plt


# spectral conditionality criterion
def cond_s(A):
    return round(norm(A) * norm(inv(A)), 2)


# volumetric criterion
def cond_v(A):
    dim = A.shape[0]
    numerator = 0
    for n in range(0, dim):
        row_sum = 0
        for m in range(0, dim):
            row_sum += A.item(n, m) ** 2
        numerator += sqrt(row_sum)
    return round(numerator / abs(det(A)), 2)


# angular criterion
def cond_a(A):
    A = np.matrix(A)
    dim = A.shape[0]
    C = A.I
    cond = 0
    for i in range(0, dim):
        cond = max(cond, norm(A[i, :]) * norm([C[:, i]]))
    return round(cond, 2)


def element(i, j, n):
    b = lambda k: k + 1
    a = lambda k: k / 2
    return a(i + 1) ** b(j + 1)


def create_vander(n):
    A = np.zeros((n, n))
    for i in range(0, n):
        for j in range(0, n):
            A[i][j] = element(i, j, n)
    return A


def dim(A):
    return A.shape[0]


def rand_x(n, max_value=1000):
    return np.random.randint(-max_value, max_value, n)


def lu_decomposition(A, b):
    n = dim(A)
    a_i = A
    b_i = b
    L = np.eye(n)
    for j in range(0, n - 1):
        m_i = np.eye(n)
        for i in range(j + 1, n):
            m_i[i][j] = -a_i[i][j] / a_i[j][j]
        L = L @ (-m_i + 2 * np.eye(n))
        a_i = m_i @ a_i
        b_i = m_i @ b_i
    U = a_i
    return L, U, b_i


def solve_with_LU_decomposition(A, b):
    L, U, b1 = lu_decomposition(A, b)
    x = b

    for i in range(dim(A)):
        x[i] /= L[i][i]
        L[i][i] = 1
        for k in range(i + 1, dim(A)):
            x[k] -= L[k][i] * x[i]
            L[k][i] = 0

    for i in range(dim(A) - 1, -1, -1):
        x[i] /= U[i][i]
        U[i][i] = 1
        for k in range(0, i):
            x[k] -= U[k][i] * x[i]
            U[k][i] = 0
    return x


def print_conds(matrices):
    for [A, matrix_name] in matrices:
        # A = np.matrix(A)
        print(
            f"cond_s({matrix_name}) = {cond_s(A):.2e}, cond_v({matrix_name}) = {cond_v(A):.2e}, cond_a({matrix_name}) = {cond_a(A):.2e}")


def hilbert(n, precision=16):
    return np.fromfunction(lambda i, j: np.round(1 / (i + j + 1), precision), (n, n))


def find_alpha(A, x_p, with_conds="false"):
    alphas = np.logspace(-12, -1, 300)
    errors = []
    conds_s = []
    conds_v = []
    conds_a = []
    b = A @ x_p

    for alpha in alphas:
        A_tilda = A + alpha * np.eye(dim(A))
        x_alpha = solve(A_tilda, b)
        error = norm(x_p - x_alpha)
        errors.append(error)
        if with_conds == "true":
            conds_s.append(cond_s(A_tilda))
            conds_v.append(cond_v(A_tilda))
            conds_a.append(cond_a(A_tilda))

    plt.xscale('log')
    plt.yscale('log')

    random_xs = []
    for i in range(2):
        random_xs.append(rand_x(dim(A), 10))

    plt.plot(alphas, errors, color='orange', label="Погрешность")
    if with_conds == "true":
        plt.plot(alphas, conds_s, color='y', label="Спектральный критерий")
        plt.plot(alphas, conds_v, color='r', label="Объёмный критерий")
        plt.plot(alphas, conds_a, color='green', label="Угловой критерий")

    plt.grid(linestyle='--')
    min_index, min_value = min(enumerate(errors), key=lambda p: p[1])
    best_alpha = alphas[min_index]
    plt.hlines(y=min_value, xmin=alphas[0], xmax=alphas[len(alphas) - 1], colors='black', linestyles='--', lw=1,
               label=f"min error = {min_value:.2e}")
    plt.vlines(x=best_alpha, ymin=min_value - 1, ymax=max(errors), colors='grey', linestyles='--', lw=1,
               label=f"best alpha = {alphas[min_index]:.2e}")

    plt.ylabel("Порядок велчин")
    plt.xlabel("alpha")

    for x_r in random_xs:
        b_r = A @ x_r
        x = np.linalg.solve(A + best_alpha * np.eye(dim(A)), b_r)
        print("error on random x:", norm(x - x_r))

    plt.legend()
    plt.show()


def example_with_vander(n, x):
    print(f"Example with vander matrix")
    A = create_vander(n)
    b = A @ x

    L, U, b1 = lu_decomposition(A, b)
    without_lu = norm(solve(U, b1) - x)

    with_lu = norm(solve_with_LU_decomposition(A, b) - x)
    print_conds([[A, "Wander"], [L, "L"], [U, "U"]])
    print(f"Solve with LU = {with_lu}, with b = {without_lu}, diff = {with_lu - without_lu}")


def example_with_hilbert(n, x):
    print(f"Example with Hilbert matrix")
    # change to 16
    A = hilbert(n, 8)
    b = A @ x

    L, U, b1 = lu_decomposition(A, b)
    with_lu = norm(solve_with_LU_decomposition(A, b) - x)
    without_lu = norm(solve(U, b1) - x)
    print_conds([[A, "H"], [L, "L"], [U, "U"]])
    print(f"Solve with LU = {with_lu}, with b = {without_lu}, diff = {with_lu - without_lu}")


def QR_decomposition(A, b):
    b_i = b
    A_i = A
    Q = np.eye(dim(A))
    U = lambda w_i, i : np.eye(dim(A) - i) - 2 * w_i.T @ w_i.conjugate()
    for i in range(0, dim(A)-1):
        a_i = A_i[i:, i]
        e_i = np.zeros(dim(A) - i)
        e_i[0] = 1

        w_i = (a_i - norm(a_i) * e_i)
        w_i = np.array([w_i/norm(w_i)])

        U_i = np.block([[np.eye(i), np.zeros((i, dim(A) - i))],
                        [np.zeros((dim(A) - i, i)), U(w_i, i)]])
        b_i = U_i @ b_i
        A_i = U_i @ A_i
        Q = Q @ U_i

    R = A_i
    return Q, R, b_i


if __name__ == '__main__':
    # n = 5
    # x = rand_x(n)
    # print(f"x = {x}")
    # example_with_vander(n, x)
    # print("\nx = [1 ... 1]")
    # example_with_vander(n, np.ones((n, 1)))
    #
    #
    # print("")
    # n = 15
    # x = np.random.randint(-1000, 1000, n)
    # print(f"x = {x}")
    # example_with_hilbert(n, np.random.randint(-10, 10, n))
    # print("\nx = [1 .. 1]")
    # example_with_hilbert(n, np.ones((n, 1)))
    #
    H = hilbert(15)
    b = H @ np.ones((dim(H), 1))
    L, U, b_i = lu_decomposition(H, b)
    print_conds([[H, "A"], [L, "L"], [U, "U"]])
    find_alpha(H, b, "true")
    # find_alpha(H, b, "false")
    #
    # H = hilbert(3)
    # b = H @ np.ones((dim(H), 1))
    # L, U, b_i = lu_decomposition(H, b)
    # print_conds([[H, "A"], [L, "L"], [U, "U"]])
    # find_alpha(H, b, "true")
    # find_alpha(H, b, "false")
    #
    # Third
    n = 17
    # H = hilbert(n)
    x = rand_x(n)
    # Q, R, b_i = QR_decomposition(H, H @ x)
    # print_conds([[H, "H"], [Q, "Q"], [R, "R"]])
    # print(norm(x - solve(R, b_i)))
    # #
    # Compare to
    H = hilbert(n)
    b = H @ x
    L, U, b_i = lu_decomposition(H, b)
    print_conds([[H, "A"], [L, "L"], [U, "U"]])
    without_lu = norm(solve(U, b_i) - x)
    print(without_lu)

    #
    # Example:

    # n = 8
    # x = rand_x(n)
    # print(f"x = {x}")
    # example_with_vander(n, x)
    # print(f"\n(QR) x = {x}")
    # H = create_vander(n)
    # Q, R, b_i = QR_decomposition(H, H @ x)
    # print_conds([[H, "H"], [Q, "Q"], [R, "R"]])
    # print(f"Error = {norm(x - solve(R, b_i))}")
    #
    # x = np.ones((n, 1))
    # print("\n\nx = [1 ... 1]")
    # example_with_vander(n, x)
    #
    # print("\n(QR) x = [1 .. 1]")

    # # Next example
    #
    # n = 15
    # x = rand_x(n)
    # print(f"x = {x}")
    # example_with_vander(n, x)
    # print(f"\n(QR) x = {x}")
    # H = create_vander(n)
    # Q, R, b_i = QR_decomposition(H, H @ x)
    # print_conds([[H, "Wander"], [Q, "Q"], [R, "R"]])
    # print(f"Error = {norm(x - solve(R, b_i))}")
    #
    # x = np.ones((n, 1))
    # print("\n\nx = [1 ... 1]")
    # example_with_vander(n, x)
    #
    # print("\n(QR) x = [1 .. 1]")
    # Q, R, b_i = QR_decomposition(H, H @ x)
    # print_conds([[H, "Wander"], [Q, "Q"], [R, "R"]])
    # print(f"Error = {norm(x - solve(R, b_i))}")


