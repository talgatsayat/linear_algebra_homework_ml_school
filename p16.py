import time
import numpy as np
rng = np.random.default_rng(0)  # random number generator for test matrices


def lu(A):
    """Return L, U with A = L @ U (no row swaps)."""
    U = np.array(A, dtype=float)
    n = len(U)
    L = np.eye(n)
    for k in range(n - 1):
        if U[k, k] == 0:
            raise ValueError("zero pivot, row swaps are needed")
        L[k + 1:, k] = U[k + 1:, k] / U[k, k]
        U[k + 1:, k] = 0
        # the rest of the matrix becomes its Schur complement
        U[k + 1:, k + 1:] -= L[k + 1:, k:k + 1] * U[k:k + 1, k + 1:]
    return L, U


def solve_lu(L, U, b):
    """Solve L U x = b by forward and back substitution."""
    n = len(b)
    y = np.zeros(n)
    for i in range(n):
        y[i] = b[i] - L[i, :i] @ y[:i]
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        if U[i, i] == 0:
            raise ValueError("matrix is singular")
        x[i] = (y[i] - U[i, i + 1:] @ x[i + 1:]) / U[i, i]
    return x


def inverse(A):
    """Solve A x = e_i for every i."""
    L, U = lu(A)
    return np.column_stack([solve_lu(L, U, e) for e in np.eye(len(A))])


def sherman_morrison(M, x):
    """Inverse of S + x x^T, where M is the inverse of S (S is symmetric)."""
    Mx = M @ x
    return M - np.outer(Mx, Mx) / (1 + x @ Mx)


def check_lu():
    A = np.array([[2, 1, 1], [4, 3, 3], [8, 7, 9]])
    L, U = lu(A)
    print("L =\n", L, "\nU =\n", U)
    ok_L = np.allclose(L, [[1, 0, 0], [2, 1, 0], [4, 3, 1]])
    ok_U = np.allclose(U, [[2, 1, 1], [0, 1, 1], [0, 0, 2]])
    print("agrees with Example 6.107:", ok_L and ok_U)
    print("x for b = (5, 13, 31):", solve_lu(L, U, np.array([5, 13, 31])))

    ok_lu = ok_solve = True
    for _ in range(20):
        A, b = rng.normal(size=(6, 6)), rng.normal(size=6)
        L, U = lu(A)
        ok_lu = ok_lu and np.allclose(L @ U, A)
        ok_solve = ok_solve and np.allclose(solve_lu(L, U, b), np.linalg.solve(A, b))
    print("L @ U = A for random matrices:", ok_lu)
    print("solve_lu agrees with numpy:", ok_solve)


def small_pivot():
    A = np.array([[1e-17, 1], [1, 1]])
    b = np.array([1.0, 2.0])
    print("no swap:  ", solve_lu(*lu(A), b))
    print("with swap:", solve_lu(*lu(A[::-1]), b[::-1]))
    print("numpy:    ", np.linalg.solve(A, b))
    # Without the swap we get (0, 1), which is wrong. The multiplier is 10^17,
    # so U22 = 1 - 10^17 and y2 = 2 - 10^17. Both are rounded to -10^17, because
    # there are only 16 digits, and the 1 and the 2 are lost. Then x2 = 1 and
    # x1 = (1 - x2) / 10^-17 = 0.
    # After the swap the multiplier is 10^-17 and nothing important is lost.


def updates():
    X = rng.normal(size=(200, 50))
    new = rng.normal(size=(100, 50))
    M = inverse(X.T @ X)

    start = time.perf_counter()
    for x in new:
        M = sherman_morrison(M, x)
    t_update = time.perf_counter() - start

    # from scratch, only once for the final matrix
    X_all = np.vstack([X, new])
    start = time.perf_counter()
    M_final = inverse(X_all.T @ X_all)
    t_once = time.perf_counter() - start

    # from scratch after every new row, with my function and with numpy
    times = []
    for inv in (inverse, np.linalg.inv):
        start = time.perf_counter()
        S = X.T @ X
        for x in new:
            S = S + np.outer(x, x)
            inv(S)
        times.append(time.perf_counter() - start)

    print("same inverse:", np.allclose(M, M_final), " max difference:", np.abs(M - M_final).max())
    print("agrees with numpy:", np.allclose(M, np.linalg.inv(X_all.T @ X_all)))
    print(f"100 Sherman-Morrison updates:         {t_update * 1000:8.2f} ms")
    print(f"from scratch once at the end:         {t_once * 1000:8.2f} ms")
    print(f"from scratch after each row:          {times[0] * 1000:8.2f} ms")
    print(f"same with np.linalg.inv:              {times[1] * 1000:8.2f} ms")
    # One update costs about n^2 operations and one inverse about n^3, so the
    # update is much better if we need the inverse after every new row.


if __name__ == "__main__":
    check_lu()
    small_pivot()
    updates()
