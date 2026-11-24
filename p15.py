import time
import numpy as np
rng = np.random.default_rng(0)  # random number generator for test matrices


# same functions as in P3
def rref(A, tol=1e-10):
    """Return (R, pivots): the RREF of A and the list of pivot columns."""
    R = np.array(A, dtype=float)  # work on a copy of A
    m, n = R.shape
    pivots = []
    r = 0  # row where the next pivot will be
    for c in range(n):
        if r == m:
            break
        # row with the largest absolute value in column c
        p = r
        for i in range(r + 1, m):
            if abs(R[i, c]) > abs(R[p, c]):
                p = i
        if abs(R[p, c]) < tol:
            R[r:, c] = 0  # only rounding errors are left here
            continue
        R[[r, p]] = R[[p, r]]
        R[r] = R[r] / R[r, c]
        for i in range(m):
            if i != r:
                R[i] = R[i] - R[i, c] * R[r]
        pivots.append(c)
        r += 1
    return R, pivots


def rank(A):
    return len(rref(A)[1])


def rank_one(u, v, w):
    return np.einsum("i,j,k->ijk", u, v, w)


def reshaping():
    T = np.arange(24).reshape(2, 3, 4)
    flat = T.reshape(-1)
    ok = all(T[i, j, k] == flat[12 * i + 4 * j + k]
             for i in range(2) for j in range(3) for k in range(4))
    print("T[i,j,k] = flat[12i+4j+k]:", ok)
    for U in (T.reshape(2, 12), T.transpose(1, 0, 2).reshape(3, 8)):
        print("unfolding", U.shape, "\n", U)
        print("rank:", rank(U), " numpy:", np.linalg.matrix_rank(U))
    # Both ranks are 2. In the second one there are 3 rows, but the step from
    # one row to the next is always (4, ..., 4), so row 1 + row 3 = 2 * row 2.


def table_1():
    A, B = rng.normal(size=(2, 4, 4))
    x, y = rng.normal(size=(2, 4))
    checks = {
        "matrix product": (np.einsum("ij,jk->ik", A, B), A @ B),
        "trace": (np.einsum("ii->", A), np.trace(A)),
        "dot product": (np.einsum("i,i->", x, y), x @ y),
        "outer product": (np.einsum("i,j->ij", x, y), np.outer(x, y)),
        "transpose": (np.einsum("ij->ji", A), A.T),
        "bilinear form": (np.einsum("i,ij,j->", x, A, y), x @ A @ y),
    }
    # batches
    A, B = rng.normal(size=(2, 100, 4, 4))
    checks["batch of traces"] = (np.einsum("bii->b", A),
                                 [np.trace(A[b]) for b in range(100)])
    checks["batch of products"] = (np.einsum("bij,bjk->bik", A, B),
                                   [A[b] @ B[b] for b in range(100)])
    for name, (a, b) in checks.items():
        print(f"{name}: {np.allclose(a, b)}")


def kronecker():
    A, B, X = rng.normal(size=(3, 4)), rng.normal(size=(2, 5)), rng.normal(size=(4, 5))
    left = (A @ X @ B.T).reshape(-1)
    right = np.kron(A, B) @ X.reshape(-1)
    print("A X B^T = kron(A, B) X:", np.allclose(left, right))
    # For 100 x 100 matrices kron(A, B) is 10^4 x 10^4, so it stores 10^8 numbers
    # (800 MB), and kron @ x needs 10^8 multiplications.
    # A X B^T is two products of 100 x 100 matrices, so 2 * 100^3 = 2 * 10^6
    # multiplications, 50 times less. And we store only A, B, X: 3 * 10^4 numbers.
    n = 100
    print("kron stores", n ** 4, "numbers,", n ** 4 * 8 / 1e6, "MB")
    A, B, X = rng.normal(size=(3, n, n))
    start = time.perf_counter()
    A @ X @ B.T
    print("time of A X B^T:", time.perf_counter() - start)


def slices():
    u, v, w = rng.normal(size=3), rng.normal(size=4), rng.normal(size=5)
    T = rank_one(u, v, w)
    # slice number k is w_k * u v^T
    ok = all(np.allclose(T[:, :, k], w[k] * np.outer(u, v)) for k in range(5))
    print("slices are multiples of u v^T:", ok)


if __name__ == "__main__":
    reshaping()
    table_1()
    kronecker()
    slices()
