import numpy as np
rng = np.random.default_rng(0)  # random number generator for test matrices


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


def null_space_basis(A):
    """One basis vector for each free variable."""
    R, pivots = rref(A)
    n = R.shape[1]
    basis = []
    for f in range(n):
        if f in pivots:
            continue
        v = np.zeros(n)
        v[f] = 1
        for row, c in enumerate(pivots):
            v[c] = -R[row, f]
        basis.append(v)
    return basis


def col_space_basis(A):
    """Pivot columns of the original matrix A."""
    A = np.array(A, dtype=float)
    return [A[:, c] for c in rref(A)[1]]


def main():
    A = np.array([[1, 2, -1], [3, -1, 4]])
    F = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    expected = [(A, [[1, 0, 1], [0, 1, -1]]), (F, [[1, 0, -1], [0, 1, 2], [0, 0, 0]])]
    for M, answer in expected:
        R, pivots = rref(M)
        # + 0.0 turns -0. into 0. when printing
        print(np.round(R, 10) + 0.0, "pivots:", pivots, "correct:", np.allclose(R, answer))
    print("null(F):", null_space_basis(F))
    print("col(F):", col_space_basis(F))

    ok = True
    for _ in range(20):
        A = rng.normal(size=(5, 7))
        basis = null_space_basis(A)
        ok = ok and all(np.allclose(A @ v, 0) for v in basis)
        ok = ok and rank(A) + len(basis) == 7
        ok = ok and rank(A) == np.linalg.matrix_rank(A)
    print("20 random 5x7 matrices:", ok)

    A = rng.normal(size=(6, 4)) @ rng.normal(size=(4, 8))
    basis = null_space_basis(A)
    print("6x8 matrix: rank", rank(A), ", dim of null space", len(basis))
    print("A v = 0 for all basis vectors:", all(np.allclose(A @ v, 0) for v in basis))


if __name__ == "__main__":
    main()
