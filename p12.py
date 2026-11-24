import numpy as np
import matplotlib.pyplot as plt
rng = np.random.default_rng(0)  # random number generator for test matrices


# same function as in P3
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


def inverse(P):
    """Gauss-Jordan: row reduce (P | I) and take the right block."""
    n = len(P)
    # tol=0: the 21 x 21 Vandermonde matrix can have small pivots that are not zero
    R, pivots = rref(np.hstack([P, np.eye(n)]), tol=0)
    if pivots != list(range(n)):
        raise np.linalg.LinAlgError("matrix is singular")
    return R[:, n:]


def dual_basis(P):
    """Rows of the result are the dual basis of the columns of P."""
    return inverse(P)


def vandermonde(t, d):
    """V[k, j] = t_k ** j"""
    return np.array([[tk ** j for j in range(d + 1)] for tk in t], dtype=float)


def interpolate(f, t):
    """Coefficients of p = sum_k f(t_k) L_k. The columns of V^-1 are the L_k."""
    L = dual_basis(vandermonde(t, len(t) - 1))
    coef = np.zeros(len(t))
    for k in range(len(t)):
        coef = coef + f(t[k]) * L[:, k]
    return coef


def evaluate(coef, x):
    """Value of the polynomial with these coefficients (Horner's method)."""
    y = np.zeros_like(x)
    for a in coef[::-1]:
        y = y * x + a
    return y


def main():
    # random basis
    P = rng.normal(size=(5, 5))
    r = dual_basis(P)
    print("r_i b_j is the identity:", np.allclose(r @ P, np.eye(5)))
    print("agrees with np.linalg.inv:", np.allclose(r, np.linalg.inv(P)))
    x = rng.normal(size=5)
    x2 = np.zeros(5)
    for i in range(5):
        x2 = x2 + (r[i] @ x) * P[:, i]
    print("x = sum (r_i x) b_i:", np.allclose(x, x2))

    # Why are the L_j the columns of V^-1?
    # If c is the coefficient vector of p, then (V c)_k = p(t_k).
    # L_j has value 1 at t_j and 0 at the other nodes, so V c = e_j.
    # Then c = V^-1 e_j, and this is the j-th column of V^-1.
    L = dual_basis(vandermonde([-1, 0, 1], 2))
    print("V^-1 for nodes -1, 0, 1:\n", np.round(L, 10) + 0.0)
    # Example 8.9: x(x-1)/2, 1-x^2, x(x+1)/2
    example = np.array([[0, -0.5, 0.5], [1, 0, -1], [0, 0.5, 0.5]]).T
    print("agrees with Example 8.9:", np.allclose(L, example))

    # interpolation
    cases = [
        (lambda t: np.sin(3 * t), -1, 1, 9, "sin(3t), 9 nodes"),
        (lambda t: 1 / (1 + t ** 2), -5, 5, 21, "1/(1+t^2), 21 nodes"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, (f, a, b, n, name) in zip(axes, cases):
        t = np.linspace(a, b, n)
        coef = interpolate(f, t)
        s = np.linspace(a, b, 1001)
        p = evaluate(coef, s)
        error = np.abs(p - f(s))
        middle = np.abs(s) < (b - a) / 4
        print(name)
        print("  error at the nodes:", np.abs(evaluate(coef, t) - f(t)).max())
        print("  largest error:", error.max(), "at t =", s[error.argmax()])
        print("  largest error in the middle half:", error[middle].max())
        ax.plot(s, f(s), label="f")
        ax.plot(s, p, "--", label="p")
        ax.plot(t, f(t), "ko", ms=4, label="nodes")
        ax.set_title(name)
        ax.legend()
    fig.tight_layout()
    fig.savefig("p12.png", dpi=150)

    # What goes wrong: sin(3t) is fine everywhere. For 1/(1+t^2) the polynomial
    # is good in the middle, but near -5 and 5 it oscillates a lot between
    # the nodes. It still goes through all the nodes.
    # More equally spaced nodes do not help, the oscillations become bigger.


if __name__ == "__main__":
    main()
