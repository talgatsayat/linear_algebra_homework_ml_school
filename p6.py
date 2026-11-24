import numpy as np
import matplotlib.pyplot as plt
rng = np.random.default_rng(0)  # random number generator for test matrices

SQUARE = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
LETTER_F = np.array([[0.2, 0.1], [0.4, 0.1], [0.4, 0.45], [0.65, 0.45], [0.65, 0.6],
                     [0.4, 0.6], [0.4, 0.75], [0.8, 0.75], [0.8, 0.9], [0.2, 0.9]])

MATRICES = {
    "stretch": np.array([[2, 0], [0, 1]]),
    "rotation by 90": np.array([[0, -1], [1, 0]]),
    "reflection in the x-axis": np.array([[1, 0], [0, -1]]),
    "shear": np.array([[1, 1], [0, 1]]),
    "singular": np.array([[1, 2], [2, 4]]),
    "Figure 4": np.array([[3, 1], [1, 2]]),
}


def plot_transformation(A, ax=None, title=""):
    """Draw the unit square, a grid and the letter F, before and after A."""
    A = np.array(A, dtype=float)
    if ax is None:
        ax = plt.gca()
    ax.axhline(0, color="gray", lw=0.8)
    ax.axvline(0, color="gray", lw=0.8)

    # before
    ax.fill(*SQUARE.T, fill=False, edgecolor="black", ls="--")
    ax.fill(*LETTER_F.T, color="lightgray")

    # after: one point per row, so we multiply by A.T on the right
    ax.fill(*(SQUARE @ A.T).T, facecolor="lavender", edgecolor="blue", alpha=0.7)
    ax.fill(*(LETTER_F @ A.T).T, color="darkorange", alpha=0.9)

    # grid lines before and after
    for t in (0.25, 0.5, 0.75):
        for line in (np.array([[t, 0], [t, 1]]), np.array([[0, t], [1, t]])):
            ax.plot(*line.T, color="gray", lw=0.6, ls=":")
            ax.plot(*(line @ A.T).T, color="blue", lw=0.6)

    # arrows: T(e1) is red, T(e2) is green
    for column, color in ((A[:, 0], "red"), (A[:, 1], "green")):
        ax.annotate("", xy=column, xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.8))

    points = np.vstack([SQUARE, SQUARE @ A.T])
    ax.set_xlim(points[:, 0].min() - 0.4, points[:, 0].max() + 0.4)
    ax.set_ylim(points[:, 1].min() - 0.4, points[:, 1].max() + 0.4)
    ax.set_aspect("equal")
    ax.set_title(title)


def signed_area(P):
    """Shoelace formula for a polygon with one corner per row."""
    total = 0.0
    k = len(P)
    for i in range(k):
        x1, y1 = P[i]
        x2, y2 = P[(i + 1) % k]
        total += x1 * y2 - x2 * y1
    return total / 2


def main():
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), layout="constrained")
    for ax, (name, A) in zip(axes.flat, MATRICES.items()):
        plot_transformation(A, ax, name)
    fig.savefig("p6.png", dpi=150, bbox_inches="tight", pad_inches=0.3)

    for name, A in MATRICES.items():
        area = signed_area(SQUARE @ A.T)
        det = np.linalg.det(A)
        print(f"{name:26s} area = {area:5.1f}  det = {det:5.1f}  equal: {np.isclose(area, det)}")


if __name__ == "__main__":
    main()
