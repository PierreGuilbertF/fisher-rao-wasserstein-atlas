"""Illustrative geodesics on the graph of a smooth bump.

The manifold is the embedded graph

    M = {(x, y, z(x, y)) : (x, y) in R^2},

with z a Gaussian bump centered at the origin. In (x, y) coordinates the
induced Riemannian metric is

    G = I + grad(z) grad(z)^T.

For a curve q(t) = (x(t), y(t)), its energy and length are

    E[q] = integral q'(t)^T G(q(t)) q'(t) dt,
    L[q] = integral sqrt(q'(t)^T G(q(t)) q'(t)) dt.

Two discrete energy minima are computed from opposite initial sides of the
bump. A third curve is deliberately arbitrary and tortuous.

The rendering uses one depth-sorted Poly3DCollection for BOTH the grey
surface and the trajectory tubes. This is intentional: Matplotlib's mplot3d
otherwise sorts each surface/tube collection as a whole, which can make tubes
vanish or, conversely, show through geometry that should occlude them.

Dependencies: numpy, scipy, matplotlib.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.optimize import minimize


# -----------------------------------------------------------------------------
# Paper-like plotting style
# -----------------------------------------------------------------------------
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 9,
        "axes.titlesize": 9,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "mathtext.fontset": "cm",
        "axes.linewidth": 0.7,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.04,
    }
)

# Muted, paper-friendly trajectory colors.
COL_GLOBAL = np.array([0.30, 0.41, 0.52])
COL_LOCAL = np.array([0.52, 0.34, 0.30])
COL_ARBITRARY = np.array([0.36, 0.45, 0.33])
COL_ENDPOINT = "0.18"


# -----------------------------------------------------------------------------
# Surface: deliberately modest bump around (0, 0)
# -----------------------------------------------------------------------------
HEIGHT = 1.05
SIGMA_X = 1.05
SIGMA_Y = 1.05


def z_surface(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Height z(x,y) of the graph manifold."""
    return HEIGHT * np.exp(-((x / SIGMA_X) ** 2 + (y / SIGMA_Y) ** 2))


def grad_z(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Analytic gradient of z(x,y)."""
    z = z_surface(x, y)
    zx = -2.0 * x * z / SIGMA_X**2
    zy = -2.0 * y * z / SIGMA_Y**2
    return zx, zy


# -----------------------------------------------------------------------------
# Induced metric, discrete energy, and discrete length
# -----------------------------------------------------------------------------
def segment_metric_sq(q_left: np.ndarray, q_right: np.ndarray) -> np.ndarray:
    """Squared Riemannian length of parameter-space segments."""
    dq = q_right - q_left
    mid = 0.5 * (q_left + q_right)
    zx, zy = grad_z(mid[:, 0], mid[:, 1])
    dz_linearized = zx * dq[:, 0] + zy * dq[:, 1]
    return dq[:, 0] ** 2 + dq[:, 1] ** 2 + dz_linearized**2


def discrete_energy_from_path(q: np.ndarray) -> float:
    """Midpoint discretization of integral g_q(q',q') dt on [0,1]."""
    n = len(q)
    dt = 1.0 / (n - 1)
    ds2 = segment_metric_sq(q[:-1], q[1:])
    return float(np.sum(ds2) / dt)


def discrete_length(q: np.ndarray) -> float:
    """Midpoint discretization of Riemannian length."""
    ds2 = segment_metric_sq(q[:-1], q[1:])
    return float(np.sum(np.sqrt(ds2)))


def unpack_path(interior: np.ndarray, q0: np.ndarray, q1: np.ndarray) -> np.ndarray:
    pts = interior.reshape(-1, 2)
    return np.vstack((q0, pts, q1))


def energy_objective(interior: np.ndarray, q0: np.ndarray, q1: np.ndarray) -> float:
    return discrete_energy_from_path(unpack_path(interior, q0, q1))


# -----------------------------------------------------------------------------
# Endpoints and initial curves
# -----------------------------------------------------------------------------
X0 = np.array([-2.40, -1.00])
X1 = np.array([+2.40, +0.30])
N_PATH = 55


def straight_path(q0: np.ndarray, q1: np.ndarray, n: int) -> np.ndarray:
    s = np.linspace(0.0, 1.0, n)
    return (1.0 - s)[:, None] * q0 + s[:, None] * q1


def side_normal(q0: np.ndarray, q1: np.ndarray) -> np.ndarray:
    d = q1 - q0
    return np.array([-d[1], d[0]]) / np.linalg.norm(d)


def tangent_unit(q0: np.ndarray, q1: np.ndarray) -> np.ndarray:
    d = q1 - q0
    return d / np.linalg.norm(d)


def bowed_initial_path(q0: np.ndarray, q1: np.ndarray, n: int, amplitude: float) -> np.ndarray:
    s = np.linspace(0.0, 1.0, n)
    q = straight_path(q0, q1, n)
    q += amplitude * np.sin(np.pi * s)[:, None] * side_normal(q0, q1)
    return q


def arbitrary_path(q0: np.ndarray, q1: np.ndarray, n: int) -> np.ndarray:
    """Tortuous deterministic curve, intentionally not optimized."""
    s = np.linspace(0.0, 1.0, n)
    q = straight_path(q0, q1, n)
    nvec = side_normal(q0, q1)
    tvec = tangent_unit(q0, q1)

    rng = np.random.default_rng(12)
    phases = rng.uniform(0.0, 2.0 * np.pi, size=4)
    amps = np.array([0.17, 0.12, 0.075, 0.045])
    wiggle = sum(
        a * np.sin(2.0 * np.pi * k * s + phi)
        for a, k, phi in zip(amps, [1, 2, 4, 6], phases)
    )

    # Keep it on one side of the summit, but make it visibly irregular.
    envelope = np.sin(np.pi * s) ** 1.05
    side_bias = -0.92 * np.sin(np.pi * s)
    tangential_shift = 0.12 * envelope * np.sin(5.5 * np.pi * s + 0.25)

    q += (side_bias + envelope * wiggle)[:, None] * nvec
    q += tangential_shift[:, None] * tvec
    return q


# -----------------------------------------------------------------------------
# Numerical geodesics
# -----------------------------------------------------------------------------
def optimize_energy_path(initial: np.ndarray) -> tuple[np.ndarray, object]:
    q0, q1 = initial[0], initial[-1]
    x_init = initial[1:-1].ravel()
    bounds = [(-3.35, 3.35)] * len(x_init)

    result = minimize(
        energy_objective,
        x_init,
        args=(q0, q1),
        method="L-BFGS-B",
        bounds=bounds,
        options={
            "maxiter": 2200,
            "maxfun": 60000,
            "ftol": 1e-12,
            "gtol": 1e-6,
            "maxls": 50,
        },
    )
    return unpack_path(result.x, q0, q1), result


def side_score(q: np.ndarray, q0: np.ndarray, q1: np.ndarray) -> float:
    base = straight_path(q0, q1, len(q))
    return float(np.mean((q - base) @ side_normal(q0, q1)))


def distinct_geodesics(q0: np.ndarray, q1: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (shortest_found, longer_local) from opposite sides of the bump."""
    seeds = np.linspace(-1.80, 1.80, 9)
    candidates = []

    for amp in seeds:
        init = bowed_initial_path(q0, q1, N_PATH, float(amp))
        q, result = optimize_energy_path(init)
        candidates.append(
            {
                "q": q,
                "length": discrete_length(q),
                "energy": discrete_energy_from_path(q),
                "side": side_score(q, q0, q1),
                "success": result.success,
            }
        )

    negative = [c for c in candidates if c["side"] < -0.08]
    positive = [c for c in candidates if c["side"] > +0.08]
    if not negative or not positive:
        raise RuntimeError("Could not recover two distinct geodesic basins.")

    pair = [
        min(negative, key=lambda c: c["energy"]),
        min(positive, key=lambda c: c["energy"]),
    ]
    pair.sort(key=lambda c: c["length"])
    global_candidate, longer_local = pair

    print("\nNumerical paths on the bump manifold")
    print("------------------------------------")
    print(
        f"Shortest geodesic candidate: L = {global_candidate['length']:.5f}, "
        f"E = {global_candidate['energy']:.5f}, side = {global_candidate['side']:+.3f}"
    )
    print(
        f"Longer local geodesic:       L = {longer_local['length']:.5f}, "
        f"E = {longer_local['energy']:.5f}, side = {longer_local['side']:+.3f}"
    )
    return global_candidate["q"], longer_local["q"]


# -----------------------------------------------------------------------------
# 3D geometry helpers
# -----------------------------------------------------------------------------
def light_vector(azdeg: float = 320.0, altdeg: float = 34.0) -> np.ndarray:
    az = np.deg2rad(azdeg)
    alt = np.deg2rad(altdeg)
    v = np.array([np.cos(alt) * np.cos(az), np.cos(alt) * np.sin(az), np.sin(alt)])
    return v / np.linalg.norm(v)


def surface_polygons(n_grid: int = 88):
    """Return surface quads and light-grey facecolors, with no mesh edges."""
    grid = np.linspace(-3.0, 3.0, n_grid)
    xx, yy = np.meshgrid(grid, grid)
    zz = z_surface(xx, yy)
    L = light_vector()

    polys = []
    colors = []
    for i in range(n_grid - 1):
        for j in range(n_grid - 1):
            polys.append(
                [
                    (xx[i, j], yy[i, j], zz[i, j]),
                    (xx[i, j + 1], yy[i, j + 1], zz[i, j + 1]),
                    (xx[i + 1, j + 1], yy[i + 1, j + 1], zz[i + 1, j + 1]),
                    (xx[i + 1, j], yy[i + 1, j], zz[i + 1, j]),
                ]
            )

            xc = 0.25 * (xx[i, j] + xx[i, j + 1] + xx[i + 1, j + 1] + xx[i + 1, j])
            yc = 0.25 * (yy[i, j] + yy[i, j + 1] + yy[i + 1, j + 1] + yy[i + 1, j])
            zx, zy = grad_z(np.asarray(xc), np.asarray(yc))
            normal = np.array([-float(zx), -float(zy), 1.0])
            normal /= np.linalg.norm(normal)
            diffuse = max(0.0, float(np.dot(normal, L)))

            # Light overall grey with enough directional shading to read depth.
            grey = 0.74 + 0.18 * diffuse
            colors.append((grey, grey, grey, 1.0))

    return polys, colors


def lift_to_surface(q: np.ndarray, lift: float = 0.034) -> np.ndarray:
    """Map q=(x,y) onto the graph, slightly lifted to carry a small tube."""
    x, y = q[:, 0], q[:, 1]
    z = z_surface(x, y) + lift
    return np.column_stack((x, y, z))


def tube_vertices(q: np.ndarray, radius: float = 0.030, n_theta: int = 14) -> np.ndarray:
    """Return vertices of a small 3D tube around the lifted space curve."""
    c = lift_to_surface(q, lift=radius * 1.12)
    tang = np.gradient(c, axis=0)
    tang /= np.linalg.norm(tang, axis=1, keepdims=True)

    normals = np.zeros_like(c)
    binormals = np.zeros_like(c)
    ref_z = np.array([0.0, 0.0, 1.0])
    ref_y = np.array([0.0, 1.0, 0.0])

    for i, t in enumerate(tang):
        ref = ref_z if abs(np.dot(t, ref_z)) < 0.90 else ref_y
        n = np.cross(t, ref)
        n /= np.linalg.norm(n)
        b = np.cross(t, n)
        b /= np.linalg.norm(b)
        normals[i] = n
        binormals[i] = b

    theta = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    return (
        c[:, None, :]
        + radius * np.cos(theta)[None, :, None] * normals[:, None, :]
        + radius * np.sin(theta)[None, :, None] * binormals[:, None, :]
    )


def tube_polygons(q: np.ndarray, base_color: np.ndarray, radius: float = 0.030, n_theta: int = 14):
    """Return tube quads and softly shaded facecolors."""
    tube = tube_vertices(q, radius=radius, n_theta=n_theta)
    centers = lift_to_surface(q, lift=radius * 1.12)
    L = light_vector()
    polys = []
    colors = []

    for i in range(len(q) - 1):
        for j in range(n_theta):
            j2 = (j + 1) % n_theta
            poly = [tube[i, j], tube[i + 1, j], tube[i + 1, j2], tube[i, j2]]
            polys.append(poly)

            face_center = 0.25 * np.sum(np.asarray(poly), axis=0)
            curve_center = 0.5 * (centers[i] + centers[i + 1])
            normal = face_center - curve_center
            normal /= max(np.linalg.norm(normal), 1e-12)
            diffuse = max(0.0, float(np.dot(normal, L)))
            factor = 0.82 + 0.18 * diffuse
            rgb = np.clip(base_color * factor, 0.0, 1.0)
            colors.append((*rgb, 1.0))

    return polys, colors


def scene_collection(q_arbitrary: np.ndarray, q_local: np.ndarray, q_global: np.ndarray):
    """Build a single collection so surface and tubes share one depth sort."""
    polygons, colors = surface_polygons(n_grid=88)

    for q, color, radius in (
        (q_arbitrary, COL_ARBITRARY, 0.030),
        (q_local, COL_LOCAL, 0.032),
        (q_global, COL_GLOBAL, 0.032),
    ):
        p, c = tube_polygons(q, color, radius=radius, n_theta=14)
        polygons.extend(p)
        colors.extend(c)

    return Poly3DCollection(
        polygons,
        facecolors=colors,
        edgecolors="none",
        linewidths=0.0,
        antialiased=False,
        zsort="average",
    )


# -----------------------------------------------------------------------------
# Main plot
# -----------------------------------------------------------------------------
def main() -> None:
    q_global, q_local = distinct_geodesics(X0, X1)
    q_arbitrary = arbitrary_path(X0, X1, N_PATH)

    print(
        f"Arbitrary curve:             L = {discrete_length(q_arbitrary):.5f}, "
        f"E = {discrete_energy_from_path(q_arbitrary):.5f}"
    )

    fig = plt.figure(figsize=(6.6, 4.6))
    ax = fig.add_subplot(111, projection="3d")

    # One depth-sorted collection gives the tubes proper surface occlusion.
    ax.add_collection3d(scene_collection(q_arbitrary, q_local, q_global))

    # Endpoints
    endpoint_z = z_surface(np.array([X0[0], X1[0]]), np.array([X0[1], X1[1]])) + 0.045
    ax.scatter(
        [X0[0], X1[0]],
        [X0[1], X1[1]],
        endpoint_z,
        s=18,
        color=COL_ENDPOINT,
        depthshade=True,
    )
    ax.text(X0[0] - 0.18, X0[1] - 0.14, endpoint_z[0] + 0.04, r"$x_0$", color=COL_ENDPOINT)
    ax.text(X1[0] + 0.06, X1[1] + 0.02, endpoint_z[1] + 0.04, r"$x_1$", color=COL_ENDPOINT)

    ax.set_title(r"Geodesics on the graph manifold $M=\{(x,y,z(x,y))\}$", pad=9)
    ax.set_xlabel(r"$x$", labelpad=3)
    ax.set_ylabel(r"$y$", labelpad=3)
    ax.set_zlabel(r"$z(x,y)$", labelpad=4)

    ax.set_xlim(-3.0, 3.0)
    ax.set_ylim(-3.0, 3.0)
    ax.set_zlim(0.0, HEIGHT * 1.30)

    # Still clearly perspective, but high enough that all three paths can be
    # read without flattening the bump into a top view.
    ax.set_box_aspect((1.34, 1.18, 0.58))
    ax.set_proj_type("persp", focal_length=0.96)
    ax.view_init(elev=36, azim=19)

    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((1.0, 1.0, 1.0, 1.0))
        axis.pane.set_edgecolor((0.89, 0.89, 0.89, 1.0))
    ax.grid(False)

    handles = [
        Line2D([0], [0], color=COL_ARBITRARY, lw=1.8, label="arbitrary curve"),
        Line2D([0], [0], color=COL_LOCAL, lw=1.8, label=r"local energy minimum (longer)"),
        Line2D([0], [0], color=COL_GLOBAL, lw=1.8, label=r"shortest geodesic"),
    ]
    ax.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.02, 0.98),
        frameon=True,
        framealpha=0.97,
        borderpad=0.55,
        handlelength=2.5,
    )

    out = Path(__file__).resolve().parent
    fig.savefig(out / "output/geodesic_bump.pdf")
    fig.savefig(out / "output/geodesic_bump.png", dpi=220)
    plt.show()


if __name__ == "__main__":
    main()
