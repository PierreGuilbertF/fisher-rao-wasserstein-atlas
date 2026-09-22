#!/usr/bin/env python3
"""
Illustration of "On the moving density and conservation of mass."

This script creates a 3-panel figure showing:
  1) an initial 2D density p_0 with sampled points and a subset Omega,
  2) a comparison panel showing Omega and phi(Omega) side by side,
     with the same indexed particles and arrows between them,
  3) the pushforward density p_1 = phi_# p_0 with transformed samples
     and the transported subset phi(Omega).

The nonlinear map phi is chosen to be a genuine diffeomorphism with an
explicit inverse, so the pushforward density can be computed exactly.

Main identity illustrated:
    ∫_{Omega_t} p_t(y) dy = ∫_{Omega_0} p_0(x) dx

Outputs:
    reynolds_transport_probability.png
    reynolds_transport_probability.pdf
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch

SEED = 7
N_SAMPLES = 180

MEAN = np.array([-0.35, 0.25])
COV = np.array([[1.20, 0.48],
                [0.48, 0.92]])

A = 1.05
B = 1.10
LAMBDA = 0.66

XMIN, XMAX = -3.7, 3.8
YMIN, YMAX = -3.2, 3.5
N_GRID = 320

OMEGA_CENTER = np.array([-0.10, 0.10])
OMEGA_AXES = np.array([1.05, 0.78])

BLUE = "#4C78A8"
RED = "#C44E52"
DARK = "#333333"
LIGHT_GREY = "#B9B9B9"
FILL_RED_ALPHA = 0.20
SAMPLE_ALPHA = 0.78

OUTPUT_BASENAME = "reynolds_transport_probability"
DPI = 220


def gaussian_pdf_points(points, mean, cov):
    points = np.asarray(points)
    inv_cov = np.linalg.inv(cov)
    det_cov = np.linalg.det(cov)
    diff = points - mean
    quad = np.einsum("...i,ij,...j->...", diff, inv_cov, diff)
    norm = 1.0 / (2.0 * np.pi * np.sqrt(det_cov))
    return norm * np.exp(-0.5 * quad)


def gaussian_pdf_grid(X, Y, mean, cov):
    pts = np.stack([X, Y], axis=-1)
    return gaussian_pdf_points(pts, mean, cov)


def phi(points):
    points = np.asarray(points)
    x = points[..., 0]
    y = points[..., 1]
    u = x + A * np.sin(y)
    v = LAMBDA * y + B * np.tanh(u)
    return np.stack([u, v], axis=-1)


def phi_inverse(points):
    points = np.asarray(points)
    u = points[..., 0]
    v = points[..., 1]
    y = (v - B * np.tanh(u)) / LAMBDA
    x = u - A * np.sin(y)
    return np.stack([x, y], axis=-1)


def pushforward_density_grid(U, V):
    preimage = phi_inverse(np.stack([U, V], axis=-1))
    return gaussian_pdf_points(preimage, MEAN, COV) / LAMBDA


def omega_boundary(n=500):
    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=True)
    radial_mod = 1.0 + 0.16 * np.cos(3.0 * t - 0.45)
    x = OMEGA_CENTER[0] + OMEGA_AXES[0] * radial_mod * np.cos(t)
    y = OMEGA_CENTER[1] + OMEGA_AXES[1] * (1.0 + 0.10 * np.sin(2.0 * t + 0.4)) * np.sin(t)
    return np.column_stack([x, y])


def points_in_region(points, boundary):
    path = MplPath(boundary)
    return path.contains_points(points)


def style_axes(ax):
    for spine in ax.spines.values():
        spine.set_linewidth(0.9)
        spine.set_color(DARK)


def add_region_patch(ax, boundary, edgecolor=RED, facealpha=FILL_RED_ALPHA, lw=2.0):
    patch = PathPatch(
        MplPath(boundary),
        facecolor=edgecolor,
        edgecolor=edgecolor,
        lw=lw,
        alpha=facealpha
    )
    ax.add_patch(patch)
    ax.plot(boundary[:, 0], boundary[:, 1], color=edgecolor, lw=2.0, zorder=6)


def prepare_display_scene(boundary, points, box):
    xmin, xmax, ymin, ymax = box
    all_pts = boundary if points.size == 0 else np.vstack([boundary, points])

    src_min = all_pts.min(axis=0)
    src_max = all_pts.max(axis=0)
    src_center = 0.5 * (src_min + src_max)
    src_size = np.maximum(src_max - src_min, 1e-8)

    dst_center = np.array([0.5 * (xmin + xmax), 0.5 * (ymin + ymax)])
    dst_size = np.array([xmax - xmin, ymax - ymin])

    scale = 0.76 * min(dst_size[0] / src_size[0], dst_size[1] / src_size[1])

    boundary_new = (boundary - src_center) * scale + dst_center
    if points.size == 0:
        points_new = points.copy()
    else:
        points_new = (points - src_center) * scale + dst_center

    return boundary_new, points_new


def main():
    rng = np.random.default_rng(SEED)
    Xs = rng.multivariate_normal(MEAN, COV, size=N_SAMPLES)

    boundary = omega_boundary()
    boundary_phi = phi(boundary)

    inside = points_in_region(Xs, boundary)
    Xs_in = Xs[inside]
    Ys = phi(Xs)
    Ys_in = Ys[inside]

    x = np.linspace(XMIN, XMAX, N_GRID)
    y = np.linspace(YMIN, YMAX, N_GRID)
    X, Y = np.meshgrid(x, y)
    p0 = gaussian_pdf_grid(X, Y, MEAN, COV)

    omega_mask = points_in_region(np.column_stack([X.ravel(), Y.ravel()]), boundary).reshape(X.shape)
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    mass_left = float(np.sum(p0[omega_mask]) * dx * dy)

    transformed_cloud = np.vstack([Ys, boundary_phi])
    umin, vmin = transformed_cloud.min(axis=0) - np.array([0.8, 0.7])
    umax, vmax = transformed_cloud.max(axis=0) + np.array([0.8, 0.7])

    u = np.linspace(umin, umax, N_GRID)
    v = np.linspace(vmin, vmax, N_GRID)
    U, V = np.meshgrid(u, v)
    p1 = pushforward_density_grid(U, V)

    phi_mask = points_in_region(np.column_stack([U.ravel(), V.ravel()]), boundary_phi).reshape(U.shape)
    du = u[1] - u[0]
    dv = v[1] - v[0]
    mass_right = float(np.sum(p1[phi_mask]) * du * dv)

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10.5,
        "axes.titlesize": 12.0,
        "axes.labelsize": 10.8,
        "xtick.labelsize": 8.8,
        "ytick.labelsize": 8.8,
        "figure.titlesize": 16.5,
        "mathtext.fontset": "dejavusans",
    })

    fig = plt.figure(figsize=(16.5, 5.8))
    gs = fig.add_gridspec(
        1, 3,
        width_ratios=[1.05, 0.82, 1.12],
        left=0.045, right=0.98,
        bottom=0.16, top=0.84,
        wspace=0.18
    )

    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])

    levels0 = np.quantile(p0.ravel(), [0.55, 0.73, 0.84, 0.91, 0.96, 0.985])
    ax0.contourf(X, Y, p0, levels=np.r_[0.0, levels0, p0.max()], cmap="Blues", alpha=0.22)
    ax0.contour(X, Y, p0, levels=levels0, colors=BLUE, linewidths=1.2, alpha=0.88)

    ax0.scatter(Xs[:, 0], Xs[:, 1], s=18, color=DARK, alpha=0.42, zorder=3)
    ax0.scatter(Xs_in[:, 0], Xs_in[:, 1], s=22, color=RED, alpha=SAMPLE_ALPHA, zorder=4)

    add_region_patch(ax0, boundary)
    ax0.text(
        0.03, 0.965, r"$\Omega$",
        transform=ax0.transAxes, ha="left", va="top",
        fontsize=12.5, color=RED,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=1.5)
    )
    ax0.text(
        0.03, 0.05,
        r"sampled points $x_i$",
        transform=ax0.transAxes, ha="left", va="bottom",
        fontsize=9.6, color=DARK,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=1.5)
    )

    ax0.set_title(r"Initial density $p_0$ and subset $\Omega$")
    ax0.set_xlabel(r"$x_1$")
    ax0.set_ylabel(r"$x_2$")
    ax0.set_aspect("equal")
    style_axes(ax0)

    ax1.set_axis_off()
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)

    left_box = (0.05, 0.45, 0.15, 0.85)
    right_box = (0.55, 0.95, 0.15, 0.85)

    bL, pL = prepare_display_scene(boundary, Xs_in, left_box)
    bR, pR = prepare_display_scene(boundary_phi, Ys_in, right_box)

    ax1.plot([left_box[0], left_box[1], left_box[1], left_box[0], left_box[0]],
             [left_box[2], left_box[2], left_box[3], left_box[3], left_box[2]],
             color=LIGHT_GREY, lw=0.9)
    ax1.plot([right_box[0], right_box[1], right_box[1], right_box[0], right_box[0]],
             [right_box[2], right_box[2], right_box[3], right_box[3], right_box[2]],
             color=LIGHT_GREY, lw=0.9)

    ax1.fill(bL[:, 0], bL[:, 1], color=RED, alpha=0.18, zorder=1)
    ax1.plot(bL[:, 0], bL[:, 1], color=RED, lw=2.0, zorder=2)
    ax1.fill(bR[:, 0], bR[:, 1], color=RED, alpha=0.18, zorder=1)
    ax1.plot(bR[:, 0], bR[:, 1], color=RED, lw=2.0, zorder=2)

    ax1.scatter(pL[:, 0], pL[:, 1], s=20, color=RED, alpha=0.85, zorder=3)
    ax1.scatter(pR[:, 0], pR[:, 1], s=20, color=RED, alpha=0.85, zorder=3)

    if len(pL) > 0:
        n_arrows = min(10, len(pL))
        idx = np.linspace(0, len(pL) - 1, n_arrows, dtype=int)
        for k in idx:
            ax1.annotate(
                "",
                xy=(pR[k, 0], pR[k, 1]),
                xytext=(pL[k, 0], pL[k, 1]),
                arrowprops=dict(arrowstyle="->", lw=1.0, color=DARK, alpha=0.55),
                zorder=2
            )

    ax1.text(0.25, 0.90, r"$\Omega$", ha="center", va="bottom", fontsize=12.2, color=RED)
    ax1.text(0.75, 0.90, r"$\phi(\Omega)$", ha="center", va="bottom", fontsize=12.2, color=RED)
    ax1.text(0.50, 0.93, r"$\phi$", ha="center", va="bottom", fontsize=13.5, color=DARK)
    ax1.text(
        0.50, 0.875,
        r"$x_i\in\Omega \;\Longleftrightarrow\; \phi(x_i)\in\phi(\Omega)$",
        ha="center", va="bottom", fontsize=10.4, color=DARK
    )
    ax1.text(
        0.50, 0.05,
        rf"$\int_\Omega p_0(x)\,dx \approx {mass_left:.3f}"
        + r"\;=\;"
        + rf"\int_{{\phi(\Omega)}} p_1(y)\,dy \approx {mass_right:.3f}$",
        ha="center", va="bottom", fontsize=10.3, color=DARK
    )
    ax1.set_title("Same material region, same probability mass")

    levels1 = np.quantile(p1.ravel(), [0.55, 0.73, 0.84, 0.91, 0.96, 0.985])
    ax2.contourf(U, V, p1, levels=np.r_[0.0, levels1, p1.max()], cmap="Blues", alpha=0.22)
    ax2.contour(U, V, p1, levels=levels1, colors=BLUE, linewidths=1.2, alpha=0.88)

    ax2.scatter(Ys[:, 0], Ys[:, 1], s=18, color=DARK, alpha=0.42, zorder=3)
    ax2.scatter(Ys_in[:, 0], Ys_in[:, 1], s=22, color=RED, alpha=SAMPLE_ALPHA, zorder=4)

    add_region_patch(ax2, boundary_phi)
    ax2.text(
        0.03, 0.965, r"$\phi(\Omega)$",
        transform=ax2.transAxes, ha="left", va="top",
        fontsize=12.5, color=RED,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=1.5)
    )
    ax2.text(
        0.03, 0.05,
        r"transformed points $\phi(x_i)$",
        transform=ax2.transAxes, ha="left", va="bottom",
        fontsize=9.6, color=DARK,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=1.5)
    )

    ax2.set_title(r"Pushforward density $p_1=\phi_\# p_0$")
    ax2.set_xlabel(r"$y_1$")
    ax2.set_ylabel(r"$y_2$")
    ax2.set_aspect("equal")
    style_axes(ax2)

    fig.suptitle("Moving density and conservation of mass under a diffeomorphism", y=0.965)

    fig.text(
        0.5, 0.895,
        r"$p_1(y)\,dy = \phi_\#\!\left(p_0(x)\,dx\right)$"
        r"$\qquad\Longrightarrow\qquad$"
        r"$\int_{\phi(\Omega)} p_1(y)\,dy = \int_\Omega p_0(x)\,dx$",
        ha="center", va="center", fontsize=13.4
    )

    fig.text(
        0.5, 0.095,
        "The middle panel is schematic: it places $\\Omega$ and $\\phi(\\Omega)$ side by side "
        "to make the transport of the same particles visually explicit.",
        ha="center", va="center", fontsize=10.0, color=DARK
    )

    out_dir = Path(__file__).resolve().parent
    png_path = out_dir / f"output/{OUTPUT_BASENAME}.png"
    pdf_path = out_dir / f"output/{OUTPUT_BASENAME}.pdf"

    fig.savefig(png_path, dpi=DPI, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")

    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")
    print(f"Approximate mass of Omega under p0:        {mass_left:.6f}")
    print(f"Approximate mass of phi(Omega) under p1:   {mass_right:.6f}")
    print(f"Number of sampled points inside Omega:     {inside.sum()} / {len(Xs)}")


if __name__ == "__main__":
    main()
