#!/usr/bin/env python3
"""
Weighted Helmholtz decomposition for Wasserstein–Otto geometry.

This script creates a 3-panel pedagogical figure showing, on top of an
isotropic Gaussian density p on R^2:

    1) an arbitrary velocity field v,
    2) its gradient part ∇φ,
    3) its weighted divergence-free part w,

with
    v = ∇φ + w,
    div(p w) = 0,
so that
    -div(p v) = -div(p ∇φ).

Interpretation:
- the gradient component ∇φ changes the density;
- the component w moves particles but is invisible to the continuity equation
  at first order.

The color palette matches the one used previously:
    density in blue, emphasis in red, text/arrows in dark gray.

Outputs:
    weighted_helmholtz_decomposition.png
    weighted_helmholtz_decomposition.pdf
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

SEED = 3
N_SAMPLES = 16
EPSILON = 0.34                 # step used only to visualize particle motion
GRID_N = 220
QUIVER_N = 17
XMIN, XMAX = -3.1, 3.1
YMIN, YMAX = -3.1, 3.1

# Color palette
BLUE = "#4C78A8"
RED = "#C44E52"
DARK = "#333333"
LIGHT_GREY = "#BDBDBD"

OUTPUT_BASENAME = "weighted_helmholtz_decomposition"
DPI = 220


# ---------------------------------------------------------------------
# Density and vector fields
# ---------------------------------------------------------------------

def gaussian_density(x, y):
    """Isotropic Gaussian density p(x,y) = (2π)^(-1) exp(-(x^2+y^2)/2)."""
    return np.exp(-0.5 * (x**2 + y**2)) / (2.0 * np.pi)


def potential(x, y):
    """
    A smooth scalar potential φ.
    Chosen so that ∇φ visibly crosses Gaussian level sets.
    """
    return (
        0.72 * x
        - 0.18 * y
        + 0.18 * x**2
        - 0.12 * y**2
        + 0.18 * x * y
    )


def grad_phi(x, y):
    """Gradient field ∇φ."""
    gx = 0.72 + 0.36 * x + 0.18 * y
    gy = -0.18 - 0.24 * y + 0.18 * x
    return gx, gy


def w_field(x, y):
    """
    Rotational field tangent to Gaussian level sets:
        w(x,y) = c (-y, x)

    For the isotropic Gaussian p, one has div(w)=0 and ∇p·w=0,
    hence div(p w)=0.
    """
    c = 0.85
    wx = -c * y
    wy = c * x
    return wx, wy


def v_field(x, y):
    """Arbitrary field v = ∇φ + w."""
    gx, gy = grad_phi(x, y)
    wx, wy = w_field(x, y)
    return gx + wx, gy + wy


def density_variation_from_field(x, y, field_fun):
    """
    Compute u = -div(p v) for a field v, analytically on the grid.

    We use:
        div(p v) = ∂x(p v_x) + ∂y(p v_y)
    and finite differences on the plotting grid.
    """
    p = gaussian_density(x, y)
    vx, vy = field_fun(x, y)
    dx = x[0, 1] - x[0, 0]
    dy = y[1, 0] - y[0, 0]
    div_pv = np.gradient(p * vx, dx, axis=1) + np.gradient(p * vy, dy, axis=0)
    return -div_pv


# ---------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------

def style_axes(ax):
    for spine in ax.spines.values():
        spine.set_linewidth(0.9)
        spine.set_color(DARK)


def setup_panel(ax, X, Y, P, title, subtitle):
    """
    Draw the common Gaussian background.
    """
    levels_fill = np.quantile(P.ravel(), [0.55, 0.72, 0.84, 0.92, 0.97, 0.992])
    levels_line = np.quantile(P.ravel(), [0.58, 0.75, 0.87, 0.94, 0.98])

    ax.contourf(X, Y, P, levels=np.r_[0.0, levels_fill, P.max()],
                cmap="Blues", alpha=0.20, zorder=0)
    ax.contour(X, Y, P, levels=levels_line, colors=BLUE,
               linewidths=1.2, alpha=0.88, zorder=1)

    ax.set_xlim(XMIN, XMAX)
    ax.set_ylim(YMIN, YMAX)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_title(title, fontsize=12.1, pad=11)

    ax.text(
        0.03, 0.97, subtitle,
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=10.0, color=DARK,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.82, pad=2.2),
        zorder=8
    )

    style_axes(ax)


def quiver_with_normalization(ax, Xq, Yq, U, V, color, label=None, alpha=0.88):
    """
    Plot arrows with mild normalization so the field is readable everywhere.
    """
    mag = np.sqrt(U**2 + V**2)
    scale = np.maximum(mag, 0.30)
    Uplot = U / scale
    Vplot = V / scale

    q = ax.quiver(
        Xq, Yq, Uplot, Vplot,
        angles="xy", scale_units="xy", scale=2.0,
        width=0.006, headwidth=3.9, headlength=5.1, headaxislength=4.6,
        color=color, alpha=alpha, zorder=4
    )
    if label is not None:
        ax.text(
            0.03, 0.06, label,
            transform=ax.transAxes,
            ha="left", va="bottom",
            fontsize=9.6, color=color,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.84, pad=1.8),
            zorder=9
        )
    return q


def draw_sample_motion(ax, pts, field_fun, color, epsilon=EPSILON, alpha=0.95):
    """
    Show a sparse set of particles and the displacement induced by the field.
    """
    x = pts[:, 0]
    y = pts[:, 1]
    u, v = field_fun(x, y)

    ax.scatter(x, y, s=18, color=DARK, alpha=0.78, zorder=5)
    ax.quiver(
        x, y, epsilon * u, epsilon * v,
        angles="xy", scale_units="xy", scale=1.0,
        width=0.005, headwidth=3.8, headlength=5.0, headaxislength=4.4,
        color=color, alpha=alpha, zorder=6
    )


def add_density_change_badge(ax, text, color):
    ax.text(
        0.97, 0.96, text,
        transform=ax.transAxes,
        ha="right", va="top",
        fontsize=10.1, color=color,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.84, pad=2.0),
        zorder=9
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    # Grids
    x = np.linspace(XMIN, XMAX, GRID_N)
    y = np.linspace(YMIN, YMAX, GRID_N)
    X, Y = np.meshgrid(x, y)
    P = gaussian_density(X, Y)

    xq = np.linspace(-2.6, 2.6, QUIVER_N)
    yq = np.linspace(-2.6, 2.6, QUIVER_N)
    Xq, Yq = np.meshgrid(xq, yq)

    # Fields on quiver grid
    Vx, Vy = v_field(Xq, Yq)
    Gx, Gy = grad_phi(Xq, Yq)
    Wx, Wy = w_field(Xq, Yq)

    # Numerical verification of the key identities
    U_v = density_variation_from_field(X, Y, v_field)
    U_g = density_variation_from_field(X, Y, grad_phi)
    U_w = density_variation_from_field(X, Y, w_field)

    max_diff_vg = float(np.max(np.abs(U_v - U_g)))
    max_uw = float(np.max(np.abs(U_w)))

    # Sample points from the Gaussian
    rng = np.random.default_rng(SEED)
    pts = rng.multivariate_normal(mean=[0.0, 0.0], cov=np.eye(2), size=N_SAMPLES)

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10.5,
        "axes.titlesize": 12.0,
        "axes.labelsize": 10.8,
        "xtick.labelsize": 8.8,
        "ytick.labelsize": 8.8,
        "figure.titlesize": 16.6,
        "mathtext.fontset": "dejavusans",
    })

    fig = plt.figure(figsize=(16.2, 5.8))
    gs = fig.add_gridspec(
        1, 3,
        left=0.045, right=0.985,
        bottom=0.16, top=0.83,
        wspace=0.17
    )

    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])

    # ---------------- Panel 1: arbitrary field v ----------------
    setup_panel(
        ax0, X, Y, P,
        title=r"Arbitrary velocity field $v$",
        subtitle=r"$v=\nabla\phi+w$"
    )
    quiver_with_normalization(ax0, Xq, Yq, Vx, Vy, DARK,
                              label=r"$u=-\mathrm{div}(p\,v)$")
    draw_sample_motion(ax0, pts, v_field, DARK)
    add_density_change_badge(ax0, r"general motion", DARK)

    # ---------------- Panel 2: gradient part ----------------
    setup_panel(
        ax1, X, Y, P,
        title=r"Gradient component $\nabla\phi$",
        subtitle=r"$-\mathrm{div}(p\,\nabla\phi)=u$"
    )
    quiver_with_normalization(ax1, Xq, Yq, Gx, Gy, RED,
                              label=r"$\nabla\phi$: changes the density")
    draw_sample_motion(ax1, pts, grad_phi, RED)
    add_density_change_badge(ax1, r"crosses level sets", RED)

    # ---------------- Panel 3: weighted divergence-free part ----------------
    setup_panel(
        ax2, X, Y, P,
        title=r"Weighted divergence-free part $w$",
        subtitle=r"$\mathrm{div}(p\,w)=0$"
    )
    quiver_with_normalization(ax2, Xq, Yq, Wx, Wy, DARK,
                              label=r"$w$: invisible to the continuity equation")
    draw_sample_motion(ax2, pts, w_field, DARK)
    add_density_change_badge(ax2, r"tangent circulation", DARK)

    # A few guide circles to emphasize that w is tangent to Gaussian level sets
    theta = np.linspace(0, 2*np.pi, 500)
    for r in [0.95, 1.55, 2.15]:
        ax2.plot(r*np.cos(theta), r*np.sin(theta),
                 color=RED, lw=0.85, alpha=0.28, ls="--", zorder=2)

    # Global title and formulas
    fig.suptitle("Weighted Helmholtz decomposition in Wasserstein–Otto geometry", y=0.962)

    fig.text(
        0.5, 0.895,
        r"$v=\nabla\phi+w,\qquad \mathrm{div}(p\,w)=0,"
        r"\qquad -\mathrm{div}(p\,v)=-\mathrm{div}(p\,\nabla\phi)$",
        ha="center", va="center", fontsize=14.2
    )

    fig.text(
        0.5, 0.860,
        r"$\|v\|_p^2=\|\nabla\phi\|_p^2+\|w\|_p^2"
        r"\qquad\Longrightarrow\qquad"
        r"\text{the minimal representative is the gradient part}$",
        ha="center", va="center", fontsize=12.6
    )

    fig.text(
        0.5, 0.080,
        "Blue contours represent the Gaussian density $p$. "
        "The red panel isolates the component that changes the density, "
        "whereas the right panel shows a circulation that moves particles "
        "without changing the density at first order.",
        ha="center", va="center", fontsize=10.0, color=DARK
    )

    out_dir = Path(__file__).resolve().parent
    png_path = out_dir / f"output/{OUTPUT_BASENAME}.png"
    pdf_path = out_dir / f"output/{OUTPUT_BASENAME}.pdf"

    fig.savefig(png_path, dpi=DPI, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")

    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")
    print(f"Max |u_v - u_grad| on the grid: {max_diff_vg:.3e}")
    print(f"Max |u_w| on the grid:          {max_uw:.3e}")


if __name__ == "__main__":
    main()
