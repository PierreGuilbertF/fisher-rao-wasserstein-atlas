#!/usr/bin/env python3
"""
Pedagogical visualization of the Kantorovich formulation of quadratic
optimal transport in one dimension.

The central panel is a genuine continuous joint density:
    pi(x,y) = p(x) q(y),
i.e. the independent coupling.

Therefore its marginals are exactly
    (pr_1)_# pi = p(x) dx,
    (pr_2)_# pi = q(y) dy.

The purpose of the figure is to visualize:
  1. a coupling pi as a probability law on R x R,
  2. its prescribed marginals p and q,
  3. the quadratic pairwise cost c(x,y)=|x-y|^2,
  4. the Kantorovich minimization over all such couplings.

Dependencies:
    numpy
    matplotlib

Run:
    python kantorovich_single_panel.py

Outputs when run:
    kantorovich_single_panel.png
    kantorovich_single_panel.pdf
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.cm import ScalarMappable


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

# First marginal: Gaussian
MU_P = 1.2
SIGMA_P = 0.95

# Second marginal: bimodal Gaussian mixture
MIXTURE_WEIGHTS = np.array([0.56, 0.44])
MIXTURE_MUS = np.array([-0.20, 2.65])
MIXTURE_SIGMAS = np.array([0.52, 0.72])

# Grid
N_GRID = 450
X_MIN, X_MAX = -3.2, 5.8

# Palette chosen to stay close to the reference figure
BLUE = "#4C78A8"
RED = "#C44E52"
DARK = "#333333"
GREY = "#8A8A8A"

# Requested blue -> red colormap
CMAP_NAME = "coolwarm"

OUTPUT_BASENAME = "kantorovich_single_panel"
DPI = 230


# ---------------------------------------------------------------------
# Density helpers
# ---------------------------------------------------------------------

def gaussian_pdf(z, mu, sigma):
    """Density of N(mu, sigma^2)."""
    z = np.asarray(z)
    return np.exp(-0.5 * ((z - mu) / sigma) ** 2) / (
        sigma * np.sqrt(2.0 * np.pi)
    )


def gaussian_mixture_pdf(z, weights, mus, sigmas):
    """Finite Gaussian-mixture density."""
    z = np.asarray(z)
    density = np.zeros_like(z, dtype=float)
    for weight, mu, sigma in zip(weights, mus, sigmas):
        density += weight * gaussian_pdf(z, mu, sigma)
    return density


# ---------------------------------------------------------------------
# Styling helpers
# ---------------------------------------------------------------------

def style_spines(ax):
    for spine in ax.spines.values():
        spine.set_linewidth(0.9)
        spine.set_color(DARK)


def draw_top_marginal(ax, x, p):
    ax.plot(x, p, color=BLUE, lw=2.4)
    ax.fill_between(x, 0.0, p, color=BLUE, alpha=0.24)

    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(bottom=0.0)
    ax.set_xticks([])
    ax.tick_params(axis="y", labelsize=8.5)
    ax.set_ylabel(r"$p(x)$", rotation=0, labelpad=18)

    ax.text(
        0.02, 0.90,
        r"$(\mathrm{pr}_1)_\#\pi=p(x)\,dx$",
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=10.2, color=DARK
    )

    style_spines(ax)


def draw_left_marginal(ax, y, q):
    ax.plot(q, y, color=RED, lw=2.4)
    ax.fill_betweenx(y, 0.0, q, color=RED, alpha=0.22)

    # Density grows toward the central square.
    ax.invert_xaxis()

    ax.set_ylim(y[0], y[-1])
    ax.set_xlim(1.08 * q.max(), 0.0)
    ax.set_yticks([])
    ax.tick_params(axis="x", labelsize=8.5)
    ax.set_xlabel(r"$q(y)$", labelpad=7)

    ax.text(
        0.34, 0.97,
        r"$(\mathrm{pr}_2)_\#\pi=q(y)\,dy$",
        transform=ax.transAxes,
        ha="center", va="top",
        fontsize=10.2, color=DARK,
        rotation=90
    )

    style_spines(ax)


def draw_joint_density(ax, x, y, pi, cmap):
    """
    Plot the actual joint probability density pi(x,y), not a relative,
    thresholded, or sparse transport matrix.
    """
    vmax = pi.max()
    norm = colors.Normalize(vmin=0.0, vmax=vmax)

    im = ax.imshow(
        pi,
        origin="lower",
        extent=[x[0], x[-1], y[0], y[-1]],
        aspect="equal",
        interpolation="bilinear",
        cmap=cmap,
        norm=norm,
        zorder=1,
    )

    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(y[0], y[-1])
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_title(
        r"An admissible joint density $\pi(x,y)$",
        fontsize=11.8,
        pad=10
    )

    style_spines(ax)
    return im


def draw_cost_geometry(ax, x, y):
    """
    Add the quadratic geometry without changing the joint density itself.
    """
    X, Y = np.meshgrid(x, y)
    C = (X - Y) ** 2

    # A few unobtrusive equal-cost curves.
    levels = [0.5, 2.0, 4.5, 8.0]
    ax.contour(
        X, Y, C,
        levels=levels,
        colors=[DARK] * len(levels),
        linewidths=0.65,
        alpha=0.22,
        zorder=4
    )

    # Zero-cost diagonal y=x.
    t = np.linspace(min(x[0], y[0]), max(x[-1], y[-1]), 500)
    ax.plot(
        t, t,
        ls="--", lw=1.25,
        color=DARK, alpha=0.80,
        zorder=5
    )

    ax.text(
        0.03, 0.045,
        r"$y=x:\quad c(x,y)=|x-y|^2=0$",
        transform=ax.transAxes,
        ha="left", va="bottom",
        fontsize=9.3, color=DARK,
        bbox=dict(
            facecolor="white",
            edgecolor="none",
            alpha=0.78,
            pad=1.8
        ),
        zorder=8
    )


def add_pairwise_cost_annotation(ax, x0, y0):
    """
    The vertical segment from (x0,x0) to (x0,y0) has length |x0-y0|.
    """
    ax.scatter([x0], [y0], s=30, color=DARK, zorder=10)

    ax.annotate(
        "",
        xy=(x0, y0),
        xytext=(x0, x0),
        arrowprops=dict(
            arrowstyle="<->",
            color=DARK,
            lw=1.35,
            shrinkA=1,
            shrinkB=1
        ),
        zorder=9
    )

    ax.text(
        x0 + 0.14,
        0.5 * (x0 + y0),
        r"$|x-y|$",
        fontsize=10,
        color=DARK,
        ha="left",
        va="center",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.82, pad=1.4),
        zorder=10
    )

    ax.text(
        x0 + 0.16,
        y0 + 0.22,
        r"$c(x,y)=|x-y|^2$",
        fontsize=10,
        color=DARK,
        ha="left",
        va="bottom",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.82, pad=1.4),
        zorder=10
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    x = np.linspace(X_MIN, X_MAX, N_GRID)
    y = x.copy()

    # Prescribed marginals.
    p = gaussian_pdf(x, MU_P, SIGMA_P)
    q = gaussian_mixture_pdf(
        y,
        MIXTURE_WEIGHTS,
        MIXTURE_MUS,
        MIXTURE_SIGMAS
    )

    # -------------------------------------------------------------
    # Genuine continuous joint density shown in the central panel.
    #
    # We deliberately use the independent coupling:
    #
    #     pi(x,y) = p(x) q(y).
    #
    # It is an admissible element of Pi(p,q), since
    #
    #     integral pi(x,y) dy = p(x),
    #     integral pi(x,y) dx = q(y).
    #
    # It is NOT claimed to be the minimizing coupling. The
    # Kantorovich formula above the figure explains that W_2^2 is
    # obtained by minimizing the same quadratic cost over all
    # admissible joint laws with these two marginals.
    # -------------------------------------------------------------
    pi = q[:, None] * p[None, :]

    # Optional numerical verification on the finite plotting window.
    dx = x[1] - x[0]
    dy = y[1] - y[0]

    marginal_x_numerical = np.trapz(pi, y, axis=0)
    marginal_y_numerical = np.trapz(pi, x, axis=1)

    # Relative errors are only from truncating R to [X_MIN, X_MAX].
    err_x = np.max(np.abs(marginal_x_numerical - p))
    err_y = np.max(np.abs(marginal_y_numerical - q))

    # Matplotlib defaults.
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10.5,
        "axes.titlesize": 11.5,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.titlesize": 17,
        "mathtext.fontset": "dejavusans",
    })

    cmap = plt.get_cmap(CMAP_NAME)

    fig = plt.figure(figsize=(10.9, 8.2))

    gs = fig.add_gridspec(
        2, 2,
        width_ratios=[1.25, 5.0],
        height_ratios=[1.35, 5.0],
        left=0.085,
        right=0.91,
        bottom=0.14,
        top=0.83,
        wspace=0.035,
        hspace=0.035,
    )

    ax_corner = fig.add_subplot(gs[0, 0])
    ax_top = fig.add_subplot(gs[0, 1])
    ax_left = fig.add_subplot(gs[1, 0])
    ax_joint = fig.add_subplot(gs[1, 1])

    ax_corner.axis("off")

    draw_top_marginal(ax_top, x, p)
    draw_left_marginal(ax_left, y, q)
    im = draw_joint_density(ax_joint, x, y, pi, cmap)

    draw_cost_geometry(ax_joint, x, y)

    # Example pair used only to explain the pairwise quadratic cost.
    add_pairwise_cost_annotation(
        ax_joint,
        x0=0.75,
        y0=2.65
    )

    # Main title.
    fig.suptitle(
        "Kantorovich formulation of quadratic optimal transport",
        y=0.965
    )

    # Main formula from the paper.
    fig.text(
        0.53, 0.905,
        r"$W_2^2(p,q)"
        r"=\inf_{\pi\in\Pi(p,q)}"
        r"\int_{\mathbb{R}\times\mathbb{R}}"
        r"|x-y|^2\,d\pi(x,y)$",
        ha="center",
        va="center",
        fontsize=14.6
    )

    fig.text(
        0.53, 0.870,
        r"$\Pi(p,q)"
        r"=\{\pi:\ "
        r"(\mathrm{pr}_1)_\#\pi=p(x)\,dx,\ "
        r"(\mathrm{pr}_2)_\#\pi=q(y)\,dy\}$",
        ha="center",
        va="center",
        fontsize=11.0
    )

    # Explicitly identify what the colormap is.
    ax_joint.text(
        0.03, 0.955,
        r"Shown coupling: $\pi(x,y)=p(x)\,q(y)$",
        transform=ax_joint.transAxes,
        ha="left",
        va="top",
        fontsize=9.7,
        color=DARK,
        bbox=dict(
            facecolor="white",
            edgecolor="none",
            alpha=0.82,
            pad=1.8
        ),
        zorder=8
    )

    # Raw joint-density colorbar.
    cbar = fig.colorbar(
        im,
        ax=ax_joint,
        orientation="vertical",
        fraction=0.045,
        pad=0.025,
        shrink=0.90
    )
    cbar.set_label(
        r"Joint density $\pi(x,y)$",
        rotation=90,
        labelpad=10
    )

    fig.text(
        0.53, 0.078,
        (
            "The colormap is the joint probability density itself. "
            "Its horizontal and vertical marginals are respectively "
            "$p(x)$ and $q(y)$. The Kantorovich problem varies the joint law "
            "$\\pi$ while keeping these marginals fixed."
        ),
        ha="center",
        va="center",
        fontsize=10.0,
        color=DARK
    )

    out_dir = Path(__file__).resolve().parent
    png_path = out_dir / f"output/{OUTPUT_BASENAME}.png"
    pdf_path = out_dir / f"output/{OUTPUT_BASENAME}.pdf"

    fig.savefig(png_path, dpi=DPI, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")

    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")
    print(f"Max finite-window marginal error in x: {err_x:.3e}")
    print(f"Max finite-window marginal error in y: {err_y:.3e}")

    plt.show()


if __name__ == "__main__":
    main()
