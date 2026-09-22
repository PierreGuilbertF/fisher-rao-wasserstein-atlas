"""
Fisher--Rao versus quadratic Wasserstein geometry on densities over R.

Left column: a mass-preserving tangent perturbation u, with the same shape,
placed once near the centre of a standard Gaussian and once in its tail. The
Fisher--Rao energy density u^2/p makes the tail placement far more expensive.

Right column: two translated Gaussians with the same variance, at a short and
at a long distance. The quadratic Wasserstein distance is the translation
length, and does not care about the density level.

Self-contained: writes figure_01_FR_vs_Wasserstein.{pdf,png} next to this
script, in the output/ directory.
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch

# --- output ---------------------------------------------------------------

OUT = Path(__file__).resolve().parent / "output"
OUT.mkdir(exist_ok=True)
STEM = "figure_01_FR_vs_Wasserstein"
DPI = 400

# --- typography and colours ----------------------------------------------

mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
        "font.size": 9.5,
        "axes.labelsize": 9.5,
        "axes.titlesize": 10.0,
        "legend.fontsize": 8.0,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "lines.solid_capstyle": "round",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

COLOR_P = "#3B6EA5"  # reference density
COLOR_Q = "#B4483F"  # translated density
COLOR_CENTRE = "#3F7F4F"  # perturbation near the mode
COLOR_TAIL = "#B4483F"  # perturbation in the tail
COLOR_ARROW = "0.25"

# --- model parameters -----------------------------------------------------

MU0, SIGMA = 0.0, 1.0
X = np.linspace(-3.4, 6.2, 2400)

EPS = 0.05  # amplitude of the tangent perturbation
WIDTH = 0.22  # width of each bump
GAP = 0.34  # distance between the negative and positive bump
C_CENTRE = 0.0  # placement near the mode of p
C_TAIL = 2.8  # placement in the tail of p

DMU_SMALL = 0.45  # short transport
DMU_LARGE = 2.90  # long transport


def gaussian(x, mu=MU0, sigma=SIGMA):
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (np.sqrt(2.0 * np.pi) * sigma)


def perturbation(x, centre):
    """Mass-preserving tangent vector: one negative then one positive bump."""
    minus = gaussian(x, mu=centre - 0.5 * GAP, sigma=WIDTH)
    plus = gaussian(x, mu=centre + 0.5 * GAP, sigma=WIDTH)
    return EPS * (plus - minus)


def fisher_rao_density(u, p):
    return np.divide(u**2, p, out=np.zeros_like(p), where=p > 1e-14)


p = gaussian(X)
u_centre = perturbation(X, C_CENTRE)
u_tail = perturbation(X, C_TAIL)

e_centre = fisher_rao_density(u_centre, p)
e_tail = fisher_rao_density(u_tail, p)

norm_centre = float(np.trapz(e_centre, X))
norm_tail = float(np.trapz(e_tail, X))

print(f"mass centre : {np.trapz(u_centre, X):+.2e}")
print(f"mass tail   : {np.trapz(u_tail, X):+.2e}")
print(f"||u||^2_FR  centre = {norm_centre:.4f}")
print(f"||u||^2_FR  tail   = {norm_tail:.4f}")
print(f"ratio tail / centre = {norm_tail / norm_centre:.1f}")


def tidy(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(X[0], X[-1])


def draw_transport(ax, dmu, title):
    q = gaussian(X, mu=MU0 + dmu)
    ax.fill_between(X, p, color=COLOR_P, alpha=0.20, lw=0.0)
    ax.plot(X, p, color=COLOR_P, lw=1.4, label=r"$p=\mathcal{N}(0,1)$")
    ax.plot(
        X,
        q,
        color=COLOR_Q,
        lw=1.4,
        label=rf"$q=\mathcal{{N}}({dmu:.2f},1)$",
    )

    quantiles = np.array([-1.15, -0.6, 0.0, 0.6, 1.15])
    xs = MU0 + SIGMA * quantiles
    heights = 0.45 * gaussian(xs)
    for x_start, y in zip(xs, heights):
        ax.add_patch(
            FancyArrowPatch(
                (x_start, y),
                (x_start + dmu, y),
                arrowstyle="-|>",
                mutation_scale=7.5,
                color=COLOR_ARROW,
                lw=0.8,
                shrinkA=0.0,
                shrinkB=0.0,
                zorder=5,
            )
        )

    ax.set_title(title)
    ax.text(
        0.975,
        0.60,
        rf"$W_2(p,q)={abs(dmu):.2f}$",
        transform=ax.transAxes,
        ha="right",
        va="top",
    )
    ax.set_ylim(0.0, 0.47)
    ax.legend(loc="upper right", frameon=False, handlelength=1.4)
    tidy(ax)


fig, axes = plt.subplots(2, 2, figsize=(9.4, 5.8))

# --- top left: the reference density and the two placements of u ----------

ax = axes[0, 0]
ax.fill_between(X, p, color=COLOR_P, alpha=0.20, lw=0.0)
ax.plot(X, p, color=COLOR_P, lw=1.4, label=r"$p=\mathcal{N}(0,1)$")
ax.plot(X, u_centre, color=COLOR_CENTRE, lw=1.3, label=r"$u$ near the mode")
ax.plot(X, u_tail, color=COLOR_TAIL, lw=1.3, label=r"$u$ in the tail")
ax.axhline(0.0, color="0.7", lw=0.6)
ax.set_title(r"Fisher–Rao: same perturbation, different density level")
ax.set_ylabel(r"$p(x),\; u(x)$")
ax.text(
    0.975,
    0.985,
    r"$\int_{\mathbb{R}} u(x)\,dx=0$",
    transform=ax.transAxes,
    ha="right",
    va="top",
)
ax.legend(loc="upper right", frameon=False, handlelength=1.4, bbox_to_anchor=(1.0, 0.90))
ax.set_ylim(-0.13, 0.47)
tidy(ax)

# --- bottom left: the Fisher--Rao energy densities ------------------------

ax = axes[1, 0]
ax.fill_between(X, e_centre, color=COLOR_CENTRE, alpha=0.30, lw=0.0)
ax.plot(
    X,
    e_centre,
    color=COLOR_CENTRE,
    lw=1.3,
    label=rf"near the mode: $\|u\|^2_{{\mathrm{{FR}},p}}={norm_centre:.3f}$",
)
ax.fill_between(X, e_tail, color=COLOR_TAIL, alpha=0.30, lw=0.0)
ax.plot(
    X,
    e_tail,
    color=COLOR_TAIL,
    lw=1.3,
    label=rf"in the tail: $\|u\|^2_{{\mathrm{{FR}},p}}={norm_tail:.3f}$",
)
ax.set_title(r"Fisher–Rao local energy density $u^2/p$")
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$u(x)^2/p(x)$")
ax.text(
    0.028,
    0.72,
    rf"ratio $\approx {norm_tail / norm_centre:.0f}$",
    transform=ax.transAxes,
    ha="left",
    va="top",
)
ax.set_ylim(0.0, 1.18 * max(e_centre.max(), e_tail.max()))
ax.legend(loc="upper left", frameon=False, handlelength=1.4)
tidy(ax)

# --- right column: short and long transport ------------------------------

draw_transport(axes[0, 1], DMU_SMALL, r"Wasserstein: short transport")
draw_transport(axes[1, 1], DMU_LARGE, r"Wasserstein: long transport")
axes[1, 1].set_xlabel(r"$x$")
axes[0, 1].set_ylabel(r"$p(x),\; q(x)$")
axes[1, 1].set_ylabel(r"$p(x),\; q(x)$")

fig.suptitle(
    r"Fisher–Rao penalises relative local changes of density; "
    r"Wasserstein penalises the displacement of mass",
    y=0.985,
)
fig.subplots_adjust(
    left=0.075, right=0.985, top=0.885, bottom=0.085, hspace=0.42, wspace=0.20
)

for suffix in ("pdf", "png"):
    path = OUT / f"{STEM}.{suffix}"
    fig.savefig(path, dpi=DPI)
    print(f"Wrote {path}")
