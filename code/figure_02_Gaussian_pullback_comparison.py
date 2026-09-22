"""
Pullback geometries of the univariate Gaussian location--scale family.

Parameter space: (μ, σ) ∈ R × R>0, densities p_{μ,σ} = N(μ, σ²).

Left:  Fisher--Rao pullback
          ds²_FR = (dμ² + 2 dσ²) / σ²
       isometric (up to a constant) to the Poincaré half-plane. Geodesics are
       semicircles orthogonal to σ = 0 in the chart (ξ, η) = (μ/√2, σ).

Right: quadratic Wasserstein--Otto pullback
          ds²_W = dμ² + dσ²
       Euclidean. Geodesics are straight segments.

Same endpoints, same domain, same (μ, σ) coordinates.

Writes figure_02_Gaussian_pullback_comparison.{pdf,png} in output/.
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse

OUT = Path(__file__).resolve().parent / "output"
OUT.mkdir(exist_ok=True)
STEM = "figure_02_Gaussian_pullback_comparison"
DPI = 400

mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
        "font.size": 9.5,
        "axes.labelsize": 9.5,
        "axes.titlesize": 10.5,
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

COLOR_START = "#3B6EA5"
COLOR_END = "#B4483F"
COLOR_GEOD = "#1F1F1F"
COLOR_EUCL = "0.55"
COLOR_BALL = "0.62"
COLOR_GLYPH = "#3B6EA5"

# Moderate endpoints: along the Fisher--Rao geodesic, σ rises from 0.8 to
# about 1.63, enough to show spreading without over-flattening the glyphs.
MU0, SIG0 = -2.0, 0.8
MU1, SIG1 = 2.0, 0.8

XLIM = (-3.15, 3.15)
YLIM = (0.0, 2.40)
R_BALL = 0.14
N_PATH = 400
GLYPH_T = (0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0)
# Callout insets above the path. Peak height is fixed; width grows only
# mildly with σ so spreading is readable without oversized blobs.
GLYPH_HALF_W = 0.28
GLYPH_SUPPORT = 2.2
GLYPH_WIDTH_CAP = 1.25
GLYPH_PEAK = 0.17
GLYPH_LIFT = 0.14
BALL_SIGS = np.array([0.45, 0.90, 1.35, 1.80])
BALL_MUS = np.linspace(-2.4, 2.4, 5)


def gaussian(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (np.sqrt(2.0 * np.pi) * sigma)


def to_poincare(mu, sigma):
    return mu / np.sqrt(2.0), sigma


def from_poincare(xi, eta):
    return np.sqrt(2.0) * xi, eta


def fisher_rao_geodesic(mu_a, sig_a, mu_b, sig_b, n=N_PATH):
    """Semicircle geodesic in the Poincaré chart (ξ, η) = (μ/√2, σ)."""
    xi_a, eta_a = to_poincare(mu_a, sig_a)
    xi_b, eta_b = to_poincare(mu_b, sig_b)

    if np.isclose(xi_a, xi_b):
        eta = np.linspace(eta_a, eta_b, n)
        xi = np.full_like(eta, xi_a)
        mu, sig = from_poincare(xi, eta)
        return mu, sig

    centre = (xi_a**2 + eta_a**2 - xi_b**2 - eta_b**2) / (2.0 * (xi_a - xi_b))
    radius = np.sqrt((xi_a - centre) ** 2 + eta_a**2)
    theta_a = np.arctan2(eta_a, xi_a - centre)
    theta_b = np.arctan2(eta_b, xi_b - centre)
    theta = np.linspace(theta_a, theta_b, n)
    xi = centre + radius * np.cos(theta)
    eta = radius * np.sin(theta)
    return from_poincare(xi, eta)


def wasserstein_geodesic(mu_a, sig_a, mu_b, sig_b, n=N_PATH):
    t = np.linspace(0.0, 1.0, n)
    return (1.0 - t) * mu_a + t * mu_b, (1.0 - t) * sig_a + t * sig_b


def interpolate_path(mu, sig, t_values):
    s = np.linspace(0.0, 1.0, len(mu))
    return np.interp(t_values, s, mu), np.interp(t_values, s, sig)


def tidy(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$\mu$")
    ax.set_ylabel(r"$\sigma$")
    ax.axhline(0.0, color="0.75", lw=0.6)


def add_endpoints(ax):
    ax.plot(MU0, SIG0, "o", color=COLOR_START, ms=7.0, zorder=6)
    ax.plot(MU1, SIG1, "s", color=COLOR_END, ms=6.5, zorder=6)
    ax.annotate(
        rf"$\mathcal{{N}}({MU0:g},{SIG0:g})$",
        (MU0, SIG0),
        textcoords="offset points",
        xytext=(-34, -16),
        color="0.25",
        fontsize=8.5,
    )
    ax.annotate(
        rf"$\mathcal{{N}}({MU1:g},{SIG1:g})$",
        (MU1, SIG1),
        textcoords="offset points",
        xytext=(6, -16),
        color="0.25",
        fontsize=8.5,
    )


def add_density_glyph(ax, mu, sigma, x_anchor, y_anchor, color):
    """Small callout above the path: fixed peak, mild width change with σ."""
    half_w = GLYPH_HALF_W * min(np.sqrt(sigma / SIG0), GLYPH_WIDTH_CAP)
    x_obs = np.linspace(
        mu - GLYPH_SUPPORT * sigma, mu + GLYPH_SUPPORT * sigma, 100
    )
    dens = gaussian(x_obs, mu, sigma)
    dens = dens / dens.max()

    y0 = y_anchor + GLYPH_LIFT
    ax.plot(
        [x_anchor, x_anchor],
        [y_anchor + 0.02, y0],
        color="0.7",
        lw=0.5,
        zorder=4,
        solid_capstyle="round",
    )
    x_loc = x_anchor + half_w * (x_obs - mu) / (GLYPH_SUPPORT * sigma)
    y_loc = y0 + GLYPH_PEAK * dens
    ax.fill_between(x_loc, y0, y_loc, color=color, alpha=0.18, lw=0, zorder=5)
    ax.plot(x_loc, y_loc, color=color, lw=0.75, zorder=5)


def add_fr_balls(ax):
    """Ellipses dθᵀ G dθ = r² with G = diag(1/σ², 2/σ²)."""
    for mu in BALL_MUS:
        for sig in BALL_SIGS:
            width = 2.0 * R_BALL * sig
            height = 2.0 * R_BALL * sig / np.sqrt(2.0)
            if sig - 0.5 * height <= 0.02:
                continue
            ell = Ellipse(
                (mu, sig),
                width=width,
                height=height,
                facecolor="none",
                edgecolor=COLOR_BALL,
                lw=0.55,
                zorder=1,
            )
            ax.add_patch(ell)


def add_w_balls(ax):
    for mu in BALL_MUS:
        for sig in BALL_SIGS:
            if sig - R_BALL <= 0.02:
                continue
            ell = Ellipse(
                (mu, sig),
                width=2.0 * R_BALL,
                height=2.0 * R_BALL,
                facecolor="none",
                edgecolor=COLOR_BALL,
                lw=0.55,
                zorder=1,
            )
            ax.add_patch(ell)


mu_fr, sig_fr = fisher_rao_geodesic(MU0, SIG0, MU1, SIG1)
mu_w, sig_w = wasserstein_geodesic(MU0, SIG0, MU1, SIG1)
print(f"FR σ along path: min={sig_fr.min():.3f}, max={sig_fr.max():.3f}")

fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.55))

# --- Fisher--Rao ----------------------------------------------------------

ax = axes[0]
add_fr_balls(ax)
ax.plot(
    [MU0, MU1],
    [SIG0, SIG1],
    ls=(0, (3.0, 2.2)),
    color=COLOR_EUCL,
    lw=0.85,
    zorder=2,
    label="Euclidean segment",
)
ax.plot(mu_fr, sig_fr, color=COLOR_GEOD, lw=1.7, zorder=3, label="geodesic")
add_endpoints(ax)
mu_g, sig_g = interpolate_path(mu_fr, sig_fr, GLYPH_T)
for mu, sig in zip(mu_g, sig_g):
    add_density_glyph(ax, mu, sig, mu, sig, COLOR_GLYPH)
tidy(ax)
ax.set_title("Fisher–Rao pullback")
ax.text(
    0.03,
    0.97,
    r"$ds^2_{\mathrm{FR}}=(d\mu^2+2\,d\sigma^2)/\sigma^2$",
    transform=ax.transAxes,
    ha="left",
    va="top",
)
ax.legend(loc="upper right", frameon=False, handlelength=1.6)

# --- Wasserstein--Otto ----------------------------------------------------

ax = axes[1]
add_w_balls(ax)
ax.plot(mu_w, sig_w, color=COLOR_GEOD, lw=1.7, zorder=3, label="geodesic")
add_endpoints(ax)
mu_g, sig_g = interpolate_path(mu_w, sig_w, GLYPH_T)
for mu, sig in zip(mu_g, sig_g):
    add_density_glyph(ax, mu, sig, mu, sig, COLOR_GLYPH)
tidy(ax)
ax.set_title("Wasserstein–Otto pullback")
ax.text(
    0.03,
    0.97,
    r"$ds^2_{\mathrm{W}}=d\mu^2+d\sigma^2$",
    transform=ax.transAxes,
    ha="left",
    va="top",
)
ax.legend(loc="upper right", frameon=False, handlelength=1.6)

fig.suptitle(
    r"Same family, same endpoints; different ambient metrics, different geometries",
    y=0.98,
)
fig.subplots_adjust(left=0.07, right=0.99, top=0.86, bottom=0.13, wspace=0.18)

for suffix in ("pdf", "png"):
    path = OUT / f"{STEM}.{suffix}"
    fig.savefig(path, dpi=DPI)
    print(f"Wrote {path}")
