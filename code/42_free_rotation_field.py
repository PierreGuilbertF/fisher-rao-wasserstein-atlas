"""
Free direction invisible to Fisher--Rao: rotation of an isotropic Gaussian.

    p(y) = (1/(2 pi)) exp(-(y_1^2 + y_2^2)/2),
    v(y) = (-y_2, y_1).

Then div v = 0 and <grad p, v> = 0, hence div(p v) = 0 and
u = -div(p v) = 0: the points move, the density does not change.

One panel: concentric level sets of p, sparse tangent arrows of v.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent / "output"
OUT.mkdir(exist_ok=True)

# --- Parameters -------------------------------------------------------------
RADII = (0.7, 1.2, 1.7, 2.2)       # contour radii (level sets of ||y||)
N_THETA = 400
ARROWS_PER_RING = (6, 8, 10, 12)   # sparse tangent arrows on each ring
ARROW_SCALE = 3.6
VIEW = (-2.85, 2.85)

# Palette shared with the other scripts.
COLOR_CONTOUR = "#4C78A8"
COLOR_ARROW = "#C44E52"
COLOR_ORIGIN = "#1F4E79"
COLOR_TEXT = "#333333"


def p_density(y1, y2):
    return np.exp(-0.5 * (y1**2 + y2**2)) / (2.0 * np.pi)


def velocity(y1, y2):
    """Rotational field v(y) = (-y_2, y_1)."""
    return -y2, y1


# --- Figure -----------------------------------------------------------------
fig, ax = plt.subplots(figsize=(5.2, 5.4))

theta = np.linspace(0.0, 2.0 * np.pi, N_THETA)

for radius, n_arrows in zip(RADII, ARROWS_PER_RING):
    # Density level set (circle, since p is radial).
    y1 = radius * np.cos(theta)
    y2 = radius * np.sin(theta)
    ax.plot(y1, y2, color=COLOR_CONTOUR, lw=1.15, alpha=0.85, zorder=2)

    # Sparse tangent arrows of v, sitting on the contour.
    angles = np.linspace(0.0, 2.0 * np.pi, n_arrows, endpoint=False)
    # Offset slightly so arrowheads do not sit on top of each other across rings.
    angles = angles + 0.12
    ay1 = radius * np.cos(angles)
    ay2 = radius * np.sin(angles)
    v1, v2 = velocity(ay1, ay2)
    ax.quiver(
        ay1,
        ay2,
        v1,
        v2,
        color=COLOR_ARROW,
        angles="xy",
        scale_units="xy",
        scale=ARROW_SCALE,
        width=0.006,
        headwidth=3.2,
        headlength=4.0,
        headaxislength=3.4,
        zorder=3,
    )

ax.plot(0.0, 0.0, "o", color=COLOR_ORIGIN, ms=4.5, zorder=4)

# Annotations ---------------------------------------------------------------
ax.text(
    0.50,
    0.97,
    r"$v\neq 0,\quad \mathrm{div}\,v=0,\quad "
    r"\langle\nabla p,\, v\rangle=0$",
    transform=ax.transAxes,
    ha="center",
    va="top",
    fontsize=10,
    color=COLOR_TEXT,
)
ax.text(
    0.50,
    0.90,
    r"$\Rightarrow\quad \mathrm{div}(pv)=0"
    r"\quad\Rightarrow\quad "
    r"u=-\mathrm{div}(pv)=0$",
    transform=ax.transAxes,
    ha="center",
    va="top",
    fontsize=10,
    color=COLOR_TEXT,
)
ax.text(
    0.50,
    0.04,
    r"points move, density unchanged",
    transform=ax.transAxes,
    ha="center",
    va="bottom",
    fontsize=9,
    color="#555555",
    style="italic",
)
ax.text(
    0.04,
    0.12,
    r"$v=(-y_{2},\,y_{1})$",
    transform=ax.transAxes,
    ha="left",
    va="bottom",
    fontsize=9,
    color=COLOR_ARROW,
)
ax.text(
    0.04,
    0.05,
    r"$p=\mathcal{N}(0,I_{2})$",
    transform=ax.transAxes,
    ha="left",
    va="bottom",
    fontsize=9,
    color=COLOR_CONTOUR,
)

ax.set_xlim(*VIEW)
ax.set_ylim(*VIEW)
ax.set_aspect("equal")
ax.set_xticks([])
ax.set_yticks([])
for spine in ax.spines.values():
    spine.set_visible(False)
ax.set_xlabel(r"$y_{1}$", fontsize=10, labelpad=2)
ax.set_ylabel(r"$y_{2}$", fontsize=10, labelpad=2)
# Keep axis labels without ticks: place them manually near the border.
ax.xaxis.set_label_coords(0.95, 0.02)
ax.yaxis.set_label_coords(0.02, 0.95)

fig.tight_layout(pad=0.4)

png = OUT / "42_free_rotation_field.png"
pdf = OUT / "42_free_rotation_field.pdf"
fig.savefig(png, dpi=220, bbox_inches="tight", pad_inches=0.08)
fig.savefig(pdf, bbox_inches="tight", pad_inches=0.08)
print(f"Wrote {png}")
print(f"Wrote {pdf}")
# Sanity: div v = 0 and <grad log p, v> = 0 along a sample ring.
r = RADII[1]
ang = np.linspace(0, 2 * np.pi, 200)
y1, y2 = r * np.cos(ang), r * np.sin(ang)
v1, v2 = velocity(y1, y2)
# grad log p = -y, so inner product with v = -y1*(-y2) + (-y2)*y1 = 0.
inner = -y1 * v1 - y2 * v2
print(f"max |<grad log p, v>| on ring = {np.max(np.abs(inner)):.2e}")
