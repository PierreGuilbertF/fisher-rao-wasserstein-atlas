from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from matplotlib.colors import LightSource


plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 10,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "mathtext.fontset": "cm",
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    }
)


# -----------------------------------------------------------------------------
# Model example for the infinitesimal transport cost
# -----------------------------------------------------------------------------
# Base density p on R^2: normalized anisotropic Gaussian.
# We choose a dilation-type potential so that the minimizing velocity field
# grows away from the mode. This makes the local kinetic density ||v||^2 p
# visibly different from p itself.

# Computational grid
n = 180
grid = np.linspace(-2.5, 2.5, n)
X, Y = np.meshgrid(grid, grid)
dx = grid[1] - grid[0]
R2 = X**2 + Y**2

# Probability density p(x, y)
sx, sy = 0.78, 1.05
rho = np.exp(-0.5 * ((X / sx) ** 2 + (Y / sy) ** 2))
p = rho / (rho.sum() * dx * dx)

# Potential phi and minimizing velocity field v = grad(phi)
# A mostly radial quadratic potential gives an expansion/compression pattern,
# so |v| is small near the mode and larger away from it.
phi = 0.22 * (X**2 + 1.35 * Y**2) + 0.06 * X * Y
vx = np.gradient(phi, dx, axis=1)
vy = np.gradient(phi, dx, axis=0)

# Tangent displacement u = -div(p v), numerically corrected to have zero mass
pvx = p * vx
pvy = p * vy
u = -(np.gradient(pvx, dx, axis=1) + np.gradient(pvy, dx, axis=0))
u = u - (u.sum() * dx * dx) / ((grid[-1] - grid[0]) ** 2)

# Infinitesimal transport cost density ||v||^2 p
speed2_p = (vx**2 + vy**2) * p

# Shared 3D view
ELEV = 29
AZIM = -57
BOX = (1.0, 1.0, 0.58)


def style_3d(ax):
    ax.set_proj_type("persp", focal_length=0.94)
    ax.view_init(elev=ELEV, azim=AZIM)
    ax.set_box_aspect(BOX)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((1, 1, 1, 0.0))
        axis.pane.set_edgecolor((1, 1, 1, 0.0))
        axis.line.set_color((0, 0, 0, 0))


def shaded_height_colors(Z, *, center_zero=False, light_az=315, light_alt=34):
    if center_zero:
        vmax = float(np.max(np.abs(Z)))
        norm = colors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    else:
        norm = colors.Normalize(vmin=float(np.min(Z)), vmax=float(np.max(Z)))

    base_rgb = cm.coolwarm(norm(Z))[..., :3]
    light = LightSource(azdeg=light_az, altdeg=light_alt)
    hill = light.hillshade(Z, vert_exag=1.35, fraction=1.0)
    intensity = 0.58 + 0.42 * hill
    shaded_rgb = np.clip(base_rgb * intensity[..., None], 0.0, 1.0)
    return np.dstack((shaded_rgb, np.ones_like(Z)))


fig = plt.figure(figsize=(11.2, 8.6))
ax1 = fig.add_subplot(2, 2, 1, projection="3d")
ax2 = fig.add_subplot(2, 2, 2, projection="3d")
ax3 = fig.add_subplot(2, 2, 3)
ax4 = fig.add_subplot(2, 2, 4, projection="3d")

# -----------------------------------------------------------------------------
# Top left: probability density p
# -----------------------------------------------------------------------------
style_3d(ax1)
p_colors = shaded_height_colors(p, light_az=315, light_alt=35)
ax1.plot_surface(
    X, Y, p,
    facecolors=p_colors,
    edgecolor="none",
    linewidth=0,
    antialiased=False,
    shade=False,
    alpha=1.0,
    rcount=n,
    ccount=n,
)
ax1.set_title(r"(a) Density $p(x)$", pad=10)
ax1.set_xlim(-2.5, 2.5)
ax1.set_ylim(-2.5, 2.5)
ax1.set_zlim(0.0, 1.12 * p.max())

# -----------------------------------------------------------------------------
# Top right: infinitesimal displacement u (with int u = 0)
# -----------------------------------------------------------------------------
style_3d(ax2)
u_colors = shaded_height_colors(u, center_zero=True, light_az=320, light_alt=34)
ax2.plot_surface(
    X, Y, u,
    facecolors=u_colors,
    edgecolor="none",
    linewidth=0,
    antialiased=False,
    shade=False,
    alpha=1.0,
    rcount=n,
    ccount=n,
)
ax2.plot_surface(
    X[::8, ::8], Y[::8, ::8], np.zeros_like(X[::8, ::8]),
    color=(0.93, 0.93, 0.93, 0.28),
    edgecolor="none",
    linewidth=0,
    shade=False,
    alpha=0.28,
)
ax2.text(-2.20, -2.15, 0.020, r"$u>0$", color="0.25", fontsize=10)
ax2.text(1.55, 1.50, -0.028, r"$u<0$", color="0.25", fontsize=10)
ax2.set_title(r"(b) Tangent displacement $u$ with $\int u = 0$", pad=10)
ax2.set_xlim(-2.5, 2.5)
ax2.set_ylim(-2.5, 2.5)
ax2.set_zlim(1.10 * u.min(), 1.10 * u.max())

# -----------------------------------------------------------------------------
# Bottom left: minimizing velocity field v = grad(phi)
# -----------------------------------------------------------------------------
ax3.set_aspect("equal")
ax3.set_xlim(-2.5, 2.5)
ax3.set_ylim(-2.5, 2.5)
ax3.set_xticks([])
ax3.set_yticks([])
for spine in ax3.spines.values():
    spine.set_visible(False)

levels = np.linspace(0.08 * p.max(), p.max(), 7)
ax3.contourf(X, Y, p, levels=levels, cmap="Greys", alpha=0.20)
ax3.contour(X, Y, p, levels=levels, colors="0.80", linewidths=0.8)

skip = 12
Xs = X[::skip, ::skip]
Ys = Y[::skip, ::skip]
VXs = vx[::skip, ::skip]
VYs = vy[::skip, ::skip]

ax3.quiver(
    Xs, Ys, VXs, VYs,
    angles="xy",
    scale_units="xy",
    scale=4.4,
    width=0.006,
    color="#5a6c7d",
    alpha=0.95,
)
ax3.set_title(r"(c) Minimizing velocity field $v=\nabla \phi$", pad=10)
ax3.text(0.02, 0.05, r"$u=-\operatorname{div}(p\,v)$", transform=ax3.transAxes, fontsize=12, color="0.22")

# -----------------------------------------------------------------------------
# Bottom right: local kinetic cost density ||v||^2 p
# -----------------------------------------------------------------------------
style_3d(ax4)
cost_colors = shaded_height_colors(speed2_p, light_az=310, light_alt=34)
ax4.plot_surface(
    X, Y, speed2_p,
    facecolors=cost_colors,
    edgecolor="none",
    linewidth=0,
    antialiased=False,
    shade=False,
    alpha=1.0,
    rcount=n,
    ccount=n,
)
ax4.set_title(r"(d) Local cost density $\|v\|^2 p$", pad=10)
ax4.set_xlim(-2.5, 2.5)
ax4.set_ylim(-2.5, 2.5)
ax4.set_zlim(0.0, 1.10 * speed2_p.max())

fig.suptitle(r"Infinitesimal transport cost in Wasserstein--Otto geometry", y=0.98, fontsize=13)
fig.savefig("output/infinitesimal_transport_cost.png", dpi=240)
fig.savefig("output/infinitesimal_transport_cost.pdf")
