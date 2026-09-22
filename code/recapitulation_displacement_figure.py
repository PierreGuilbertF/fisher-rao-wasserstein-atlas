from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "mathtext.fontset": "cm",
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    }
)

# -----------------------------------------------------------------------------
# One-dimensional example for the recapitulation section.
# We choose a standard Gaussian density and a dilation-type minimal velocity
# field v(x)=alpha x. Then
#   u = -(pv)' = alpha (x^2-1) p,
# which has zero mass and clearly distinguishes the Fisher--Rao and
# Wasserstein--Otto integrands.
# -----------------------------------------------------------------------------

x = np.linspace(-3.6, 3.6, 1200)
dx = x[1] - x[0]
alpha = 0.95

p = (1.0 / np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * x**2)
v = alpha * x
u = alpha * (x**2 - 1.0) * p
fr_density = u**2 / p
wo_density = v**2 * p

# Numerical checks
mass_u = np.trapz(u, x)
fr_cost = np.trapz(fr_density, x)
wo_cost = np.trapz(wo_density, x)

# Colors inspired by the previous figure palette.
COL_P = "#4C78A8"
COL_P_FILL = "#D5DFEC"
COL_U = "#C44E52"
COL_V = "#5F6B7A"
COL_FR = "#C44E52"
COL_WO = "#4C78A8"

fig, axs = plt.subplots(2, 2, figsize=(12.2, 7.7), sharex=True)
ax1, ax2, ax3, ax4 = axs[0, 0], axs[0, 1], axs[1, 0], axs[1, 1]

for ax in axs.flat:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=3.0, width=0.8)
    ax.set_xlim(-3.4, 3.4)

# -----------------------------------------------------------------------------
# (a) Density p
# -----------------------------------------------------------------------------
ax1.fill_between(x, 0.0, p, color=COL_P_FILL, alpha=1.0)
ax1.plot(x, p, color=COL_P, lw=2.4)
ax1.set_ylim(0.0, 0.43)
ax1.set_ylabel(r"$p(x)$")
ax1.set_title(r"(a) Density $p$", pad=8)
ax1.text(0.04, 0.86, r"$p(x)=\dfrac{1}{\sqrt{2\pi}}e^{-x^2/2}$", transform=ax1.transAxes, color="0.25")

# -----------------------------------------------------------------------------
# (b) Tangent displacement u and minimal velocity field v
# -----------------------------------------------------------------------------
# Use symmetric y-limits on both axes so that y=0 is the same horizontal line
# for u and v. This makes the odd field v(x)=alpha x visibly cross zero at x=0.
u_lim = 1.08 * np.max(np.abs(u))
v_lim = 1.08 * np.max(np.abs(v))

ax2.axhline(0.0, color="0.70", lw=0.9)
ax2.axvline(0.0, color="0.88", lw=0.8, ls=":")
ax2.plot(x, u, color=COL_U, lw=2.2, label=r"$u(x)$")
ax2.fill_between(x, 0.0, u, where=(u >= 0), color=COL_U, alpha=0.18)
ax2.fill_between(x, 0.0, u, where=(u <= 0), color=COL_U, alpha=0.10)
ax2.set_ylim(-u_lim, u_lim)
ax2.set_ylabel(r"$u(x)$")
ax2.set_title(r"(b) Displacement $u$ and minimal field $v$", pad=8)

ax2b = ax2.twinx()
ax2b.spines["top"].set_visible(False)
ax2b.plot(x, v, color=COL_V, lw=2.0, ls="--", label=r"$v(x)=0.95x$")
ax2b.set_ylim(-v_lim, v_lim)
ax2b.set_ylabel(r"$v(x)$", color=COL_V)
ax2b.tick_params(axis="y", colors=COL_V)

lines1, labels1 = ax2.get_legend_handles_labels()
lines2, labels2 = ax2b.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper left", frameon=False)
ax2.text(0.05, 0.08, r"$u=-(pv)'$", transform=ax2.transAxes, color="0.25")

# -----------------------------------------------------------------------------
# (c) Fisher--Rao density u^2/p
# -----------------------------------------------------------------------------
ax3.fill_between(x, 0.0, fr_density, color=COL_FR, alpha=0.22)
ax3.plot(x, fr_density, color=COL_FR, lw=2.2)
ax3.set_ylabel(r"$u(x)^2/p(x)$")
ax3.set_xlabel(r"$x$")
ax3.set_ylim(0.0, 1.08 * fr_density.max())
ax3.set_title(r"(c) Fisher--Rao sees $u^2/p$", pad=8)
ax3.text(0.04, 0.87, r"$g_p^{\mathrm{FR}}(u,u)=\int \dfrac{u^2}{p}\,dx$", transform=ax3.transAxes, color="0.25")

# -----------------------------------------------------------------------------
# (d) Wasserstein--Otto density |v|^2 p
# -----------------------------------------------------------------------------
ax4.fill_between(x, 0.0, wo_density, color=COL_WO, alpha=0.22)
ax4.plot(x, wo_density, color=COL_WO, lw=2.2)
ax4.set_ylabel(r"$|v(x)|^2 p(x)$")
ax4.set_xlabel(r"$x$")
ax4.set_ylim(0.0, 1.08 * wo_density.max())
ax4.set_title(r"(d) Wasserstein--Otto sees $|v|^2p$", pad=8)
ax4.text(0.04, 0.87, r"$\|u\|_{p,\mathrm{WO}}^2=\int |v|^2p\,dx$", transform=ax4.transAxes, color="0.25")

fig.suptitle(
    r"Recapitulation: one displacement, two infinitesimal costs in $\mathcal{P}_{+}(\mathbb{R})$",
    y=0.98,
    fontsize=14,
)

# Small footer with consistency check and values.
fig.text(
    0.5,
    0.01,
    rf"Here $\int u\,dx \approx {mass_u:.1e}$,  "
    rf"$g_p^{{\mathrm{{FR}}}}(u,u) \approx {fr_cost:.3f}$,  "
    rf"$\|u\|_{{p,\mathrm{{WO}}}}^2 \approx {wo_cost:.3f}$.",
    ha="center",
    color="0.30",
    fontsize=10,
)

fig.savefig("output/recapitulation_displacement.png", dpi=240)
fig.savefig("output/recapitulation_displacement.pdf")
