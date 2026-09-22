"""
Fisher--Rao vs Wasserstein, as a reading of the metric on P+(R).

Reference density p = N(1, 1). Tangent vectors u satisfy ∫ u = 0.

Left (Fisher--Rao): the same local mass dipole, once in the bulk and once
in the tail. The cost density u²/p explodes where p is small.
Right (Wasserstein): the same Gaussian, translated by a short gap and by a
long one. The cost is the squared length of the transport map.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch
from scipy.stats import norm

OUT = Path(__file__).resolve().parent / "output"
OUT.mkdir(exist_ok=True)

COLOR_P = "#4C78A8"
COLOR_Q = "#C44E52"
COLOR_BULK = "#54A24B"
COLOR_TAIL = "#C44E52"

MU, SIG = 1.0, 1.0
X = np.linspace(-2.2, 6.4, 1200)


def gaussian(x, mu=MU, sigma=SIG):
    return (1.0 / (np.sqrt(2.0 * np.pi) * sigma)) * np.exp(
        -0.5 * ((x - mu) / sigma) ** 2
    )


def bump(x, center, width):
    return gaussian(x, mu=center, sigma=width)


def dipole(x, center, width, gap):
    """Mass-conserving local shift: -bump then +bump, ∫u=0."""
    return bump(x, center + 0.5 * gap, width) - bump(x, center - 0.5 * gap, width)


p = gaussian(X)

# Same geometric dipole (same gap, same width), two locations.
EPS = 0.06
WIDTH, GAP = 0.22, 0.28
u_bulk = EPS * dipole(X, center=1.0, width=WIDTH, gap=GAP)
u_tail = EPS * dipole(X, center=3.35, width=WIDTH, gap=GAP)

cost_bulk = np.divide(
    u_bulk**2, p, out=np.zeros_like(p), where=p > 1e-12
)
cost_tail = np.divide(
    u_tail**2, p, out=np.zeros_like(p), where=p > 1e-12
)
g_bulk = float(np.trapz(cost_bulk, X))
g_tail = float(np.trapz(cost_tail, X))
print(
    f"FR  bulk: ∫ u²/p = {g_bulk:.4f}   tail: ∫ u²/p = {g_tail:.4f}   "
    f"ratio={g_tail/g_bulk:.1f}"
)
print(
    f"∫ u_bulk={np.trapz(u_bulk, X):.2e}   "
    f"∫ u_tail={np.trapz(u_tail, X):.2e}"
)


def draw_transport(ax, mu_from, mu_to, sigma, title):
    y_from = gaussian(X, mu_from, sigma)
    y_to = gaussian(X, mu_to, sigma)
    ax.fill_between(X, y_from, color=COLOR_P, alpha=0.28)
    ax.plot(
        X,
        y_from,
        color=COLOR_P,
        lw=2.0,
        label=rf"$p=\mathcal{{N}}({mu_from:g},\ {sigma:g})$",
    )
    ax.plot(
        X,
        y_to,
        color=COLOR_Q,
        lw=2.0,
        label=rf"$q=\mathcal{{N}}({mu_to:g},\ {sigma:g})$",
    )

    delta = mu_to - mu_from
    probs = np.linspace(0.12, 0.88, 7)
    xs = mu_from + sigma * norm.ppf(probs)
    xt = xs + delta
    ys = 0.42 * np.minimum(
        gaussian(xs, mu_from, sigma), gaussian(xt, mu_to, sigma)
    )
    for a, b, y in zip(xs, xt, ys):
        ax.add_patch(
            FancyArrowPatch(
                (a, y),
                (b, y),
                arrowstyle="-|>",
                mutation_scale=10,
                color="0.2",
                lw=1.15,
                zorder=5,
            )
        )

    w2 = abs(delta)
    ax.set_title(title + "\n" + rf"$W_2(p,q)={w2:g}$", fontsize=10)
    ax.set_xlim(X[0], X[-1])
    ax.set_ylim(bottom=0.0)
    ax.legend(loc="upper right", fontsize=7.5, frameon=False)


fig, axes = plt.subplots(
    2,
    2,
    figsize=(12.0, 6.6),
    gridspec_kw={"hspace": 0.42, "wspace": 0.28},
)

# Fisher--Rao densities and tangent vectors.
ax = axes[0, 0]
ax.fill_between(X, p, color=COLOR_P, alpha=0.30)
ax.plot(X, p, color=COLOR_P, lw=2.0, label=r"$p=\mathcal{N}(1,1)$")
ax.plot(X, u_bulk, color=COLOR_BULK, lw=1.8, label=r"$u$ in the bulk")
ax.plot(X, u_tail, color=COLOR_TAIL, lw=1.8, label=r"$u$ in the tail")
ax.axhline(0.0, color="0.6", lw=0.6)
ax.set_xlim(X[0], X[-1])
ax.set_ylabel(r"$p(x),\ u(x)$")
ax.set_title(
    r"Fisher--Rao: same dipole, two locations"
    "\n"
    r"($\int u=0$)",
    fontsize=10,
)
ax.legend(loc="upper right", fontsize=7.5, frameon=False)

# Fisher--Rao cost densities.
ax = axes[1, 0]
ax.fill_between(
    X,
    cost_bulk,
    color=COLOR_BULK,
    alpha=0.40,
    label=rf"$u^{{2}}/p$ bulk (${g_bulk:.3f}$)",
)
ax.plot(X, cost_bulk, color=COLOR_BULK, lw=1.6)
ax.fill_between(
    X,
    cost_tail,
    color=COLOR_TAIL,
    alpha=0.40,
    label=rf"$u^{{2}}/p$ tail (${g_tail:.3f}$)",
)
ax.plot(X, cost_tail, color=COLOR_TAIL, lw=1.6)
ax.set_xlim(X[0], X[-1])
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$u(x)^{2}/p(x)$")
ax.set_title(
    rf"$g^{{\mathrm{{FR}}}}_p(u,u)=\int u^{{2}}/p$  "
    rf"(tail $/$ bulk $\approx {g_tail/g_bulk:.0f}$)",
    fontsize=10,
)
ax.legend(loc="upper right", fontsize=7.5, frameon=False)
ax.set_ylim(bottom=0.0)

# Wasserstein transport over short and long distances.
draw_transport(
    axes[0, 1],
    mu_from=1.0,
    mu_to=1.45,
    sigma=1.0,
    title=r"Wasserstein: small gap",
)
draw_transport(
    axes[1, 1],
    mu_from=1.0,
    mu_to=4.2,
    sigma=1.0,
    title=r"Wasserstein: large and distant gap",
)
axes[0, 1].set_ylabel(r"$p(x)$")
axes[1, 1].set_xlabel(r"$x$")
axes[1, 1].set_ylabel(r"$p(x)$")

fig.suptitle(
    r"Two readings of a displacement in $\mathcal{P}_{+}(\mathbb{R})$: "
    r"the ratio $u/p$ (Fisher--Rao) vs. transport distance (Wasserstein)",
    y=1.02,
    fontsize=12,
)
fig.subplots_adjust(
    left=0.07,
    right=0.98,
    top=0.88,
    bottom=0.09,
    hspace=0.42,
    wspace=0.28,
)
path = OUT / "26_fisher_rao_wasserstein.png"
fig.savefig(path, dpi=180, bbox_inches="tight")
print(f"Wrote {path}")
