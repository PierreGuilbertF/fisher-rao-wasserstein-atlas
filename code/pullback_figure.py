from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from matplotlib.patches import FancyArrowPatch

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 10,
        "axes.linewidth": 0.8,
        "mathtext.fontset": "cm",
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.06,
    }
)


# -----------------------------------------------------------------------------
# Two small curved 3D surface patches: both are concave caps so that the
# tangent plane naturally lies above the surface.
# -----------------------------------------------------------------------------
def surface_M(x, y):
    return 0.62 + 0.10 * x + 0.06 * y - 0.22 * x**2 - 0.12 * y**2 - 0.06 * x * y


def grad_M(x, y):
    return np.array([0.10 - 0.44 * x - 0.06 * y, 0.06 - 0.24 * y - 0.06 * x])


def surface_N(x, y):
    return 0.56 - 0.07 * x + 0.10 * y - 0.10 * x**2 - 0.24 * y**2 + 0.05 * x * y


def grad_N(x, y):
    return np.array([-0.07 - 0.20 * x + 0.05 * y, 0.10 - 0.48 * y + 0.05 * x])


def tangent_basis(grad):
    e1 = np.array([1.0, 0.0, grad[0]])
    e2 = np.array([0.0, 1.0, grad[1]])
    e1 = e1 / np.linalg.norm(e1)
    e2 = e2 - np.dot(e2, e1) * e1
    e2 = e2 / np.linalg.norm(e2)
    n = np.cross(e1, e2)
    n = n / np.linalg.norm(n)
    if n[2] < 0:
        n = -n
    return e1, e2, n


# -----------------------------------------------------------------------------
# Drawing helpers
# -----------------------------------------------------------------------------
def add_surface(ax, which):
    g = np.linspace(-1.0, 1.0, 100)
    X, Y = np.meshgrid(g, g)
    if which == "M":
        Z = surface_M(X, Y)
        color = (0.78, 0.82, 0.89, 1.0)
    else:
        Z = surface_N(X, Y)
        color = (0.88, 0.84, 0.79, 1.0)

    surf = ax.plot_surface(
        X,
        Y,
        Z,
        color=color,
        edgecolor="none",
        linewidth=0,
        antialiased=False,
        shade=True,
        alpha=1.0,
        rcount=100,
        ccount=100,
    )
    surf.set_zorder(0)


def add_tangent_plane(ax, point, e1, e2, n, metric, label_text, edge_color):
    a = np.linspace(-0.60, 0.60, 70)
    b = np.linspace(-0.46, 0.46, 60)
    A, B = np.meshgrid(a, b)

    # Slight lift along the normal so that the tangent plane is geometrically
    # above the surface patch and still clearly visible.
    lift = 0.035
    P = point[None, None, :] + A[..., None] * e1 + B[..., None] * e2 + lift * n
    X, Y, Z = P[..., 0], P[..., 1], P[..., 2]

    q = metric[0, 0] * A**2 + 2.0 * metric[0, 1] * A * B + metric[1, 1] * B**2
    val = np.sqrt(np.maximum(q, 0.0))
    norm = colors.Normalize(vmin=0.0, vmax=np.percentile(val, 96))
    rgba = cm.coolwarm(norm(val))
    rgba[..., 3] = 0.28

    plane = ax.plot_surface(
        X,
        Y,
        Z,
        facecolors=rgba,
        shade=False,
        linewidth=0,
        antialiased=False,
        edgecolor="none",
    )
    plane.set_zorder(10)

    corners = np.array(
        [
            point - 0.60 * e1 - 0.46 * e2 + lift * n,
            point + 0.60 * e1 - 0.46 * e2 + lift * n,
            point + 0.60 * e1 + 0.46 * e2 + lift * n,
            point - 0.60 * e1 + 0.46 * e2 + lift * n,
            point - 0.60 * e1 - 0.46 * e2 + lift * n,
        ]
    )
    edge_line, = ax.plot(corners[:, 0], corners[:, 1], corners[:, 2], color=edge_color, lw=1.2, alpha=0.95)
    edge_line.set_zorder(11)

    # Subtle label for the colormap meaning.
    text_pos = point - 0.43 * e1 - 0.44 * e2 + 0.030 * n
    txt = ax.text(*text_pos, label_text, fontsize=11, color="0.24")
    txt.set_zorder(12)


def draw_angle_arc(ax, base, n, v1, v2, radius=0.18, color="0.25", label=None, label_offset=0.03):
    u1 = v1 / np.linalg.norm(v1)
    v2_orth = v2 - np.dot(v2, u1) * u1
    u2 = v2_orth / np.linalg.norm(v2_orth)
    theta2 = np.arctan2(np.dot(v2, u2), np.dot(v2, u1))
    t = np.linspace(0.15, theta2 - 0.08, 60)
    pts = base[None, :] + radius * (np.cos(t)[:, None] * u1 + np.sin(t)[:, None] * u2) + 0.02 * n
    ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=color, lw=1.1, zorder=25)
    if label is not None:
        mid = base + radius * (np.cos(0.5 * theta2) * u1 + np.sin(0.5 * theta2) * u2) + label_offset * n
        ax.text(*mid, label, fontsize=12, color="0.20")


def style_3d_axis(ax):
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-1.0, 1.0)
    ax.set_zlim(0.0, 0.95)
    ax.set_box_aspect((1, 1, 0.62))
    ax.set_proj_type("persp", focal_length=0.88)
    ax.view_init(elev=24, azim=-57)
    ax.computed_zorder = False
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.grid(False)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.line.set_color((0, 0, 0, 0))
        axis.pane.set_facecolor((1, 1, 1, 0))
        axis.pane.set_edgecolor((1, 1, 1, 0))
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_zlabel("")


# -----------------------------------------------------------------------------
# Figure layout
# -----------------------------------------------------------------------------
fig = plt.figure(figsize=(12.3, 6.3))
axL = fig.add_axes([0.00, 0.24, 0.36, 0.60], projection="3d")
axC = fig.add_axes([0.35, 0.25, 0.30, 0.50])
axR = fig.add_axes([0.64, 0.24, 0.36, 0.60], projection="3d")
axC.axis("off")

# Left panel: M, TxM, and the pullback metric on the tangent plane.
add_surface(axL, "M")
style_3d_axis(axL)
axL.text2D(0.06, 0.96, r"$M$", transform=axL.transAxes, fontsize=22)

x_xy = np.array([0.10, -0.08])
x = np.array([x_xy[0], x_xy[1], surface_M(*x_xy)])
gradx = grad_M(*x_xy)
e1M, e2M, nM = tangent_basis(gradx)
G_pull = np.array([[1.55, 0.48], [0.48, 0.78]])
add_tangent_plane(axL, x, e1M, e2M, nM, G_pull, r"$\|w\|_{(F^{*}h)_x}$", edge_color=(0.42, 0.52, 0.74))

u = 0.58 * e1M + 0.10 * e2M
v = 0.16 * e1M + 0.56 * e2M
baseM = x + 0.040 * nM
axL.scatter([baseM[0]], [baseM[1]], [baseM[2]], s=26, color="k", depthshade=False, zorder=30)
axL.quiver(*baseM, *u, color="#4c72b0", linewidth=2.9, arrow_length_ratio=0.13)
axL.quiver(*baseM, *v, color="#c44e52", linewidth=2.9, arrow_length_ratio=0.13)
axL.text(*(baseM + u + 0.055 * nM + np.array([-0.01, 0.01, 0.00])), r"$u$", color="#355e9a", fontsize=15)
axL.text(*(baseM + v + 0.055 * nM + np.array([0.01, 0.01, 0.00])), r"$v$", color="#a63d42", fontsize=15)
axL.text(*(baseM - 0.15 * e1M - 0.10 * e2M + 0.050 * nM), r"$x$", fontsize=13)
axL.text2D(0.37, 0.10, r"$T_xM$", transform=axL.transAxes, fontsize=17)
draw_angle_arc(axL, baseM, nM, u, v, radius=0.16, label=r"$g_x$")

# Right panel: N, T_{F(x)}N, and the metric h on the tangent plane.
add_surface(axR, "N")
style_3d_axis(axR)
axR.text2D(0.88, 0.96, r"$N$", transform=axR.transAxes, fontsize=22)

Fx_xy = np.array([0.02, -0.10])
Fx = np.array([Fx_xy[0], Fx_xy[1], surface_N(*Fx_xy)])
gradFx = grad_N(*Fx_xy)
e1N, e2N, nN = tangent_basis(gradFx)
H_amb = np.array([[0.72, -0.22], [-0.22, 1.62]])
add_tangent_plane(axR, Fx, e1N, e2N, nN, H_amb, r"$\|w\|_{h_{F(x)}}$", edge_color=(0.74, 0.58, 0.42))

Du = 0.52 * e1N + 0.34 * e2N
Dv = 0.56 * e1N + 0.00 * e2N
baseN = Fx + 0.040 * nN
axR.scatter([baseN[0]], [baseN[1]], [baseN[2]], s=26, color="k", depthshade=False, zorder=30)
axR.quiver(*baseN, *Du, color="#4c72b0", linewidth=2.9, arrow_length_ratio=0.13)
axR.quiver(*baseN, *Dv, color="#c44e52", linewidth=2.9, arrow_length_ratio=0.13)
axR.text(*(baseN + Du + 0.055 * nN + np.array([-0.04, 0.01, 0.00])), r"$dF_x(u)$", color="#355e9a", fontsize=15)
axR.text(*(baseN + Dv + 0.055 * nN + np.array([0.00, 0.01, 0.00])), r"$dF_x(v)$", color="#a63d42", fontsize=15)
axR.text(*(baseN - 0.20 * e1N - 0.10 * e2N + 0.050 * nN), r"$F(x)$", fontsize=13)
axR.text2D(0.28, 0.10, r"$T_{F(x)}N$", transform=axR.transAxes, fontsize=17)
draw_angle_arc(axR, baseN, nN, Dv, Du, radius=0.13, label=r"$h$")

# Center annotations: pushforward and pullback.
push = FancyArrowPatch((0.08, 0.71), (0.92, 0.71), arrowstyle="simple", mutation_scale=42,
                       fc="#7d7d7d", ec="#666666", alpha=0.80)
axC.add_patch(push)
axC.text(0.50, 0.86, r"$F$", ha="center", fontsize=24)
axC.text(0.50, 0.57, "push forward of tangent vectors", ha="center", fontsize=14, color="0.25")
axC.text(0.50, 0.45, r"$dF_x:T_xM\to T_{F(x)}N$", ha="center", fontsize=18)

pull = FancyArrowPatch((0.90, 0.20), (0.10, 0.20), arrowstyle="-|>", mutation_scale=22,
                       lw=2.4, linestyle=(0, (6, 3)), color="#8a8a8a")
axC.add_patch(pull)
axC.text(0.50, 0.31, "pull back of the metric", ha="center", fontsize=14, color="0.28")
axC.text(0.50, 0.08, r"$g=F^{*}h$", ha="center", fontsize=21)

fig.text(0.5, 0.080, r"$g_x(u,v)=h_{F(x)}(dF_x(u),dF_x(v))$", ha="center", fontsize=24)
fig.text(0.5, 0.022,
         r"If $F$ is an immersion, then $F^{*}h$ is positive definite and defines a Riemannian metric on $M$.",
         ha="center", fontsize=12.5, color="0.28")

fig.savefig('output/pullback_figure.png', dpi=240)
fig.savefig('output/data/pullback_figure.pdf')
