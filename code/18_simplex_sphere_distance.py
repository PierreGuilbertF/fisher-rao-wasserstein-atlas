"""
Fisher--Rao geometry on the 3-outcome simplex.

Left:  the simplex {p1+p2+p3 = 1, pi > 0} sitting in R^3.
Right: its image by psi(p) = 2 sqrt(p), the positive orthant of the sphere
       of radius 2.

Both surfaces are coloured by the Fisher--Rao distance to a reference point
(the uniform distribution by default), with iso-distance lines. On the simplex the
distance is the pullback of the round distance; on the sphere it is the
great-circle distance, 2 * arccos(<sqrt(p), sqrt(q)>).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from matplotlib import cm, colors
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

OUT = Path(__file__).resolve().parent / "output"
OUT.mkdir(exist_ok=True)

RADIUS = 2.0
P_REF = np.array([1.0, 1.0, 1.0]) / 3.0
N_SUBDIV = 130
CMAP = plt.get_cmap("coolwarm")
ELEV, AZIM = 28.0, 40.0
EPS = 1e-9


def fisher_rao_distance(p, q=P_REF):
    """Round distance on the sphere of radius 2 pulled back to the simplex."""
    overlap = np.sqrt(np.clip(p, 0.0, None)) @ np.sqrt(q)
    return RADIUS * np.arccos(np.clip(overlap, -1.0, 1.0))


def barycentric_grid(n):
    """Points (u, v) of a regular grid on {u, v >= 0, u + v <= 1}."""
    us, vs = [], []
    for i in range(n + 1):
        for j in range(n + 1 - i):
            us.append(i / n)
            vs.append(j / n)
    return np.array(us), np.array(vs)


def to_simplex(u, v):
    return np.column_stack([u, v, 1.0 - u - v])


def to_sphere(p):
    return RADIUS * np.sqrt(np.clip(p, 0.0, None))


u, v = barycentric_grid(N_SUBDIV)
tri = mtri.Triangulation(u, v)
simplex_pts = to_simplex(u, v)
sphere_pts = to_sphere(simplex_pts)
dist = np.array([fisher_rao_distance(p) for p in simplex_pts])

d_max = float(dist.max())
norm = colors.Normalize(vmin=0.0, vmax=d_max)
face_dist = dist[tri.triangles].mean(axis=1)
face_colors = CMAP(norm(face_dist))

# Iso-distance curves are extracted in the (u, v) parameter domain, then
# pushed onto each surface.
levels = np.arange(0.25, d_max, 0.25)
helper_fig, helper_ax = plt.subplots()
iso = helper_ax.tricontour(tri, dist, levels=levels)
iso_segments = [seg for level_segs in iso.allsegs for seg in level_segs]
plt.close(helper_fig)

NORMAL = np.ones(3) / np.sqrt(3.0)


def lift_segments(segments, target):
    lifted = []
    for seg in segments:
        if len(seg) < 2:
            continue
        pts = to_simplex(seg[:, 0], seg[:, 1])
        if target == "simplex":
            lifted.append(pts + 0.004 * NORMAL)
        else:
            lifted.append(to_sphere(pts) * 1.004)
    return lifted


def draw_surface(ax, vertices, title, iso_target, ref_point, ref_label):
    polys = vertices[tri.triangles]
    collection = Poly3DCollection(
        polys,
        facecolors=face_colors,
        edgecolors="none",
        shade=False,
    )
    ax.add_collection3d(collection)

    for pts in lift_segments(iso_segments, iso_target):
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color="k", lw=0.6, alpha=0.7)

    # Outline: the three edges of the parameter triangle.
    corners = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    border = to_simplex(corners[:, 0], corners[:, 1])
    if iso_target == "sphere":
        border = to_sphere(border) * 1.004
    else:
        border = border + 0.004 * NORMAL
    ax.plot(border[:, 0], border[:, 1], border[:, 2], color="#333333", lw=1.1)

    ax.plot(
        [ref_point[0]],
        [ref_point[1]],
        [ref_point[2]],
        "o",
        color="white",
        ms=8,
        mew=1.3,
        mec="black",
        zorder=10,
    )
    offset = 0.16 if iso_target == "simplex" else 0.32
    ax.text(
        ref_point[0],
        ref_point[1],
        ref_point[2] + offset,
        ref_label,
        fontsize=8,
        ha="center",
        zorder=12,
    )

    ax.set_title(title, fontsize=11, pad=6)
    ax.view_init(elev=ELEV, azim=AZIM)
    ax.tick_params(labelsize=7, pad=-1)
    ax.set_box_aspect((1.0, 1.0, 1.0))


fig = plt.figure(figsize=(11.0, 4.8))

ax_left = fig.add_subplot(1, 2, 1, projection="3d")
draw_surface(
    ax_left,
    simplex_pts,
    r"Simplex $\mathcal{P}_{+}$: $p_{1}+p_{2}+p_{3}=1$",
    "simplex",
    P_REF,
    r"$p_{\star}$ uniform",
)
ax_left.set_xlim(0.0, 1.0)
ax_left.set_ylim(0.0, 1.0)
ax_left.set_zlim(0.0, 1.0)
ax_left.set_xlabel(r"$p_{1}$", labelpad=-4)
ax_left.set_ylabel(r"$p_{2}$", labelpad=-4)
ax_left.set_zlabel(r"$p_{3}$", labelpad=-4)

ax_right = fig.add_subplot(1, 2, 2, projection="3d")
draw_surface(
    ax_right,
    sphere_pts,
    r"Positive orthant of the sphere of radius $2$: $\psi(p)=2\sqrt{p}$",
    "sphere",
    to_sphere(P_REF),
    r"$\psi(p_{\star})$",
)
ax_right.set_xlim(0.0, RADIUS)
ax_right.set_ylim(0.0, RADIUS)
ax_right.set_zlim(0.0, RADIUS)
ax_right.set_xlabel(r"$2\sqrt{p_{1}}$", labelpad=-4)
ax_right.set_ylabel(r"$2\sqrt{p_{2}}$", labelpad=-4)
ax_right.set_zlabel(r"$2\sqrt{p_{3}}$", labelpad=-4)

mappable = cm.ScalarMappable(norm=norm, cmap=CMAP)
mappable.set_array(dist)
cbar = fig.colorbar(mappable, ax=[ax_left, ax_right], pad=0.04, shrink=0.82)
cbar.set_label(r"Fisher--Rao distance to $p_{\star}$")
cbar.add_lines(levels, ["k"] * len(levels), [0.6] * len(levels))

fig.suptitle(
    r"Fisher--Rao distance on the simplex and on the sphere",
    y=0.97,
    fontsize=12,
)

path = OUT / "18_simplex_sphere_distance.png"
fig.savefig(path, dpi=180, bbox_inches="tight")
print(f"Wrote {path}")
# plt.show()
