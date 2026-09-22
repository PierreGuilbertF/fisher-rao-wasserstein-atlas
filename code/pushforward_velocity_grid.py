"""A map-induced spatial velocity, illustrated on a transported uniform grid.

Run: python visualization_scripts/pushforward_velocity_grid.py
Requires numpy and matplotlib. Saves PNG/PDF in output/ and captions alongside
this script. The map is globally invertible for theta_2>0. It is nonlinear in
space but affine in theta, so displayed epsilon-scaled tangent arrows reach the
new grid nodes exactly for the chosen straight parameter path.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

THETA = np.array([.75, 1.15])
DTHETA = np.array([.85, -.45])
EPSILON = .45


def phi(points, theta):
    points = np.asarray(points)
    return np.stack([points[..., 0]+theta[0]*np.sin(points[..., 1]),
                     theta[1]*points[..., 1]], axis=-1)


def phi_inverse(points, theta):
    points = np.asarray(points)
    return np.stack([points[..., 0]-theta[0]*np.sin(points[..., 1]/theta[1]),
                     points[..., 1]/theta[1]], axis=-1)


def parameter_velocity(points, dtheta):
    """d_theta phi_theta(dtheta)(x), expressed at reference points x."""
    return np.stack([dtheta[0]*np.sin(points[..., 1]), dtheta[1]*points[..., 1]], axis=-1)


def spatial_velocity(points, theta, dtheta):
    """v_{theta,dtheta}(y), expressed at current positions y."""
    return np.stack([dtheta[0]*np.sin(points[..., 1]/theta[1]),
                     dtheta[1]*points[..., 1]/theta[1]], axis=-1)


def main():
    theta_next = THETA+EPSILON*DTHETA
    assert THETA[1] > 0 and theta_next[1] > 0
    coordinates = np.linspace(-2.4, 2.4, 7)
    X1, X2 = np.meshgrid(coordinates, coordinates)
    nodes = np.stack([X1, X2], axis=-1).reshape(-1, 2)
    current = phi(nodes, THETA)
    following = phi(nodes, theta_next)
    velocities = parameter_velocity(nodes, DTHETA)
    assert np.allclose(phi_inverse(current, THETA), nodes)
    assert np.allclose(spatial_velocity(current, THETA, DTHETA), velocities)
    assert np.allclose(following, current+EPSILON*velocities)
    h = 1e-6
    assert np.allclose((phi(nodes, THETA+h*DTHETA)-phi(nodes, THETA-h*DTHETA))/(2*h), velocities)
    # At y_2=0 the chosen velocity is zero; dots there deliberately have no arrow.
    assert np.allclose(velocities[np.isclose(nodes[:, 1], 0)], 0)
    smooth = np.linspace(-2.4, 2.4, 400)
    lines = []
    for coordinate in coordinates:
        lines.extend([np.column_stack([np.full_like(smooth, coordinate), smooth]),
                      np.column_stack([smooth, np.full_like(smooth, coordinate)])])
    selected = np.array([.8, 1.6])
    selected_current = phi(selected, THETA)
    selected_next = phi(selected, theta_next)
    plt.rcParams.update({'font.family': 'serif', 'mathtext.fontset': 'cm', 'font.size': 11,
                         'axes.titlesize': 15, 'axes.labelsize': 14, 'savefig.facecolor': 'white'})
    blue, red, gold = '#326a9b', '#bf5360', '#bf8914'
    fig, axes = plt.subplots(1, 3, figsize=(16.2, 8.3))
    fig.subplots_adjust(left=.055, right=.975, bottom=.31, top=.745, wspace=.20)
    fig.suptitle('A parameter direction becomes a velocity field on the transported grid', y=.965, fontsize=21)
    fig.text(.5, .899, r'$\phi_\theta(x_1,x_2)=(x_1+\theta_1\sin x_2,\ \theta_2x_2),'
             r'\qquad \theta=(0.75,1.15),\quad d\theta=(0.85,-0.45)$', ha='center', fontsize=17)
    fig.text(.5, .848, r'$\theta(t)=\theta+t\,d\theta,\qquad \theta^{\prime}(t)=d\theta,'
             r'\qquad \varepsilon=0.45$', ha='center', fontsize=15, color='.3')
    titles = ['(a) Reference grid', '(b) Current transported grid', '(c) Velocity and the next grid']
    for i, ax in enumerate(axes):
        for line in lines:
            if i == 0:
                ax.plot(*line.T, color='#7f8d99', lw=.85, alpha=.75)
            else:
                ax.plot(*phi(line, THETA).T, color=blue, lw=1., alpha=.85 if i == 1 else .5)
                if i == 2:
                    ax.plot(*phi(line, theta_next).T, color=red, lw=.9, ls=(0, (4, 3)), alpha=.7)
        points = nodes if i == 0 else current
        ax.scatter(*points.T, s=10, color='#667989' if i == 0 else blue, zorder=4)
        if i == 2:
            ax.quiver(*current.T, *(EPSILON*velocities).T, angles='xy', scale_units='xy', scale=1,
                      color=gold, width=.0045, headwidth=3.5, headlength=4.5, zorder=7)
        marker = selected if i == 0 else selected_current
        ax.scatter(*marker, s=56, color=red, edgecolor='white', linewidth=.9, zorder=8)
        if i == 2:
            ax.scatter(*selected_next, s=30, facecolor='white', edgecolor=red, linewidth=1.3, zorder=8)
        else:
            label = r'$x$' if i == 0 else r'$y=\phi_\theta(x)$'
            ax.annotate(label, marker, xytext=(8, 10), textcoords='offset points', fontsize=14,
                        color=red, bbox=dict(fc='white', ec='none', alpha=.9, pad=1))
        ax.set(xlim=(-3.8, 3.8), ylim=(-3.2, 3.2), aspect='equal',
               xlabel=r'$x_1$' if i == 0 else r'$y_1$', ylabel=r'$x_2$' if i == 0 else r'$y_2$')
        ax.set_xticks([-3, -2, -1, 0, 1, 2, 3]); ax.set_yticks([-3, -2, -1, 0, 1, 2, 3])
        ax.tick_params(labelsize=9, length=3)
        ax.spines[['top', 'right']].set_visible(False)
        ax.set_title(titles[i], pad=14)
    axes[2].legend(handles=[Line2D([], [], color=blue, lw=1.2, label=r'$\phi_\theta$'),
                            Line2D([], [], color=red, ls='--', lw=1.2, label=r'$\phi_{\theta+\varepsilon d\theta}$'),
                            Line2D([], [], color=gold, lw=2, label=r'$\varepsilon v_{\theta,d\theta}$')],
                   loc='lower center', bbox_to_anchor=(.5, -.23), ncol=3, frameon=False,
                   fontsize=10, columnspacing=1.0, handlelength=1.5)
    fig.text(.5, .166,
             r'$v_{\theta,d\theta}\left(\phi_\theta(x)\right)'
             r'=d_\theta\phi_\theta(d\theta)(x)=(d\theta_1\sin x_2,\ d\theta_2x_2)$',
             ha='center', fontsize=18)
    fig.text(.5, .105,
             r'$v_{\theta,d\theta}(y)='
             r'\left(d\theta_1\sin\!\left(\frac{y_2}{\theta_2}\right),'
             r'\ d\theta_2\frac{y_2}{\theta_2}\right)$',
             ha='center', fontsize=18)
    fig.text(.5, .055, 'Arrows start at current positions and show the velocity multiplied by '+r'$\varepsilon$'+'. '
             'The red dot tracks one reference node.', ha='center', fontsize=11)
    fig.text(.5, .023, 'Here the map is affine in the parameters: every arrow tip lies exactly on the corresponding next grid node.',
             ha='center', fontsize=10, color='.35')
    folder = Path(__file__).resolve().parent
    out = folder/'output'; out.mkdir(exist_ok=True)
    for ext in ['png', 'pdf']:
        fig.savefig(out/f'pushforward_velocity_grid.{ext}', dpi=220)
    plt.close(fig)
    caption = (
        'From a parameter direction to a spatial velocity field. The globally invertible deformation '
        'φ_θ(x₁,x₂)=(x₁+θ₁sin x₂,θ₂x₂), with Θ=R×(0,∞), combines a sinusoidal horizontal shear '
        'and vertical scaling; its spatial Jacobian determinant is θ₂>0. '
        '(a) A uniform reference grid in the x coordinates. (b) Its image under φ_θ at '
        'θ=(0.75,1.15), in the current y coordinates. The red dot identifies the same reference '
        'node x=(0.8,1.6) in both panels. (c) The current grid (solid blue), the nearby grid '
        'φ_{θ+εdθ} (dashed red), and the induced velocity arrows for dθ=(0.85,−0.45), ε=0.45. '
        'Along θ(t)=θ+t dθ, the parameter tangent is θ′(t)=dθ. Differentiating a particle trajectory '
        'at fixed reference x gives d_θφ_θ(dθ)(x)=(dθ₁ sin x₂,dθ₂x₂). Reading the same vector '
        'at its current position y=φ_θ(x) gives v_{θ,dθ}(y)=d_θφ_θ(dθ)(φ_θ⁻¹(y)) '
        '=(dθ₁ sin(y₂/θ₂),dθ₂y₂/θ₂). Each gold arrow is anchored at a current node and has '
        'displacement εv_{θ,dθ}(y); velocities are zero on the central row y₂=0. The open red '
        'circle marks the selected node’s next position. Because this map is affine in θ, '
        'φ_{θ+εdθ}(x)=φ_θ(x)+εd_θφ_θ(dθ)(x) exactly, so arrow tips coincide with their '
        'corresponding next grid nodes. For a general nonlinear parameter dependence, this '
        'agreement is only to first order. All panels share spatial limits and equal aspect. '
        'For any admissible reference density q, this map-induced field realizes the density '
        'variation through d_θp_θ(dθ)=−div_y(p_θv_{θ,dθ}); it is not necessarily the '
        'minimum-energy gradient field used by the Wasserstein–Otto metric.'
    )
    for name in ['pushforward_velocity_grid_caption.txt', 'caption.txt']:
        (folder/name).write_text(caption+'\n', encoding='utf-8')
    print('Saved figure and captions. Verified inverse, parameter derivative, spatial velocity, and arrow endpoints.')


if __name__ == '__main__':
    main()
