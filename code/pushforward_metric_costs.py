"""Six-panel comparison of FR and WO costs for a nonlinear push-forward.

Run: python visualization_scripts/pushforward_metric_costs.py
Requires numpy and matplotlib. Outputs PNG/PDF, captions and diagnostics.
Map: phi_theta(x)=(x1+theta1*tanh(x1), theta2*x2), theta1>0,theta2>0.
Its induced velocity is separable and therefore a spatial gradient. Hence
p*|v|^2 is exactly the WO energy density, not the cost of an unprojected field.
All normalization and energy integrations use whole-space reference formulas.
"""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm

THETA = np.array([1.8, .8])
DTHETA = np.array([.9, .35])
ARROW_SCALE = .55


def phi(points, theta=THETA):
    points = np.asarray(points)
    return np.stack([points[..., 0]+theta[0]*np.tanh(points[..., 0]),
                     theta[1]*points[..., 1]], axis=-1)


def inverse_first(y1, theta=THETA):
    # Monotonicity and |tanh|<=1 give an exact bracket for the inverse.
    lo, hi = np.asarray(y1)-theta[0], np.asarray(y1)+theta[0]
    for _ in range(55):
        mid = (lo+hi)/2
        residual = mid+theta[0]*np.tanh(mid)-y1
        lo = np.where(residual < 0, mid, lo)
        hi = np.where(residual >= 0, mid, hi)
    return (lo+hi)/2


def fields(y1, y2, theta=THETA):
    x1, x2 = inverse_first(y1, theta), np.asarray(y2)/theta[1]
    tangent = np.tanh(x1)
    sech_squared = 1-tangent**2
    jacobian1 = 1+theta[0]*sech_squared
    jacobian1_derivative = -2*theta[0]*sech_squared*tangent
    q = np.exp(-(x1**2+x2**2)/2)/(2*np.pi)
    p = q/(theta[1]*jacobian1)
    v1, v2 = DTHETA[0]*tangent, DTHETA[1]*x2
    divergence = DTHETA[0]*sech_squared/jacobian1+DTHETA[1]/theta[1]
    relative_rate = (divergence+v1*(-x1-jacobian1_derivative/jacobian1)/jacobian1
                     -DTHETA[1]*x2**2/theta[1])
    u = -p*relative_rate
    return p, v1, v2, u, p*relative_rate**2, p*(v1**2+v2**2)


def integrated_costs():
    x = np.linspace(-9, 9, 18001)
    q = np.exp(-x*x/2)/np.sqrt(2*np.pi)
    tangent = np.tanh(x); sech_squared = 1-tangent**2
    jacobian1 = 1+THETA[0]*sech_squared
    jacobian1_derivative = -2*THETA[0]*sech_squared*tangent
    first_rate = DTHETA[0]*(sech_squared/jacobian1
                  +tangent*(-x-jacobian1_derivative/jacobian1)/jacobian1)
    assert abs(np.trapezoid(q*first_rate, x)) < 1e-10
    # The second-coordinate rate is (dtheta2/theta2)*(1-X2^2), with
    # mean zero and second moment 2*(dtheta2/theta2)^2 under standard Gaussian q.
    fr = np.trapezoid(q*first_rate**2, x)+2*(DTHETA[1]/THETA[1])**2
    wo = DTHETA[0]**2*np.trapezoid(q*tangent**2, x)+DTHETA[1]**2
    return float(fr), float(wo)


def verify():
    points = np.array([[-2., -1.], [-.7, .6], [0., 0.], [.4, -.7], [1.8, 1.2]])
    mapped = phi(points)
    assert np.allclose(inverse_first(mapped[:, 0]), points[:, 0])
    h = 1e-5
    p, v1, v2, u, _, _ = fields(mapped[:, 0], mapped[:, 1])
    difference = (fields(mapped[:, 0], mapped[:, 1], THETA+h*DTHETA)[0]
                 -fields(mapped[:, 0], mapped[:, 1], THETA-h*DTHETA)[0])/(2*h)
    assert np.allclose(difference, u, atol=1e-9, rtol=1e-6)
    def flux(y1, y2, component):
        values = fields(y1, y2)
        return values[0]*values[component]
    divergence = ((flux(mapped[:, 0]+h, mapped[:, 1], 1)-flux(mapped[:, 0]-h, mapped[:, 1], 1))/(2*h)
                  +(flux(mapped[:, 0], mapped[:, 1]+h, 2)-flux(mapped[:, 0], mapped[:, 1]-h, 2))/(2*h))
    assert np.allclose(u, -divergence, atol=1e-9, rtol=1e-6)
    velocity = (phi(points, THETA+h*DTHETA)-phi(points, THETA-h*DTHETA))/(2*h)
    assert np.allclose(velocity, np.column_stack([v1, v2]), rtol=1e-6, atol=1e-9)


def main():
    verify()
    fr_total, wo_total = integrated_costs()
    y1, y2 = np.linspace(-5.5, 5.5, 661), np.linspace(-3.3, 3.3, 401)
    Y1, Y2 = np.meshgrid(y1, y2)
    p, v1, v2, u, fr_cost, wo_cost = fields(Y1, Y2)
    reference = np.exp(-(Y1**2+Y2**2)/2)/(2*np.pi)
    integrate = lambda values: np.trapezoid(np.trapezoid(values, y1, axis=1), y2)
    assert integrate(p) > .999
    assert abs(integrate(u)) < .002
    assert np.isclose(integrate(fr_cost), fr_total, rtol=.015)
    assert np.isclose(integrate(wo_cost), wo_total, rtol=.003)
    plt.rcParams.update({'font.family': 'serif', 'mathtext.fontset': 'cm', 'font.size': 11,
                         'axes.labelsize': 12, 'axes.titlesize': 14, 'savefig.facecolor': 'white'})
    blue_cmap = LinearSegmentedColormap.from_list('density_blue',
                   [(0, (1, 1, 1, 0)), (.35, (.50, .72, .86, .5)), (1, (.04, .22, .41, 1))])
    cost_cmap = LinearSegmentedColormap.from_list('cost_warm', ['#ffffff', '#ffe3a2', '#f7944c', '#bd2631', '#581641'])
    density_norm = Normalize(0, reference.max())
    cost_norm = Normalize(0, max(fr_cost.max(), wo_cost.max()))
    fig, axes = plt.subplots(2, 3, figsize=(17, 10.2))
    fig.subplots_adjust(left=.055, right=.965, bottom=.22, top=.78, wspace=.30, hspace=.50)
    fig.suptitle('One density variation, two geometric costs', y=.966, fontsize=23)
    fig.text(.5, .917, r'$q=\mathcal{N}(0,I_2),\qquad'
             r'\phi_\theta(x_1,x_2)=(x_1+\theta_1\tanh x_1,\ \theta_2x_2)$', ha='center', fontsize=18)
    fig.text(.5, .87, r'$\theta=(1.8,0.8),\qquad d\theta=(0.9,0.35),\qquad'
             r'v_{\theta,d\theta}(\phi_\theta(x))=(d\theta_1\tanh x_1,\ d\theta_2x_2)$', ha='center', fontsize=16)
    fig.text(.5, .832, 'This induced velocity is a gradient: no WO projection is needed.', ha='center', fontsize=12, color='.35')
    titles = ['(a) Reference Gaussian and grid', '(b) Deformed density and grid', '(c) Induced velocity on the density',
              '(d) Signed density variation', '(e) Fisher–Rao cost density', '(f) Wasserstein–Otto cost density']
    for i, ax in enumerate(axes.flat):
        ax.set_title(titles[i], fontsize=13, pad=12)
        ax.set(xlim=(-5.5, 5.5), ylim=(-3.3, 3.3), aspect='equal',
               xlabel=r'$x_1$' if i == 0 else r'$y_1$', ylabel=r'$x_2$' if i == 0 else r'$y_2$')
        ax.set_xticks([-4, -2, 0, 2, 4]); ax.set_yticks([-2, 0, 2]); ax.tick_params(labelsize=9)
        ax.spines[['top', 'right']].set_visible(False)
    for i, ax in enumerate(axes[0]):
        values = reference if i == 0 else p
        mesh = ax.imshow(values, origin='lower', extent=[y1[0], y1[-1], y2[0], y2[-1]],
                         cmap=blue_cmap, norm=density_norm, interpolation='bilinear')
        ax.contour(Y1, Y2, values, levels=[.012, .035, .065, .11], colors='#326a9b', linewidths=.8, alpha=.85)
    grid_ticks = np.linspace(-2.5, 2.5, 9)
    smooth = np.linspace(-2.5, 2.5, 300)
    for tick in grid_ticks:
        for line in [np.column_stack([smooth, np.full_like(smooth, tick)]),
                     np.column_stack([np.full_like(smooth, tick), smooth])]:
            axes[0, 0].plot(*line.T, color='.5', lw=.6, alpha=.5)
            axes[0, 1].plot(*phi(line).T, color='.45', lw=.6, alpha=.6)
    density_bar = fig.colorbar(mesh, ax=axes[0, 2], fraction=.036, pad=.025)
    density_bar.set_label('Density (top row)', fontsize=9)
    density_bar.ax.tick_params(labelsize=8)
    a, b = np.meshgrid(np.linspace(-2.6, 2.6, 11), np.linspace(-2.8, 2.8, 9))
    nodes = np.stack([a, b], axis=-1)
    mapped = phi(nodes)
    induced = np.stack([DTHETA[0]*np.tanh(a), DTHETA[1]*b], axis=-1)
    axes[0, 2].quiver(mapped[..., 0], mapped[..., 1], ARROW_SCALE*induced[..., 0], ARROW_SCALE*induced[..., 1],
                       color='#ad790b', angles='xy', scale_units='xy', scale=1, width=.004,
                       headwidth=3.5, headlength=4, zorder=5)
    axes[0, 2].text(.04, .92, r'Arrows: $0.55\,v_{\theta,d\theta}$', transform=axes[0, 2].transAxes, fontsize=9)
    signed = axes[1, 0].imshow(u, origin='lower', extent=[y1[0], y1[-1], y2[0], y2[-1]],
                               cmap='coolwarm', norm=TwoSlopeNorm(vmin=-np.abs(u).max(), vcenter=0, vmax=np.abs(u).max()))
    signed_bar = fig.colorbar(signed, ax=axes[1, 0], fraction=.036, pad=.025)
    signed_bar.set_label(r'$u$ (red: increase; blue: decrease)', fontsize=9)
    signed_bar.ax.tick_params(labelsize=8)
    for ax, values in zip(axes[1, 1:], [fr_cost, wo_cost]):
        cost_mesh = ax.imshow(values, origin='lower', extent=[y1[0], y1[-1], y2[0], y2[-1]],
                             cmap=cost_cmap, norm=cost_norm, interpolation='bilinear')
    for ax in axes[1]:
        ax.contour(Y1, Y2, p, levels=[.012, .035], colors='#536d83', linewidths=.6, alpha=.55)
    cost_bar = fig.colorbar(cost_mesh, ax=axes[1, 2], fraction=.036, pad=.025)
    cost_bar.set_label('Cost density (shared for e–f)', fontsize=9)
    cost_bar.ax.tick_params(labelsize=8)
    labels = [r'$u=d_\theta p_\theta(d\theta)=-\mathrm{div}_y(p_\theta v_{\theta,d\theta})$',
              r'$\frac{[\mathrm{div}_y(p_\theta v_{\theta,d\theta})]^2}{p_\theta}$'+'\n'+rf'Integral: ${fr_total:.3f}$',
              r'$p_\theta\|v_{\theta,d\theta}\|^2$'+'\n'+rf'Integral: ${wo_total:.3f}$']
    for ax, label in zip(axes[1], labels):
        ax.text(.5, -.23, label, transform=ax.transAxes, ha='center', va='top', fontsize=12, linespacing=1.7)
    fig.text(.5, .067, 'Fisher–Rao weights relative density change; WO weights the squared speed needed to move mass.',
             ha='center', fontsize=12)
    fig.text(.5, .028, 'Identical spatial axes. The two cost maps share one color scale; their integrals are squared tangent norms, not distances.',
             ha='center', fontsize=11, color='.35')
    folder = Path(__file__).resolve().parent
    out = folder/'output'; out.mkdir(exist_ok=True)
    for ext in ['png', 'pdf']:
        fig.savefig(out/f'pushforward_metric_costs.{ext}', dpi=220)
    plt.close(fig)
    caption = (
        'One push-forward density variation viewed through Fisher–Rao and Wasserstein–Otto. '
        'The reference density is q=N(0,I₂), and φ_θ(x₁,x₂)=(x₁+θ₁tanh x₁,θ₂x₂), '
        'with θ₁>0 and θ₂>0. Its determinant θ₂(1+θ₁sech²x₁) is positive everywhere, '
        'so the map is a global diffeomorphism. The figure uses θ=(1.8,0.8) and dθ=(0.9,0.35). '
        '(a) The isotropic reference Gaussian with a uniform grid. (b) The transported density '
        'p_θ=(φ_θ)♯q and the image of the same grid. (c) The induced field '
        'v_{θ,dθ}(φ_θ(x))=(dθ₁tanh x₁,dθ₂x₂), with arrows scaled by 0.55 for display. '
        'Its first component depends only on y₁ and its second only on y₂, so it is a '
        'spatial gradient on R² and is already the minimum-energy velocity for this variation. '
        '(d) The signed density variation u=−div_y(p_θv_{θ,dθ}), with red indicating an increase '
        'and blue a decrease. (e) The nonnegative Fisher–Rao integrand u²/p_θ. '
        '(f) The nonnegative WO integrand p_θ||v_{θ,dθ}||². The last expression is exact here '
        'because v is a gradient; an arbitrary map-induced velocity would first require its '
        'weighted gradient projection. Faint contours in the bottom row repeat the current '
        'density levels, helping locate the cost relative to the probability mass. '
        f'The whole-space integrals are g_FR(dθ,dθ)={fr_total:.6f} and '
        f'g_WO(dθ,dθ)={wo_total:.6f}; these are squared infinitesimal norms, not endpoint distances. '
        'All panels have identical spatial limits and equal aspect. The top row shares a density '
        'normalization; the two cost maps share a separate numerical color scale. '
        'Only panel (d) uses a signed color scale. The density differential, continuity equation '
        'and map velocity were verified independently by finite differences; total costs were '
        'evaluated in reference coordinates. The arrows’ display factor is not included in the costs.'
    )
    for name in ['pushforward_metric_costs_caption.txt', 'caption.txt']:
        (folder/name).write_text(caption+'\n', encoding='utf-8')
    diagnostics = {'FR_squared_norm': fr_total, 'WO_squared_norm': wo_total,
                   'display_mass': float(integrate(p)), 'display_integral_u': float(integrate(u)),
                   'display_integral_FR': float(integrate(fr_cost)), 'display_integral_WO': float(integrate(wo_cost))}
    (out/'pushforward_metric_costs_diagnostics.json').write_text(json.dumps(diagnostics, indent=2)+'\n')
    print(json.dumps(diagnostics, indent=2))


if __name__ == '__main__':
    main()
