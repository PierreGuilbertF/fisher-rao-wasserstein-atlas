"""Four two-parameter Gibbs families on R or R^2.

Run: python visualization_scripts/gibbs_family_examples.py
Requires numpy and matplotlib. Outputs PNG/PDF and numerical checks to output/,
and both a named caption and caption.txt. Potentials and parameter gradients
are exposed below for reuse in subsequent metric computations.

'Linear' always means U_theta(x)=phi(x)^T theta in the displayed parameters.
The first three normalizers use converged quadrature; the banana normalizer
is exact: integrating the conditional Gaussian gives Z=2*pi*theta_2.
"""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

PARAMETERS = {
    'linear_1d': np.array([.9, .35]),
    'nonlinear_1d': np.array([1.4, 2.4]),
    'linear_2d': np.array([.3, 1.1]),
    'nonlinear_2d': np.array([.65, .55]),
}


def linear_1d(x, theta):
    """Theta=R x (0,infinity); sufficient statistics (x,x^4)."""
    return theta[0]*x + theta[1]*x**4


def grad_linear_1d(x, theta):
    return np.stack([x, x**4], axis=-1)


def nonlinear_1d(x, theta):
    """Theta=(0,infinity)^2: modulation amplitude and spatial frequency."""
    return x**4/4 + theta[0]*np.cos(theta[1]*x)


def grad_nonlinear_1d(x, theta):
    return np.stack([np.cos(theta[1]*x), -theta[0]*x*np.sin(theta[1]*x)], axis=-1)


def linear_2d(x1, x2, theta):
    """Theta=(0,infinity)^2; sufficient statistics (x1^4+x2^4,(x1-x2)^2)."""
    return theta[0]*(x1**4+x2**4) + theta[1]*(x1-x2)**2


def grad_linear_2d(x1, x2, theta):
    return np.stack([x1**4+x2**4, (x1-x2)**2], axis=-1)


def nonlinear_2d(x1, x2, theta):
    """Theta=R x (0,infinity): curvature and conditional width."""
    return x1**2/2 + (x2-theta[0]*x1**2)**2/(2*theta[1]**2)


def grad_nonlinear_2d(x1, x2, theta):
    return np.stack([-(x2-theta[0]*x1**2)*x1**2/theta[1]**2,
                     -(x2-theta[0]*x1**2)**2/theta[1]**3], axis=-1)


def integrate_2d(values, x1, x2):
    return np.trapezoid(np.trapezoid(values, x1, axis=1), x2, axis=0)


def compute_normalizers():
    result = {}
    for name, potential in [('linear_1d', linear_1d), ('nonlinear_1d', nonlinear_1d)]:
        estimates = []
        for bound, count in [(5., 6001), (7., 12001)]:
            x = np.linspace(-bound, bound, count)
            estimates.append(np.trapezoid(np.exp(-potential(x, PARAMETERS[name])), x))
        assert np.isclose(*estimates, rtol=1e-8)
        result[name] = float(estimates[-1])
    estimates = []
    for bound, count in [(4., 701), (5., 1001)]:
        x = np.linspace(-bound, bound, count)
        X1, X2 = np.meshgrid(x, x)
        estimates.append(integrate_2d(np.exp(-linear_2d(X1, X2, PARAMETERS['linear_2d'])), x, x))
    assert np.isclose(*estimates, rtol=1e-8)
    result['linear_2d'] = float(estimates[-1])
    result['nonlinear_2d'] = float(2*np.pi*PARAMETERS['nonlinear_2d'][1])
    # Independent 2D quadrature verifies the analytic banana normalizer.
    x1, x2 = np.linspace(-7., 7., 901), np.linspace(-6., 38., 1401)
    X1, X2 = np.meshgrid(x1, x2)
    numeric = integrate_2d(np.exp(-nonlinear_2d(X1, X2, PARAMETERS['nonlinear_2d'])), x1, x2)
    assert np.isclose(numeric, result['nonlinear_2d'], rtol=1e-8)
    return result


def verify_gradients():
    for name, potential, gradient, points in [
        ('linear_1d', linear_1d, grad_linear_1d, (np.array([-.9, .4, 1.3]),)),
        ('nonlinear_1d', nonlinear_1d, grad_nonlinear_1d, (np.array([-.9, .4, 1.3]),)),
        ('linear_2d', linear_2d, grad_linear_2d, (np.array([-.9, .4, 1.3]), np.array([.2, -.7, 1.]))),
        ('nonlinear_2d', nonlinear_2d, grad_nonlinear_2d, (np.array([-.9, .4, 1.3]), np.array([.2, -.7, 1.]))),
    ]:
        theta = PARAMETERS[name]
        for i in range(2):
            increment = np.zeros(2); increment[i] = 1e-6
            difference = (potential(*points, theta+increment)-potential(*points, theta-increment))/(2e-6)
            assert np.allclose(difference, gradient(*points, theta)[..., i], rtol=1e-6, atol=1e-8)


def main():
    normalizers = compute_normalizers()
    verify_gradients()
    plt.rcParams.update({'font.family': 'serif', 'mathtext.fontset': 'cm', 'font.size': 11,
                         'axes.titlesize': 14, 'axes.labelsize': 13, 'savefig.facecolor': 'white'})
    blue = '#326a9b'
    density_cmap = LinearSegmentedColormap.from_list('gibbs_blue',
        [(0., (.90, .96, 1., 0.)), (.25, (.55, .76, .90, .40)),
         (.65, (.19, .45, .68, .85)), (1., (.03, .16, .33, 1.))])
    fig = plt.figure(figsize=(14, 11))
    fig.suptitle('Four Gibbs families: linear and nonlinear potentials', fontsize=22, y=.974)
    fig.text(.5, .928, r'$p_\theta(x)=Z_\theta^{-1}e^{-U_\theta(x)},\qquad\theta=(\theta_1,\theta_2)$',
             ha='center', fontsize=19)
    fig.text(.5, .89, 'Linearity refers to the parameters, not to the spatial variable.',
             ha='center', fontsize=12, color='.35')
    axes = [fig.add_axes([.075, .575, .365, .215]), fig.add_axes([.57, .575, .365, .215]),
            fig.add_axes([.075, .13, .365, .29]), fig.add_axes([.57, .13, .365, .29])]
    titles = ['(a) Linear · 1D', '(b) Nonlinear · 1D', '(c) Linear · 2D', '(d) Nonlinear · 2D']
    formulas = [r'$U_\theta(x)=\theta_1x+\theta_2x^4$',
                r'$U_\theta(x)=\frac{x^4}{4}+\theta_1\cos(\theta_2x)$',
                r'$U_\theta(x_1,x_2)=\theta_1(x_1^4+x_2^4)+\theta_2(x_1-x_2)^2$',
                r'$U_\theta(x_1,x_2)=\frac{x_1^2}{2}+\frac{(x_2-\theta_1x_1^2)^2}{2\theta_2^2}$']
    for index, ax in enumerate(axes):
        col, row = index%2, index//2
        center = .2575+col*.495
        fig.text(center, .842 if row == 0 else .487, titles[index], ha='center', fontsize=16)
        fig.text(center, .809 if row == 0 else .449, formulas[index], ha='center', fontsize=15)
        ax.spines[['top', 'right']].set_visible(False)
    x = np.linspace(-3.1, 3.1, 1601)
    top_values = [np.exp(-potential(x, PARAMETERS[name]))/normalizers[name]
                  for name, potential in [('linear_1d', linear_1d), ('nonlinear_1d', nonlinear_1d)]]
    top_max = max(values.max() for values in top_values)*1.22
    for ax, name, values in zip(axes[:2], ['linear_1d', 'nonlinear_1d'], top_values):
        ax.plot(x, values, color=blue, lw=2.4)
        ax.fill_between(x, 0, values, color=blue, alpha=.14)
        ax.set(xlim=(-3.1, 3.1), ylim=(0, top_max), xlabel=r'$x$', ylabel=r'$p_\theta(x)$')
        ax.set_xticks([-3, -2, -1, 0, 1, 2, 3])
        ax.text(.97, .89, rf'$\theta=({PARAMETERS[name][0]:g},{PARAMETERS[name][1]:g})$',
                transform=ax.transAxes, ha='right', fontsize=12)
        ax.grid(alpha=.12)
    displayed_mass = {}
    bottom = [('linear_2d', linear_2d, (-2.5, 2.5), (-2.5, 2.5)),
              ('nonlinear_2d', nonlinear_2d, (-3.2, 3.2), (-2., 6.))]
    for ax, (name, potential, xlim, ylim) in zip(axes[2:], bottom):
        x1, x2 = np.linspace(*xlim, 501), np.linspace(*ylim, 601)
        X1, X2 = np.meshgrid(x1, x2)
        values = np.exp(-potential(X1, X2, PARAMETERS[name]))/normalizers[name]
        displayed_mass[name] = float(integrate_2d(values, x1, x2))
        mesh = ax.imshow(values, origin='lower', extent=[*xlim, *ylim], cmap=density_cmap,
                         vmin=0, vmax=values.max(), interpolation='bilinear', aspect='equal')
        levels = values.max()*np.array([.08, .25, .5, .8])
        ax.contour(X1, X2, values, levels=levels, colors='#356584', linewidths=.7, alpha=.75)
        ax.set(xlim=xlim, ylim=ylim, xlabel=r'$x_1$', ylabel=r'$x_2$')
        ax.text(.97, .95, rf'$\theta=({PARAMETERS[name][0]:g},{PARAMETERS[name][1]:g})$',
                transform=ax.transAxes, ha='right', va='top', fontsize=12,
                bbox=dict(fc='white', ec='none', alpha=.85, pad=2))
        fig.colorbar(mesh, ax=ax, fraction=.035, pad=.035, label=r'$p_\theta(x_1,x_2)$')
    fig.text(.5, .058, 'All four families are defined on the whole space and have two parameters. '
             'The 2D colorbars show probability density.', ha='center', fontsize=11)
    fig.text(.5, .025, '1D: density curves. 2D: blue density maps with constant-density contours; '
             'the two maps have separate color scales.', ha='center', fontsize=10, color='.35')
    folder = Path(__file__).resolve().parent
    out = folder/'output'; out.mkdir(exist_ok=True)
    for ext in ['png', 'pdf']:
        fig.savefig(out/f'gibbs_family_examples.{ext}', dpi=220)
    plt.close(fig)
    caption = (
        'Four smooth two-parameter Gibbs families p_θ=Z_θ⁻¹ exp(−U_θ), defined on R (top) '
        'or R² (bottom). Linearity refers to the dependence of U_θ on θ, not to its dependence on x. '
        '(a) Linear 1D: U_θ(x)=θ₁x+θ₂x⁴, with Θ=R×(0,∞), shown at θ=(0.9,0.35). '
        'The linear tilt produces an asymmetric, non-Gaussian density; φ(x)=(x,x⁴)ᵀ. '
        '(b) Nonlinear 1D: U_θ(x)=x⁴/4+θ₁cos(θ₂x), with Θ=(0,∞)², shown at θ=(1.4,2.4). '
        'The parameters control the amplitude and frequency of a bounded modulation inside '
        'a confining quartic potential, producing multiple wells. '
        '(c) Linear 2D: U_θ(x₁,x₂)=θ₁(x₁⁴+x₂⁴)+θ₂(x₁−x₂)², with Θ=(0,∞)², shown at '
        'θ=(0.3,1.1). Quartic confinement and a quadratic coupling create a correlated '
        'non-Gaussian density; φ(x₁,x₂)=(x₁⁴+x₂⁴,(x₁−x₂)²)ᵀ. '
        '(d) Nonlinear 2D: U_θ(x₁,x₂)=x₁²/2+(x₂−θ₁x₁²)²/(2θ₂²), with Θ=R×(0,∞), '
        'shown at θ=(0.65,0.55). Here θ₁ controls curvature of the high-density region '
        'and θ₂ its conditional vertical width. The first marginal is standard Gaussian '
        'and the conditional second coordinate is Gaussian with mean θ₁x₁² and standard '
        'deviation θ₂, so Z_θ=2πθ₂. All four families are normalizable on their full spatial '
        'domains and have finite moments of every order; the windows shown are for display, '
        'not truncations defining the probability laws. The first three normalization '
        'constants were evaluated by numerical quadrature and checked under refinement '
        'and enlargement of the integration domains. The top panels share axes; each bottom '
        'panel uses equal spatial aspect and its own labeled density color scale. Blue fades '
        'to transparent as density decreases, and contours indicate constant density '
        '(8%, 25%, 50% and 80% of each displayed peak), not prescribed enclosed probabilities. '
        'The potentials and their parameter gradients are retained in the script for subsequent '
        'metric computations.'
    )
    for name in ['gibbs_family_examples_caption.txt', 'caption.txt']:
        (folder/name).write_text(caption+'\n', encoding='utf-8')
    diagnostics = {'parameters': {name: value.tolist() for name, value in PARAMETERS.items()},
                   'normalizers': normalizers, 'mass_in_2d_display_windows': displayed_mass}
    (out/'gibbs_family_examples_diagnostics.json').write_text(json.dumps(diagnostics, indent=2)+'\n')
    print(json.dumps(diagnostics, indent=2))
    print('Verified normalization convergence and analytic parameter gradients. Saved figure and captions.')


if __name__ == '__main__':
    main()
