"""Four-panel illustration of the Wasserstein--Otto pullback norm.

Run: python visualization_scripts/quantile_pullback_norm.py
Requires numpy, scipy, matplotlib. Writes PNG and PDF in output/ next to
this script. All plotted variations are derivatives, not finite increments.
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad
from scipy.special import ndtr, ndtri


def main():
    plt.rcParams.update({
        'font.family': 'serif', 'mathtext.fontset': 'cm', 'font.size': 11,
        'axes.titlesize': 14, 'axes.labelsize': 12, 'legend.fontsize': 10,
        'savefig.facecolor': 'white',
    })
    mu, sigma = 0.0, 1.0
    dmu, dsigma = 0.6, 0.4
    red, blue, green = '#BF4348', '#326A9B', '#258178'
    x = np.linspace(-4.5, 4.5, 2200)
    z = (x-mu)/sigma
    p = np.exp(-z*z/2)/(sigma*np.sqrt(2*np.pi))
    velocity = dmu + dsigma*z
    u = p/sigma * (dmu*z + dsigma*(z*z-1))
    F = ndtr(z)
    U = -p*velocity
    # Sampling through z resolves the steep quantile tails without using r=0,1.
    r = ndtr(z)
    Q = mu + sigma*ndtri(r)
    dQ = dmu + dsigma*ndtri(r)
    energy_x = p*velocity**2  # U^2/p, evaluated without division in the tails.
    energy_r = dQ**2
    exact = dmu**2 + dsigma**2
    normal = lambda t: np.exp(-t*t/2)/np.sqrt(2*np.pi)
    integral_x = quad(lambda t: (dmu+dsigma*t)**2*normal(t), -np.inf, np.inf)[0]
    integral_r = quad(lambda t: (dmu+dsigma*ndtri(t))**2, 0, 1,
                      epsabs=1e-10, limit=200)[0]
    assert np.allclose([integral_x, integral_r], exact, atol=1e-8)
    assert np.allclose(dQ, -U/p, atol=1e-8)

    fig = plt.figure(figsize=(13.8, 10.6))
    grid = fig.add_gridspec(2, 2, left=.075, right=.965, top=.855,
                           bottom=.12, hspace=.39, wspace=.24)
    ax_p = fig.add_subplot(grid[0, 0])
    ax_f = fig.add_subplot(grid[0, 1])
    ax_q = fig.add_subplot(grid[1, 0])
    # Two separate coordinate systems inside the fourth panel: x and r.
    energy_grid = grid[1, 1].subgridspec(2, 1, hspace=.78)
    ax_ex = fig.add_subplot(energy_grid[0])
    ax_er = fig.add_subplot(energy_grid[1])
    axes = [ax_p, ax_f, ax_q, ax_ex, ax_er]
    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False)
        ax.axhline(0, color='.78', lw=.8, zorder=0)
        ax.tick_params(labelsize=10)

    fig.suptitle('The same tangent direction measured in two spaces', y=.975, fontsize=19)
    fig.text(.5, .928,
             r'$p_\theta=\mathcal{N}(\mu,\sigma^2),\quad\theta=(0,1),'
             r'\quad d\theta=(0.6,0.4)$', ha='center', fontsize=15)
    fig.text(.5, .89,
             r'$\|d\theta\|_{g_\theta^{\mathrm{WO}}}'
             r'=\|d_\theta Q_\theta(d\theta)\|_{L^2(0,1)}'
             r'=\sqrt{0.52}\simeq 0.7211$', ha='center', fontsize=16)

    ax_p.set_title(r'(a) Density and density variation', loc='left', pad=12)
    ax_p.plot(x, p, color=blue, lw=2.3, label=r'$p_\theta(x)$')
    ax_p.plot(x, u, color=red, lw=2, alpha=.8,
              label=r'$u(x)=d_\theta p_\theta(d\theta)(x)$')
    ax_p.fill_between(x, 0, u, color=red, alpha=.22)
    ax_p.set(xlim=(-4.5, 4.5), ylim=(-.32, .62), xlabel=r'$x$')
    ax_p.legend(frameon=False, loc='upper left')

    ax_f.set_title(r'(b) Cumulative function and its variation', loc='left', pad=12)
    ax_f.plot(x, F, color=blue, lw=2.3, label=r'$F_\theta(x)$')
    ax_f.plot(x, U, color=red, lw=2, alpha=.8,
              label=r'$U(x)=d_\theta F_\theta(d\theta)(x)$')
    ax_f.fill_between(x, 0, U, color=red, alpha=.22)
    ax_f.set(xlim=(-4.5, 4.5), ylim=(-.38, 1.18), xlabel=r'$x$')
    ax_f.legend(frameon=False, loc='upper left')

    ax_q.set_title(r'(c) Quantile function and its variation', loc='left', pad=12)
    ax_q.plot(r, Q, color=blue, lw=2.3, label=r'$Q_\theta(r)$')
    ax_q.plot(r, dQ, color=red, lw=2, alpha=.8,
              label=r'$d_\theta Q_\theta(d\theta)(r)$')
    ax_q.fill_between(r, 0, dQ, color=red, alpha=.22)
    ax_q.set(xlim=(0, 1), ylim=(-4.6, 4.6), xlabel=r'$r$')
    ax_q.legend(frameon=False, loc='upper left')
    ax_q.text(.5, .035,
              r'$d_\theta Q_\theta(d\theta)(r)=0.6+0.4\,\Phi^{-1}(r)$',
              transform=ax_q.transAxes, ha='center', fontsize=12,
              bbox=dict(facecolor='white', edgecolor='none', alpha=.9, pad=3))

    ax_ex.set_title('(d) Equal areas: the squared norms', loc='left', pad=12)
    ax_ex.plot(x, energy_x, color=green, lw=2)
    ax_ex.fill_between(x, 0, energy_x, color=green, alpha=.25)
    ax_ex.set(xlim=(-4.5, 4.5), ylim=(0, .38), xlabel=r'$x$',
              ylabel=r'$U(x)^2/p_\theta(x)$')
    ax_ex.text(.025, .83,
               r'$\int_{\mathbb{R}} U^2/p_\theta\,dx=0.52$',
               transform=ax_ex.transAxes, fontsize=12)
    ax_er.plot(r, energy_r, color=green, lw=2)
    ax_er.fill_between(r, 0, energy_r, color=green, alpha=.25)
    ax_er.set(xlim=(0, 1), ylim=(0, 6), xlabel=r'$r$',
              ylabel=r'$|d_\theta Q_\theta(d\theta)(r)|^2$')
    ax_er.text(.025, .80,
               r'$\int_0^1|d_\theta Q_\theta(d\theta)(r)|^2\,dr=0.52$',
               transform=ax_er.transAxes, fontsize=12)
    fig.text(.5, .055,
             r'$\|d\theta\|_{g_\theta^{\mathrm{WO}}}^2'
             r'=\int_{\mathbb{R}}\frac{U(x)^2}{p_\theta(x)}\,dx'
             r'=\int_0^1|d_\theta Q_\theta(d\theta)(r)|^2\,dr'
             r'=\|d_\theta Q_\theta(d\theta)\|_{L^2(0,1)}^2$',
             ha='center', fontsize=16)
    fig.text(.5, .013,
             'Curves are displayed over a finite range; integral values use the full domains.',
             ha='center', fontsize=10, color='.4')
    out = Path(__file__).resolve().parent/'output'
    out.mkdir(exist_ok=True)
    for ext in ('png', 'pdf'):
        target = out/f'{Path(__file__).stem}.{ext}'
        fig.savefig(target, dpi=220, bbox_inches='tight', pad_inches=.14)
        print(target)
    plt.close(fig)
    print(f'Full-domain squared norms: spatial={integral_x:.10f}, '
          f'quantile={integral_r:.10f}, exact={exact:.10f}')


if __name__ == '__main__':
    main()
