"""Equal-rank transport: quantile gaps and CDF correspondences.

Run: python visualization_scripts/wasserstein_1d_correspondence.py
Dependencies: numpy, scipy, matplotlib.
Saves PNG and PDF to visualization_scripts/output, independently of the cwd.
The shaded inset integrates the SQUARED quantile gap, not the gap itself.
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import numpy as np
from scipy.integrate import quad
from scipy.special import ndtr, ndtri


def main():
    plt.rcParams.update({
        'font.family': 'serif', 'mathtext.fontset': 'cm', 'font.size': 11,
        'axes.titlesize': 14, 'axes.labelsize': 13, 'legend.fontsize': 10,
        'savefig.facecolor': 'white',
    })
    mu0, sigma0, mu1, sigma1 = -.8, .7, .9, 1.1
    q0 = lambda r: mu0 + sigma0*ndtri(r)
    q1 = lambda r: mu1 + sigma1*ndtri(r)
    f0 = lambda x: ndtr((x-mu0)/sigma0)
    f1 = lambda x: ndtr((x-mu1)/sigma1)
    delta = lambda r: np.abs(q1(r)-q0(r))
    exact_squared = (mu1-mu0)**2 + (sigma1-sigma0)**2
    computed = quad(lambda r: float(delta(r)**2), 0, 1,
                    epsabs=1e-10, limit=200)[0]
    assert np.isclose(computed, exact_squared, atol=1e-8)
    levels = [.10, .35, .65, .90]
    for r_level in levels:
        assert np.allclose([f0(q0(r_level)), f1(q1(r_level))], r_level)

    blue, red, green = '#326A9B', '#B74C54', '#258178'
    rank_colors = ['#75569A', '#B87522', '#2F8B6A', '#527B9D']
    r = ndtr(np.linspace(-4, 4, 2600))
    x = np.linspace(-3.5, 4.1, 2200)
    r_low, r_high = .625, .675
    fig, (ax_q, ax_f) = plt.subplots(1, 2, figsize=(14.4, 7.5))
    fig.subplots_adjust(left=.07, right=.96, bottom=.23, top=.80, wspace=.23)
    fig.suptitle('One-dimensional Wasserstein distance: matching equal probability ranks',
                 y=.975, fontsize=18)
    fig.text(.5, .918,
             r'$p_0=\mathcal{N}(-0.8,0.7^2),\qquad p_1=\mathcal{N}(0.9,1.1^2)$',
             ha='center', fontsize=14)
    for ax in (ax_q, ax_f):
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(labelsize=10)

    ax_q.set_title('(a) Quantiles: vertical displacements', loc='left', pad=14)
    ax_q.plot(r, q0(r), lw=2.4, color=blue, label=r'$Q_0(r)$')
    ax_q.plot(r, q1(r), lw=2.4, color=red, label=r'$Q_1(r)$')
    ax_q.set(xlim=(0, 1), ylim=(-3.5, 4.1), xlabel=r'Probability rank $r$',
             ylabel='Position')
    ax_q.legend(loc='upper left', frameon=False)
    ax_q.axvspan(r_low, r_high, color=green, alpha=.10, zorder=0)
    ax_q.text(.65, 3.76, r'$\delta r=0.05$', ha='center', fontsize=10, color=green)

    ax_f.set_title('(b) CDFs: horizontal transport', loc='left', pad=14)
    ax_f.plot(x, f0(x), lw=2.4, color=blue, label=r'$F_0(x)$')
    ax_f.plot(x, f1(x), lw=2.4, color=red, label=r'$F_1(x)$')
    ax_f.set(xlim=(-3.5, 4.1), ylim=(0, 1.06), xlabel=r'Position $x$',
             ylabel=r'Cumulative probability $r$')
    ax_f.legend(loc='lower right', frameon=False)
    ax_f.axhspan(r_low, r_high, color=green, alpha=.10, zorder=0)
    # Thicker CDF pieces identify the same mass interval in both distributions.
    r_band = np.linspace(r_low, r_high, 120)
    ax_f.plot(q0(r_band), r_band, color=blue, lw=5, alpha=.65)
    ax_f.plot(q1(r_band), r_band, color=red, lw=5, alpha=.65)
    ax_f.annotate('', xy=(3.55, r_high), xytext=(3.55, r_low),
                  arrowprops=dict(arrowstyle='|-|', color=green, lw=1.4))
    ax_f.text(3.67, .65, r'$\delta r$', va='center', fontsize=11, color=green)

    for rank, color in zip(levels, rank_colors):
        a, b = float(q0(rank)), float(q1(rank))
        ax_q.add_patch(FancyArrowPatch((rank, a), (rank, b),
                       arrowstyle='<->', mutation_scale=10, color=color, lw=1.5))
        ax_q.scatter([rank, rank], [a, b], s=22, color=color, zorder=5)
        ax_q.text(rank+.018, (a+b)/2, rf'$\Delta({rank:.2f})$',
                  fontsize=9, color=color, va='center',
                  bbox=dict(facecolor='white', edgecolor='none', alpha=.8, pad=.5))
        ax_f.hlines(rank, -3.5, a, color=color, lw=.7, ls=':', alpha=.65)
        ax_f.add_patch(FancyArrowPatch((a, rank), (b, rank), arrowstyle='->',
                       mutation_scale=11, color=color, lw=1.6))
        ax_f.scatter([a, b], [rank, rank], s=24, color=color, zorder=5)
        ax_f.text((a+b)/2, rank+.023, rf'$r={rank:.2f}:\ \Delta={b-a:.2f}$',
                  ha='center', color=color, fontsize=9,
                  bbox=dict(facecolor='white', edgecolor='none', alpha=.85, pad=.5))

    # A separate inset shows precisely the integrand of W_2^2.
    inset = ax_q.inset_axes([.54, .10, .43, .22])
    inset.set_facecolor('#FAFAFA')
    inset.plot(r, delta(r)**2, color=green, lw=1.4)
    inset.fill_between(r, 0, delta(r)**2, color=green, alpha=.24)
    inset.set(xlim=(0, 1), ylim=(0, 11.3), xticks=[0, .5, 1], yticks=[0, 5, 10])
    inset.tick_params(labelsize=8, pad=2)
    inset.set_title(r'$\Delta(r)^2\quad\mathrm{area}=W_2^2=3.05$', fontsize=10, pad=5)
    inset.set_xlabel(r'$r$', fontsize=9, labelpad=0)
    inset.spines[['top', 'right']].set_visible(False)

    fig.text(.5, .142,
             r'$F_0(Q_0(r))=F_1(Q_1(r))=r,\qquad'
             r'\Delta(r)=|Q_1(r)-Q_0(r)|$', ha='center', fontsize=15)
    fig.text(.5, .085,
             r'$W_2^2(p_0,p_1)=\int_0^1\Delta(r)^2\,dr=3.05,'
             r'\qquad W_2(p_0,p_1)=\sqrt{3.05}\simeq1.7464$',
             ha='center', fontsize=15)
    fig.text(.5, .037,
             'The highlighted rank interval carries mass '+r'$\delta r$'+
             '; its cost is approximately '+r'$\Delta(r)^2\delta r$'+'.',
             ha='center', fontsize=11)
    fig.text(.5, .004,
             'Quantile tails are cropped for display; the reported distance uses the full interval (0, 1).',
             ha='center', fontsize=9, color='.4')
    out = Path(__file__).resolve().parent/'output'
    out.mkdir(exist_ok=True)
    for extension in ('png', 'pdf'):
        destination = out/f'{Path(__file__).stem}.{extension}'
        fig.savefig(destination, dpi=220, bbox_inches='tight', pad_inches=.15)
        print(destination)
    plt.close(fig)
    print(f'Squared distance: numerical={computed:.10f}, exact={exact_squared:.10f}')


if __name__ == '__main__':
    main()
