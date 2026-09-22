"""Illustrate a Gaussian density variation, its CDF, and its quantile.

Run: python density_variation_cdf_quantile.py
Requires numpy, scipy, matplotlib. Outputs are relative to this script.
U is the parameter variation of F, not F itself. Only F and Q are inverses.
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import ndtr, ndtri


def main():
    plt.rcParams.update({
        'font.family': 'serif', 'mathtext.fontset': 'cm', 'font.size': 11,
        'axes.titlesize': 13, 'axes.labelsize': 12, 'legend.fontsize': 10,
        'savefig.facecolor': 'white',
    })
    mu, sigma, dmu = 0.40, 0.22, 0.10
    lo, hi = -0.35, 1.35
    x = np.linspace(lo, hi, 1800)
    z = (x - mu) / sigma
    p = np.exp(-z*z/2) / (sigma*np.sqrt(2*np.pi))
    u = dmu * (x-mu) * p / sigma**2
    U = -dmu*p
    F = ndtr(z)
    r = np.linspace(ndtr((lo-mu)/sigma), ndtr((hi-mu)/sigma), 1800)
    Q = mu + sigma*ndtri(r)
    # Analytic inverse and finite-difference checks of the parameter derivative.
    assert np.allclose(ndtr((Q-mu)/sigma), r, atol=1e-12)
    eps = 1e-6
    p_shift = lambda m: np.exp(-((x-m)/sigma)**2/2)/(sigma*np.sqrt(2*np.pi))
    assert np.allclose((p_shift(mu+eps*dmu)-p_shift(mu-eps*dmu))/(2*eps), u, atol=1e-8)
    assert np.allclose((ndtr((x-mu-eps*dmu)/sigma)-ndtr((x-mu+eps*dmu)/sigma))/(2*eps), U, atol=1e-8)

    red, blue, green = '#BF4348', '#326A9B', '#258178'
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 5.2))
    fig.subplots_adjust(left=.06, right=.985, bottom=.19, top=.79, wspace=.29)
    fig.suptitle('Density variation, cumulative distribution, and quantile', y=.975, fontsize=16)
    fig.text(.5, .895,
             r'$p_\theta=\mathcal{N}(\mu,\sigma^2),\quad \theta=(0.40,0.22),\quad d\theta=(0.10,0)$',
             ha='center', fontsize=13)
    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False)
        ax.axhline(0, color='.80', lw=.7, zorder=0)
        ax.set_box_aspect(1)
        ax.tick_params(labelsize=9)
    ax = axes[0]
    ax.plot(x, u, color=red, alpha=.72, lw=2.3)
    ax.fill_between(x, 0, u, color=red, alpha=.24)
    ax.set(xlim=(lo, hi), ylim=(-.58, .58), xlabel=r'$x$', ylabel=r'$u(x)$')
    ax.set_title(r'(a) $u=d_\theta p_\theta(d\theta)$', pad=13)
    ax.text(.04, .94, r'$\int_{\mathbb{R}}u(x)\,dx=0$', transform=ax.transAxes, va='top')
    ax.text(.5, -.24, r'$u(x)=d\mu\,\dfrac{x-\mu}{\sigma^2}p_\theta(x)$',
            transform=ax.transAxes, ha='center', fontsize=12)

    for ax in axes[1:]:
        ax.set(xlim=(lo, hi), ylim=(lo, hi))
        ax.set_aspect('equal', adjustable='box')
        ax.plot([lo, hi], [lo, hi], '--', color='.65', lw=1, label=r'$y=x$')
    ax = axes[1]
    ax.plot(x, F, color=blue, lw=2.5, label=r'$F_\theta(x)$')
    ax.plot(x, U, color=red, lw=1.7, ls='--', label=r'$U(x)=d_\theta F_\theta(d\theta)(x)$')
    ax.set(xlabel=r'$x$', ylabel='Function value')
    ax.set_title(r'(b) $F_\theta$ and its variation $U$', pad=13)
    ax.legend(loc='upper left', frameon=False, fontsize=9)
    ax.text(.5, -.24, r'$F_\theta(x)=\Phi((x-\mu)/\sigma),\quad U(x)=-d\mu\,p_\theta(x)$',
            transform=ax.transAxes, ha='center', fontsize=10)

    ax = axes[2]
    ax.plot(r, Q, color=green, lw=2.5, label=r'$Q_\theta(r)$')
    ax.plot(x, F, color=blue, lw=1.5, alpha=.35, label=r'$F_\theta$ (reference)')
    # The reflected point pair and connecting perpendicular to y=x expose inversion.
    r_star = .8
    x_star = mu + sigma*ndtri(r_star)
    ax.plot([x_star, r_star], [r_star, x_star], ':', color='.45', lw=1)
    ax.scatter([x_star], [r_star], color=blue, s=26, zorder=4)
    ax.scatter([r_star], [x_star], color=green, s=26, zorder=4)
    axes[1].scatter([x_star], [r_star], color=blue, s=26, zorder=4)
    ax.set(xlabel=r'$r$ (quantile input)', ylabel=r'$Q_\theta(r)$')
    ax.set_title(r'(c) $Q_\theta=F_\theta^{-1}$', pad=13)
    ax.legend(loc='upper left', frameon=False, fontsize=9)
    ax.text(.5, -.24, r'$Q_\theta(r)=\mu+\sigma\Phi^{-1}(r),\quad 0<r<1$',
            transform=ax.transAxes, ha='center', fontsize=11)
    out = Path(__file__).resolve().parent / 'output'
    out.mkdir(exist_ok=True)
    # Prefer the PDF in the paper (vector). Keep a high-DPI PNG for preview.
    fig.savefig(out / 'density_variation_cdf_quantile.pdf')
    fig.savefig(out / 'density_variation_cdf_quantile.png', dpi=400)
    plt.close(fig)


if __name__ == '__main__':
    main()
