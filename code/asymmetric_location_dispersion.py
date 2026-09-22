"""Location--dispersion transformations of a numerically standardized density.

Run: python visualization_scripts/asymmetric_location_dispersion.py
Requires numpy, scipy, matplotlib. Saves PNG/PDF in output/ next to script.
Normalization, centering, and variance rescaling use numerical quadrature;
no analytic mixture moments are used to construct the reference density.
"""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad


def main():
    plt.rcParams.update({
        'font.family': 'serif', 'mathtext.fontset': 'cm', 'font.size': 11,
        'axes.titlesize': 14, 'axes.labelsize': 12, 'legend.fontsize': 10,
        'savefig.facecolor': 'white',
    })
    def raw(x):
        return (.8*np.exp(-.5*((x+.8)/.45)**2)
                +.25*np.exp(-.5*((x-1.5)/.65)**2))

    integrate = lambda function: quad(function, -np.inf, np.inf,
                                      epsabs=1e-10, epsrel=1e-10, limit=200)[0]
    normalization = integrate(raw)
    original_mean = integrate(lambda x: x*raw(x))/normalization
    original_sd = np.sqrt(integrate(lambda x: (x-original_mean)**2*raw(x))/normalization)

    def p(x):
        # If Y has normalized density raw/Z, X=(Y-mean)/sd has this density.
        return original_sd*raw(original_mean+original_sd*x)/normalization

    mass = integrate(p)
    mean = integrate(lambda x: x*p(x))
    second_moment = integrate(lambda x: x*x*p(x))
    assert np.allclose([mass, mean, second_moment], [1, 0, 1], atol=1e-8)
    assert not np.allclose(p(np.linspace(0, 3, 100)), p(-np.linspace(0, 3, 100)))
    scenarios = [
        (0., 1., '(a) Reference density', r'$p(x)$'),
        (1.6, 1., '(b) Shift only', r'$p_{1.6,1}(x)=p(x-1.6)$'),
        (0., 1.6, '(c) Enlargement only', r'$p_{0,1.6}(x)=\frac{1}{1.6}p\!\left(\frac{x}{1.6}\right)$'),
        (1.6, 1.6, '(d) Shift and enlargement', r'$p_{1.6,1.6}(x)=\frac{1}{1.6}p\!\left(\frac{x-1.6}{1.6}\right)$'),
    ]
    x = np.linspace(-5, 7, 2600)
    reference = p(x)
    ymax = reference.max()*1.32
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.8))
    fig.subplots_adjust(left=.075, right=.965, bottom=.13, top=.825,
                        hspace=.52, wspace=.19)
    fig.suptitle('One asymmetric reference density, four location–dispersion choices',
                 y=.972, fontsize=18)
    fig.text(.5, .916,
             r'$p_{\mu,\sigma}(x)=\frac{1}{\sigma}p\!\left(\frac{x-\mu}{\sigma}\right),'
             r'\qquad \int_{\mathbb{R}}xp(x)\,dx=0,\quad'
             r'\int_{\mathbb{R}}x^2p(x)\,dx=1$', ha='center', fontsize=14)
    colors = ['#326A9B', '#BE4A51', '#258178', '#82629D']
    for ax, (mu, sigma, title, formula), color in zip(axes.flat, scenarios, colors):
        def transformed(value):
            return p((value-mu)/sigma)/sigma
        # Verify each displayed density keeps unit mass and has the stated moments.
        assert np.isclose(integrate(transformed), 1, atol=1e-8)
        assert np.isclose(integrate(lambda value: value*transformed(value)), mu, atol=1e-8)
        assert np.isclose(integrate(lambda value: (value-mu)**2*transformed(value)), sigma**2, atol=1e-8)
        values = transformed(x)
        if mu != 0 or sigma != 1:
            ax.plot(x, reference, color='.60', ls='--', lw=1.2,
                    label=r'$p(x)$ (reference)', zorder=1)
        ax.plot(x, values, color=color, lw=2.5, label=r'$p_{\mu,\sigma}(x)$', zorder=3)
        ax.fill_between(x, 0, values, color=color, alpha=.13)
        ax.axvline(mu, color=color, ls=':', lw=1.1, alpha=.75)
        ax.set(xlim=(-5,7), ylim=(0,ymax), xlabel=r'$x$', ylabel='Probability density')
        ax.set_title(title, loc='left', pad=12)
        ax.text(.975,.93, rf'$\mu={mu:g},\quad\sigma={sigma:g}$',
                transform=ax.transAxes, ha='right', va='top', fontsize=12,
                bbox=dict(facecolor='white', edgecolor='none', alpha=.9, pad=2))
        if mu != 0 or sigma != 1:
            ax.legend(loc='upper left', frameon=False, fontsize=9)
        ax.text(.5,-.25,formula,transform=ax.transAxes, ha='center', fontsize=12)
        ax.spines[['top','right']].set_visible(False)
        ax.tick_params(labelsize=10)
        ax.set_xticks([-4,-2,0,2,4,6])
    fig.text(.5,.038,
             'The reference was normalized, centered, and scaled numerically. '
             'Dotted lines mark the means; dashed curves repeat the reference.',
             ha='center', fontsize=10, color='.35')
    out = Path(__file__).resolve().parent/'output'
    out.mkdir(exist_ok=True)
    fig.savefig(out / "asymmetric_location_dispersion.pdf")
    fig.savefig(out / "asymmetric_location_dispersion.png", dpi=220)
    plt.close(fig)
    print(f'Numerical raw normalization={normalization:.10f}, mean={original_mean:.10f}, sd={original_sd:.10f}')
    print(f'Standardized reference: mass={mass:.10f}, mean={mean:.10f}, second moment={second_moment:.10f}')
    print('All four transformed densities: unit mass, expected means and variances verified.')


if __name__ == '__main__':
    main()
