"""Translation of narrow/broad Gaussian references: local FR and WO costs.

Run: python visualization_scripts/pushforward_translation_metrics.py
Requires numpy and matplotlib. Saves PNG/PDF in visualization_scripts/output.
Each row is a separate fixed-shape translation family; sigma is not varied
along the tangent. All panels use physical coordinates and common column scales.
"""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SIGMAS = (0.6, 1.5)
MU = 0.0
DMU = 0.6
EPSILON = 0.5
BLUE, RED, PURPLE, TEAL = '#2867a3', '#c95548', '#8350a0', '#16877e'


def fields(x, sigma, mu=MU):
    p = np.exp(-0.5*((x-mu)/sigma)**2)/(np.sqrt(2*np.pi)*sigma)
    relative_rate = DMU*(x-mu)/sigma**2
    u = p*relative_rate
    return p, u, p*relative_rate**2, p*DMU**2


def verify():
    results = []
    for sigma in SIGMAS:
        x = np.linspace(-10*sigma, 10*sigma, 30001)
        p, u, fr, wo = fields(x, sigma)
        h = 1e-5
        derivative = (fields(x, sigma, MU+h*DMU)[0]
                      -fields(x, sigma, MU-h*DMU)[0])/(2*h)
        assert np.allclose(derivative, u, atol=1e-9, rtol=1e-6)
        actual = [np.trapezoid(a, x) for a in (p, u, fr, wo)]
        expected = [1, 0, DMU**2/sigma**2, DMU**2]
        assert np.allclose(actual, expected, atol=1e-10)
        results.append(dict(sigma=sigma, mass=float(actual[0]),
                            integral_u=float(actual[1]),
                            fisher_rao_squared_norm=float(actual[2]),
                            wasserstein_otto_squared_norm=float(actual[3])))
    return results


CAPTION = r'''Translation of a narrow and a broad Gaussian: Fisher–Rao and Wasserstein–Otto costs. Each row is a separate fixed-shape family p_mu(y) = q(y-mu), with q = N(0,sigma^2), sigma = 0.6 (top) or 1.5 (bottom). In both rows mu = 0 and dmu = 0.6. The map phi_mu(x) = x+mu induces the same constant velocity v = dmu, whose ordinary divergence is zero. First column: the current density (solid) and its exact translation by epsilon dmu = 0.3 (dashed, epsilon = 0.5); equal arrows show this displayed displacement. Second column: the infinitesimal density variation u = d_mu p_mu(dmu) = -dmu partial_y p_mu, which is negative behind and positive ahead of the moving density. Thus zero divergence of v does not imply zero density change: -partial_y(p_mu v) is nonzero. Third column: the Fisher–Rao local squared-cost density u^2/p_mu = p_mu[dmu(y-mu)/sigma^2]^2, with integral (dmu)^2/sigma^2. Fourth column: the Wasserstein–Otto local squared-cost density p_mu|v|^2, with integral (dmu)^2. This constant velocity is a gradient and is the minimum-energy velocity realizing u, so its kinetic energy equals the WO squared norm. FR charges more for translating the narrow density, whose relative density change is larger; WO assigns the same integrated cost to the same displacement at either width. The printed integrals are squared tangent norms, not norms, and use dmu rather than the finite display increment epsilon dmu. Axes are shared within each column; both cost columns also use the same vertical scale, and shaded areas represent their integrals. The panels compare local costs, not geodesics in a family with variable sigma.'''


def main():
    diagnostics = verify()
    plt.rcParams.update({'font.family': 'serif', 'font.size': 11,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'mathtext.fontset': 'dejavuserif', 'savefig.dpi': 220})
    x = np.linspace(-5, 5, 2001)
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), sharex=True,
                             sharey='col')
    titles = ['A constant translation', 'Density variation', 'Fisher–Rao', 'Wasserstein–Otto']
    for j, title in enumerate(titles):
        axes[0, j].set_title(title, fontsize=15, pad=16)
    for i, sigma in enumerate(SIGMAS):
        p, u, fr, wo = fields(x, sigma)
        ax = axes[i, 0]
        ax.plot(x, p, color=BLUE, lw=2, label=r'$p_\mu$')
        ax.plot(x, fields(x, sigma, MU+EPSILON*DMU)[0], color=RED,
                lw=1.8, ls='--', label=r'$p_{\mu+\varepsilon d\mu}$')
        for anchor in (-1.8, -0.9, 0, .9, 1.8):
            ax.annotate('', (anchor+EPSILON*DMU, .75), (anchor, .75),
                        arrowprops=dict(arrowstyle='->', color=RED, lw=1.4))
        ax.text(.5, .88, r'$v(y)=d\mu=0.6$', transform=ax.transAxes, ha='center')
        ax.set_ylabel(('Narrow' if i == 0 else 'Broad') + '\n' + rf'$\sigma={sigma}$', fontsize=14)
        ax.set_ylim(0, .91)
        ax.legend(loc='upper right', frameon=False, fontsize=10, bbox_to_anchor=(1, .8))
        ax = axes[i, 1]
        ax.axhline(0, color='#888888', lw=.7)
        ax.plot(x, u, color=RED, lw=1.8)
        ax.fill_between(x, 0, u, where=u >= 0, color=RED, alpha=.23)
        ax.fill_between(x, 0, u, where=u < 0, color=BLUE, alpha=.23)
        ax.text(.5, .92, r'$u=-d\mu\,\partial_y p_\mu$', transform=ax.transAxes, ha='center')
        ax.set_ylim(-.48, .58)
        for j, values, color, formula, total in (
            (2, fr, PURPLE, r'$u^2/p_\mu$', DMU**2/sigma**2),
            (3, wo, TEAL, r'$p_\mu|v|^2$', DMU**2)):
            ax = axes[i, j]
            ax.plot(x, values, color=color, lw=2)
            ax.fill_between(x, 0, values, color=color, alpha=.25)
            ax.text(.5, .92, formula, transform=ax.transAxes, ha='center', fontsize=14)
            ax.text(.5, .8, f'Integral = {total:.2f}', transform=ax.transAxes,
                    ha='center', color=color, fontsize=12,
                    bbox=dict(facecolor='white', edgecolor='none', alpha=.85))
            ax.set_ylim(0, .67)
        for ax in axes[i]:
            ax.set_xlim(-5, 5)
            ax.set_xticks([-4, -2, 0, 2, 4])
            ax.grid(alpha=.14, lw=.6)
            if i == 1:
                ax.set_xlabel('$y$')
    fig.suptitle('Same translation, different density widths', fontsize=21, y=.975)
    fig.text(.5, .914, r'$\phi_\mu(x)=x+\mu,\quad \mu=0,\quad d\mu=0.6$'
             '     |     Dashed density and arrows: 'r'$\varepsilon d\mu=0.3$', ha='center', fontsize=12)
    fig.text(.5, .035, r'FR: $(d\mu)^2/\sigma^2$ — sensitive to width'
             '                  'r'WO: $(d\mu)^2$ — cost of displacement',
             ha='center', fontsize=14)
    fig.subplots_adjust(left=.065, right=.985, bottom=.12, top=.83, wspace=.23, hspace=.2)
    base = Path(__file__).resolve().parent
    output = base/'output'
    output.mkdir(exist_ok=True)
    for ext in ('png', 'pdf'):
        fig.savefig(output/f'pushforward_translation_metrics.{ext}', bbox_inches='tight')
    plt.close(fig)
    for name in ('pushforward_translation_metrics_caption.txt', 'caption.txt'):
        (base/name).write_text(CAPTION+'\n', encoding='utf-8')
    (output/'pushforward_translation_metrics_diagnostics.json').write_text(
        json.dumps(diagnostics, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(diagnostics, indent=2))


if __name__ == '__main__':
    main()
