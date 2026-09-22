"""Exact intrinsic distance maps for Gaussian Fisher--Rao and Wasserstein--Otto.

Run: python visualization_scripts/gaussian_metric_distance_maps.py
Dependencies: numpy, matplotlib.
Outputs: output/gaussian_metric_distance_maps.png and .pdf, relative to script.
The two panels share parameter limits, axis aspect, contour levels, and color
normalization. Distances (not squared distances) are displayed.
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np


def fisher_distance(mu, sigma, mu_ref, sigma_ref):
    argument = 1 + ((mu-mu_ref)**2 + 2*(sigma-sigma_ref)**2)/(4*sigma*sigma_ref)
    return np.sqrt(2)*np.arccosh(np.maximum(argument, 1))


def wasserstein_distance(mu, sigma, mu_ref, sigma_ref):
    return np.hypot(mu-mu_ref, sigma-sigma_ref)


def main():
    plt.rcParams.update({
        'font.family': 'serif', 'mathtext.fontset': 'cm', 'font.size': 11,
        'axes.titlesize': 15, 'axes.labelsize': 14, 'savefig.facecolor': 'white',
    })
    mu_ref, sigma_ref = 0.0, 0.22
    mu_bounds, sigma_bounds = (-1.6, 1.6), (0.025, 2.50)
    mu = np.linspace(*mu_bounds, 850)
    sigma = np.linspace(*sigma_bounds, 760)
    MU, SIGMA = np.meshgrid(mu, sigma)
    fr = fisher_distance(MU, SIGMA, mu_ref, sigma_ref)
    wo = wasserstein_distance(MU, SIGMA, mu_ref, sigma_ref)
    # Independent closed-form check along the vertical Gaussian geodesic.
    sample_sigmas = np.array([.03, .1, .22, .7, 2.4])
    assert np.allclose(fisher_distance(mu_ref, sample_sigmas, mu_ref, sigma_ref),
                       np.sqrt(2)*np.abs(np.log(sample_sigmas/sigma_ref)))
    assert fisher_distance(mu_ref, sigma_ref, mu_ref, sigma_ref) == 0
    assert wasserstein_distance(mu_ref, sigma_ref, mu_ref, sigma_ref) == 0
    # The same radius produces different contour centers for the two metrics.
    for radius in [.5, 1, 2]:
        angle = np.linspace(0, 2*np.pi, 100)
        fr_mu = mu_ref + np.sqrt(2)*sigma_ref*np.sinh(radius/np.sqrt(2))*np.cos(angle)
        fr_sigma = sigma_ref*np.cosh(radius/np.sqrt(2)) + sigma_ref*np.sinh(radius/np.sqrt(2))*np.sin(angle)
        assert np.allclose(fisher_distance(fr_mu, fr_sigma, mu_ref, sigma_ref), radius)

    vmax = np.ceil(max(fr.max(), wo.max())*2)/2
    norm = Normalize(0, vmax)
    levels = [.25, .5, 1, 1.5, 2, 3, 4, 5]
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 7.6))
    fig.subplots_adjust(left=.07, right=.88, bottom=.19, top=.80, wspace=.20)
    fig.suptitle('Gaussian family: distance from a narrow reference distribution',
                 y=.96, fontsize=18)
    fig.text(.47, .898, r'$\Theta=\mathbb{R}\times(0,\infty),\qquad'
             r'(\mu_\star,\sigma_\star)=(0,0.22)$', ha='center', fontsize=15)
    titles = [r'Fisher--Rao: $ds^2=(d\mu^2+2\,d\sigma^2)/\sigma^2$',
              r'Wasserstein--Otto: $ds^2=d\mu^2+d\sigma^2$']
    for ax, field, title in zip(axes, [fr, wo], titles):
        mesh = ax.pcolormesh(MU, SIGMA, field, shading='auto', cmap='coolwarm',
                             norm=norm, rasterized=True)
        visible_levels = [value for value in levels if value < field.max()]
        contours = ax.contour(MU, SIGMA, field, levels=visible_levels,
                              colors='#233344', linewidths=.85, alpha=.85)
        # Prefer automatic placement: manual anchors for large Fisher radii
        # often fall outside the clipped contour arcs and crash clabel.
        label_levels = [value for value in visible_levels if value >= 1]
        if label_levels:
            ax.clabel(contours, levels=label_levels, fmt='%g',
                      inline=True, inline_spacing=5, fontsize=9)
        ax.scatter([mu_ref], [sigma_ref], marker='*', s=175, color='white',
                   edgecolor='#192C43', linewidth=1.1, zorder=5)
        ax.annotate(r'$\theta_\star$', xy=(mu_ref, sigma_ref),
                    xytext=(10, -16), textcoords='offset points', fontsize=12,
                    color='#172B45', bbox=dict(facecolor='white', alpha=.85,
                                              edgecolor='none', pad=1.5))
        ax.set(xlim=mu_bounds, ylim=sigma_bounds, xlabel=r'Mean $\mu$',
               ylabel=r'Standard deviation $\sigma$')
        ax.set_aspect('equal', adjustable='box')
        ax.set_title(title, fontsize=13, pad=16)
        ax.spines[['top', 'right']].set_visible(False)
        ax.set_xticks([-1.5, -1, -.5, 0, .5, 1, 1.5])
        ax.set_yticks([.25, .5, 1, 1.5, 2, 2.5])
    bar_ax = fig.add_axes([.91, .265, .018, .44])
    bar = fig.colorbar(mesh, cax=bar_ax)
    bar.set_label('Distance from '+r'$\theta_\star$'+' (shared scale)', labelpad=12)
    fig.text(.27, .12,
             r'$d_{\mathrm{FR}}=\sqrt{2}\,\operatorname{arcosh}'
             r'\!\left(1+\frac{\mu^2+2(\sigma-0.22)^2}{4\sigma\,(0.22)}\right)$',
             ha='center', fontsize=13)
    fig.text(.69, .12,
             r'$d_{\mathrm{WO}}=\sqrt{\mu^2+(\sigma-0.22)^2}$',
             ha='center', fontsize=13)
    fig.text(.5, .055,
             'Identical axes, contour levels, and color scale in both panels. '
             'The star marks the same reference Gaussian.',
             ha='center', fontsize=11)
    fig.text(.5, .02,
             r'The boundary $\sigma=0$ is excluded: it lies at infinite Fisher--Rao '
             r'distance and finite Wasserstein--Otto distance.',
             ha='center', fontsize=11, color='.30')
    out = Path(__file__).resolve().parent/'output'
    out.mkdir(exist_ok=True)
    fig.savefig(out / "gaussian_metric_distance_maps.pdf")
    fig.savefig(out / "gaussian_metric_distance_maps.png", dpi=220)
    plt.close(fig)
    print(f'Checked exact vertical distances and Fisher contour equations. Shared color range: 0 to {vmax:g}.')
    print(f'Wrote {out / "gaussian_metric_distance_maps.pdf"}')
    print(f'Wrote {out / "gaussian_metric_distance_maps.png"}')

if __name__ == '__main__':
    main()
