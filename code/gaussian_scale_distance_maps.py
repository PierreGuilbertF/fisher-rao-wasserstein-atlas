"""Exact FR and WO distances for fixed-mean, fixed-orientation 2D Gaussians.

Run: python visualization_scripts/gaussian_scale_distance_maps.py
Requires numpy and matplotlib. Saves PNG/PDF in output/ and caption files.
Only the ordered principal-scale chamber sigma_1 > sigma_2 > 0 is displayed.
These are distances, not squared distances. The exact geodesics stay in this
fixed-mean, fixed-basis family, so these also equal the full Gaussian distances.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

REFERENCE = np.array([1.65, .55])
MU = np.array([.5, -.5])
ALPHA_DEGREES = 30.


def distances(sigma1, sigma2):
    fr = np.sqrt(2)*np.hypot(np.log(sigma1/REFERENCE[0]), np.log(sigma2/REFERENCE[1]))
    wo = np.hypot(sigma1-REFERENCE[0], sigma2-REFERENCE[1])
    return fr, wo


def power(matrix, exponent):
    values, vectors = np.linalg.eigh(matrix)
    return (vectors*values**exponent) @ vectors.T


def verify():
    assert np.allclose(distances(*REFERENCE), [0, 0])
    angle = np.deg2rad(ALPHA_DEGREES)
    R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    S0 = R @ np.diag(REFERENCE**2) @ R.T
    root = power(S0, .5)
    invroot = power(S0, -.5)
    # Compare the plotted scalar formulas with the full covariance formulas.
    for scales in [[2.4, .9], [.6, .2], [3., 2.5], [1.7, .15]]:
        S1 = R @ np.diag(np.array(scales)**2) @ R.T
        fr_matrix = np.sqrt(.5*np.sum(np.log(np.linalg.eigvalsh(invroot @ S1 @ invroot))**2))
        wo_matrix = np.sqrt(np.trace(S0+S1-2*power(root @ S1 @ root, .5)))
        assert np.allclose(distances(*scales), [fr_matrix, wo_matrix])
        t = np.linspace(0, 1, 101)[:, None]
        fr_path = REFERENCE**(1-t)*np.array(scales)**t
        wo_path = (1-t)*REFERENCE+t*np.array(scales)
        assert np.all(fr_path[:, 0] > fr_path[:, 1])
        assert np.all(wo_path[:, 0] > wo_path[:, 1])
        assert np.allclose(distances(fr_path[:, 0], fr_path[:, 1])[0], t[:, 0]*fr_matrix)
        assert np.allclose(distances(wo_path[:, 0], wo_path[:, 1])[1], t[:, 0]*wo_matrix)


def main():
    verify()
    plt.rcParams.update({'font.family': 'serif', 'mathtext.fontset': 'cm', 'font.size': 11,
                         'axes.titlesize': 16, 'axes.labelsize': 15, 'savefig.facecolor': 'white'})
    scales = np.linspace(.06, 3.2, 850)
    S1, S2 = np.meshgrid(scales, scales)
    fr, wo = distances(S1, S2)
    outside = S1 <= S2
    fields = [np.ma.masked_where(outside, field) for field in [fr, wo]]
    vmax = np.ceil(max(float(field.max()) for field in fields))
    levels = [.4, .8, 1.2, 1.6, 2., 2.5, 3., 4., 5.]
    norm = Normalize(0, vmax)
    fig, axes = plt.subplots(1, 2, figsize=(14, 8.4))
    fig.subplots_adjust(left=.065, right=.88, bottom=.29, top=.77, wspace=.20)
    fig.suptitle('Gaussian distance maps in principal-scale coordinates', y=.96, fontsize=21)
    fig.text(.47, .899, r'$\mu=(0.5,-0.5)^T,\quad\alpha=30^\circ\quad\mathrm{(fixed)},'
             r'\qquad\Sigma=R(\alpha)\,\mathrm{diag}(\sigma_1^2,\sigma_2^2)\,R(\alpha)^T$',
             ha='center', fontsize=16)
    fig.text(.47, .846, r'Reference: $(\sigma_{1,\star},\sigma_{2,\star})=(1.65,0.55)$',
             ha='center', fontsize=14)
    for ax, field, title in zip(axes, fields, ['Fisher–Rao', 'Wasserstein–Otto']):
        ax.set_facecolor('#f6f6f6')
        mesh = ax.pcolormesh(S1, S2, field, cmap='coolwarm', norm=norm, shading='auto', rasterized=True)
        contours = ax.contour(S1, S2, field, levels=[v for v in levels if v < field.max()],
                              colors='#24374b', linewidths=.8, alpha=.87, corner_mask=False)
        ax.clabel(contours, fmt='%g', fontsize=9, inline=True)
        ax.plot([.06, 3.2], [.06, 3.2], color='.45', ls='--', lw=.9)
        ax.text(.58, 2.5, r'$\sigma_1\leq\sigma_2$', fontsize=15, color='.4')
        ax.text(.36, 2.12, 'Outside the chosen\nordered-scale region.', fontsize=10, color='.45')
        ax.scatter(*REFERENCE, marker='*', s=190, color='white', edgecolor='#19324e', linewidth=1.1, zorder=6)
        ax.annotate(r'$p_\star$', REFERENCE, xytext=(10, 15), textcoords='offset points', fontsize=14,
                    bbox=dict(fc='white', ec='none', alpha=.8, pad=1))
        ax.set(xlim=(.06, 3.2), ylim=(.06, 3.2), xlabel=r'$\sigma_1$', ylabel=r'$\sigma_2$')
        ax.set_aspect('equal')
        ax.set_title(title, pad=13)
        ax.set_xticks([.5, 1, 1.5, 2, 2.5, 3]); ax.set_yticks([.5, 1, 1.5, 2, 2.5, 3])
        ax.spines[['top', 'right']].set_visible(False)
    cax = fig.add_axes([.91, .295, .017, .405])
    bar = fig.colorbar(mesh, cax=cax)
    bar.set_label(r'Distance from $p_\star$ (shared scale)', labelpad=10)
    fig.text(.265, .17,
             r'$d_{\mathrm{FR}}=\sqrt{2\left[\log^2\!\left(\frac{\sigma_1}{1.65}\right)'
             r'+\log^2\!\left(\frac{\sigma_2}{0.55}\right)\right]}$', ha='center', fontsize=16)
    fig.text(.705, .17,
             r'$d_{\mathrm{WO}}=\sqrt{(\sigma_1-1.65)^2+(\sigma_2-0.55)^2}$', ha='center', fontsize=16)
    fig.text(.5, .093, 'Fisher–Rao measures relative scale changes; Wasserstein–Otto measures absolute scale changes.',
             ha='center', fontsize=12)
    fig.text(.5, .045, 'Identical axes, contour levels, and color scale. The diagonal marks equal principal scales and is excluded.',
             ha='center', fontsize=11, color='.35')
    folder = Path(__file__).resolve().parent
    out = folder/'output'
    out.mkdir(exist_ok=True)
    for extension in ['png', 'pdf']:
        fig.savefig(out/f'gaussian_scale_distance_maps.{extension}', dpi=220)
    plt.close(fig)
    caption = (
        'Distance maps for bivariate Gaussian densities with fixed mean μ=(0.5,−0.5)ᵀ and fixed '
        'principal-axis orientation α=30°. Covariances are parameterized by '
        'Σ=R(α)diag(σ₁²,σ₂²)R(α)ᵀ, where R(α) is the counterclockwise planar rotation. '
        'The white star marks the reference Gaussian with principal standard deviations '
        '(σ₁,σ₂)=(1.65,0.55). Left: Fisher–Rao distance '
        'd_FR=√{2[log²(σ₁/1.65)+log²(σ₂/0.55)]}. Right: Wasserstein–Otto distance '
        'd_WO=√{(σ₁−1.65)²+(σ₂−0.55)²}. Colors and labeled contours represent distances, '
        'not squared distances, using identical axes, contour levels and color normalization. '
        'Only the ordered region σ₁>σ₂>0 is shown, avoiding the repeated-eigenvalue boundary σ₁=σ₂ '
        'where the spectral orientation becomes nonidentifiable; the Gaussian metrics themselves '
        'remain regular at isotropic positive covariances. Because all displayed covariances share '
        'the same principal basis and mean, the exact Gaussian geodesics stay in this slice: '
        'Fisher–Rao interpolates each principal standard deviation geometrically, while '
        'Wasserstein–Otto interpolates it arithmetically. Both preserve the strict ordering. '
        'Thus the displayed distances also equal the distances in the full Gaussian family. '
        'WO contours are circles clipped to the displayed region; FR contours become circles '
        'in logarithmic scale coordinates. Approaching a zero scale has infinite FR distance '
        'but finite WO distance. The fixed values of μ and α do not affect these distances.'
    )
    for name in ['gaussian_scale_distance_maps_caption.txt', 'caption.txt']:
        (folder/name).write_text(caption+'\n', encoding='utf-8')
    print(f'Saved figure and captions; common distance scale 0–{vmax:g}.')
    print('Verified scalar distances against full covariance formulas and constant-speed geodesic paths.')


if __name__ == '__main__':
    main()
