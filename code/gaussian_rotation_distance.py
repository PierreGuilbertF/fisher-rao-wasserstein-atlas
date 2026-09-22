"""Endpoint FR and WO distances as a 2D Gaussian covariance rotates.

Run: python visualization_scripts/gaussian_rotation_distance.py
Requires numpy and matplotlib. Saves PNG/PDF in output/, and both a named
caption and caption.txt. Mean and principal scales are fixed at the endpoints;
the geodesics joining them are allowed to vary principal scales.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SCALES = np.array([1.65, .55])
MEAN = np.array([.5, -.5])
REFERENCE_ANGLE = 30.


def covariance(angle_degrees):
    angle = np.deg2rad(angle_degrees)
    R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    return R @ np.diag(SCALES**2) @ R.T


def power(matrix, exponent):
    values, vectors = np.linalg.eigh(matrix)
    return (vectors*values**exponent) @ vectors.T


def distances(angle_degrees):
    # Same spectrum at both endpoints. These stable formulas are equivalent to
    # the general affine-invariant and Bures covariance distance formulas.
    variance1, variance2 = SCALES**2
    sine = np.abs(np.sin(np.deg2rad(np.asarray(angle_degrees)-REFERENCE_ANGLE)))
    fr = 2*np.arcsinh((variance1-variance2)*sine/(2*np.sqrt(variance1*variance2)))
    trace_root = np.sqrt((variance1+variance2)**2-(variance1-variance2)**2*sine**2)
    wo = np.sqrt(2)*(variance1-variance2)*sine/np.sqrt(variance1+variance2+trace_root)
    return fr, wo


def verify():
    reference = covariance(REFERENCE_ANGLE)
    root = power(reference, .5)
    inverse_root = power(reference, -.5)
    for angle in [-60, -37, -15, 12, 30, 48, 75, 99, 120]:
        target = covariance(angle)
        matrix_fr = np.sqrt(.5*np.sum(np.log(np.linalg.eigvalsh(inverse_root @ target @ inverse_root))**2))
        matrix_wo = np.sqrt(max(0., np.trace(reference+target-2*power(root @ target @ root, .5))))
        assert np.allclose(distances(angle), [matrix_fr, matrix_wo], atol=5e-8)
    angles = np.linspace(-60, 120, 251)
    assert np.allclose(distances(angles), distances(2*REFERENCE_ANGLE-angles))
    assert np.allclose(distances(angles), distances(angles+180))
    assert np.allclose(distances(REFERENCE_ANGLE), [0, 0])
    assert np.allclose(distances(REFERENCE_ANGLE+90),
                       [2*np.log(SCALES[0]/SCALES[1]), np.sqrt(2)*(SCALES[0]-SCALES[1])])
    # Local distance slopes recover the rotational metric coefficients.
    variance1, variance2 = SCALES**2
    delta = 1e-6
    slopes = np.array(distances(REFERENCE_ANGLE+np.rad2deg(delta)))/delta
    assert np.allclose(slopes, [(variance1-variance2)/np.sqrt(variance1*variance2),
                               (variance1-variance2)/np.sqrt(variance1+variance2)], rtol=1e-6)


def main():
    verify()
    plt.rcParams.update({'font.family': 'serif', 'mathtext.fontset': 'cm', 'font.size': 12,
                         'axes.titlesize': 16, 'axes.labelsize': 14, 'savefig.facecolor': 'white'})
    angles = np.linspace(-60, 120, 1001)
    sample_angles = np.array([-60, -15, 30, 75, 120])
    fields = distances(angles)
    samples = distances(sample_angles)
    blue, gold = '#326a9b', '#bf8b19'
    fig = plt.figure(figsize=(14, 8.8))
    fig.suptitle('Gaussian distance as the covariance rotates', y=.965, fontsize=22)
    fig.text(.5, .909, r'$\mu=(0.5,-0.5)^T,\quad(\sigma_1,\sigma_2)=(1.65,0.55)\quad\mathrm{(fixed)},'
             r'\qquad\alpha_\star=30^\circ$', ha='center', fontsize=16)
    fig.text(.5, .865, r'$\Sigma(\alpha)=R(\alpha)\,\mathrm{diag}(\sigma_1^2,\sigma_2^2)\,R(\alpha)^T$',
             ha='center', fontsize=16)
    theta = np.linspace(0, 2*np.pi, 300)
    unit = np.array([np.cos(theta), np.sin(theta)])
    ellipse_reference = power(covariance(REFERENCE_ANGLE), .5) @ unit
    for col, (name, curve, sample_values) in enumerate(zip(['Fisher–Rao', 'Wasserstein–Otto'], fields, samples)):
        left = .075+col*.49
        width = .415
        ax = fig.add_axes([left, .425, width, .35])
        ax.plot(angles, curve, color=blue, lw=2.7)
        ax.scatter(sample_angles, sample_values, s=38, color=blue, edgecolor='white', zorder=4)
        ax.axvline(REFERENCE_ANGLE, ls=':', color=gold, lw=1.3)
        ax.scatter(REFERENCE_ANGLE, 0, marker='*', s=165, color=gold, edgecolor='white', zorder=6)
        ax.annotate(r'$p_\star$', (REFERENCE_ANGLE, 0), xytext=(9, 12), textcoords='offset points',
                    color='#9a6b06', fontsize=15)
        ax.set(xlim=(-65, 125), ylim=(-.10, 2.45), xlabel=r'Orientation $\alpha$ (degrees)', ylabel='Distance')
        ax.set_xticks(sample_angles)
        ax.set_yticks([0, .5, 1, 1.5, 2])
        ax.set_title(name, pad=13)
        ax.grid(alpha=.16, lw=.6)
        ax.spines[['top', 'right']].set_visible(False)
        ax.text(.5, .92, r'$d\!\left(p_{\mu,\Sigma(\alpha_\star)},p_{\mu,\Sigma(\alpha)}\right)$',
                transform=ax.transAxes, ha='center', fontsize=16)
        # Same spatial aspect and scale for all thumbnails, centered at the
        # common mean. A dashed outline repeats the reference orientation.
        thumb_width = width/5*.88
        for index, angle in enumerate(sample_angles):
            thumb = fig.add_axes([left+(index+.06)*width/5, .225, thumb_width, .115])
            current = power(covariance(angle), .5) @ unit
            color = gold if angle == REFERENCE_ANGLE else blue
            thumb.plot(*ellipse_reference, color='.6', ls='--', lw=.8)
            thumb.fill(*current, color=color, alpha=.13)
            thumb.plot(*current, color=color, lw=1.6)
            thumb.scatter(0, 0, s=8, color='.25')
            thumb.set(xlim=(-1.85, 1.85), ylim=(-1.85, 1.85), aspect='equal')
            thumb.axis('off')
            thumb.set_title(rf'${angle:g}^\circ$', fontsize=11, pad=2)
        fig.text(left+width/2, .183, 'Density-level ellipses; dashed outline: reference',
                 ha='center', fontsize=10, color='.4')
    fig.text(.5, .115, 'Distances between endpoint Gaussians: the connecting geodesics may change their principal scales.',
             ha='center', fontsize=12)
    fig.text(.5, .071, r'The distance is symmetric about $\alpha_\star$ and $180^\circ$-periodic; '
             'the two endpoint orientations describe the same covariance.', ha='center', fontsize=11)
    fig.text(.5, .029, 'Both graphs use identical axes. Ellipses are centered at the common mean and drawn at the same spatial scale.',
             ha='center', fontsize=10, color='.35')
    folder = Path(__file__).resolve().parent
    out = folder/'output'
    out.mkdir(exist_ok=True)
    for extension in ['png', 'pdf']:
        fig.savefig(out/f'gaussian_rotation_distance.{extension}', dpi=220)
    plt.close(fig)
    caption = (
        'Fisher–Rao (left) and Wasserstein–Otto (right) distances from a reference bivariate Gaussian '
        'as its covariance orientation varies. The mean μ=(0.5,−0.5)ᵀ and principal standard deviations '
        '(σ₁,σ₂)=(1.65,0.55) are fixed, and Σ(α)=R(α)diag(σ₁²,σ₂²)R(α)ᵀ, with R(α) the '
        'counterclockwise planar rotation. The reference angle is α⋆=30°. Both panels use the same '
        'angle range [−60°,120°] and distance scale. Curves show the full Gaussian endpoint distances, '
        'not path lengths constrained to rotation at fixed scales: the connecting geodesics may change '
        'the principal scales while keeping the common mean fixed. '
        'Writing δ=α−α⋆, the exact distances are '
        'd_FR=2 asinh(|σ₁²−σ₂²| |sin δ|/(2σ₁σ₂)) and '
        'd_WO=[2(σ₁²+σ₂²)−2√(4σ₁²σ₂²+(σ₁²−σ₂²)² cos²δ)]^(1/2). '
        'They vanish at the reference angle, are symmetric about it, and are periodic with period 180°, '
        'because a half-turn leaves the covariance unchanged. The maximum on the displayed interval '
        'occurs at an angular difference of 90°: d_FR=2 log(σ₁/σ₂)≈2.197 and '
        'd_WO=√2(σ₁−σ₂)≈1.556. The thumbnails depict the density levels '
        '(x−μ)ᵀΣ(α)⁻¹(x−μ)=1 at five angles, with a dashed reference outline and a common spatial '
        'scale; these outlines do not assert a prescribed enclosed probability. '
        'The highlighted reference ellipse corresponds to the gold star. The first and last '
        'thumbnails coincide, illustrating the angular periodicity. '
        'Distances were checked against the general covariance matrix formulas and their local '
        'slopes against the rotational coefficients of both pullback metrics.'
    )
    for name in ['gaussian_rotation_distance_caption.txt', 'caption.txt']:
        (folder/name).write_text(caption+'\n', encoding='utf-8')
    print('Saved figure and captions. Verified matrix formulas, symmetry, periodicity, maxima, and local metric slopes.')


if __name__ == '__main__':
    main()
