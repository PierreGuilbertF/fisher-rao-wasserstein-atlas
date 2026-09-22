"""Illustrate a 3D Gaussian covariance and a mean tangent in its eigenbasis.

Run: python visualization_scripts/gaussian_spectral_decomposition.py
Requires numpy and matplotlib. Saves PNG/PDF in output/ and both caption files.
The displayed ellipsoid is a constant-density surface, not a probability region
of prescribed mass. Tangent arrows use an arbitrary common display scale.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


CAPTION = r"""Spectral decomposition of a three-dimensional Gaussian covariance. The translucent surface is the constant-density ellipsoid (x−μ)ᵀΣ⁻¹(x−μ)=1, on which p_{μ,Σ}(x)=e^(−1/2)p_{μ,Σ}(μ). Its orthonormal principal directions are the columns of R=(e₁,e₂,e₃), and its semi-axis lengths are σ₁, σ₂, σ₃. Thus Σ=RDRᵀ with D=diag(σ₁²,σ₂²,σ₃²). The gold arrow represents a mean tangent dμ, drawn at μ with an arbitrary display scale. The dashed colored segments decompose the same arrow into (eᵢᵀdμ)eᵢ; the components form Rᵀdμ in the principal basis. This changes the coordinates of dμ without changing the vector itself. Here σ=(2.2,1.25,0.7) and Rᵀdμ=(2.7,0.6,1.6)ᵀ. R is evaluated at the displayed covariance: Rᵀdμ denotes the components of the tangent, rather than the differential of a moving-frame position. The ellipsoid is a density level set, not a region with a specified enclosed probability."""


def main():
    plt.rcParams.update({'font.family': 'serif', 'mathtext.fontset': 'cm',
                         'font.size': 12, 'savefig.facecolor': 'white'})
    a, b, c = np.deg2rad([30., -20., 15.])
    rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    ry = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
    rx = np.array([[1, 0, 0], [0, np.cos(c), -np.sin(c)], [0, np.sin(c), np.cos(c)]])
    R = rz @ ry @ rx
    sigma = np.array([2.2, 1.25, .7])
    D = np.diag(sigma**2)
    Sigma = R @ D @ R.T
    mu = np.zeros(3)
    components = np.array([2.7, .6, 1.6])
    dmu = R @ components
    assert np.allclose(R.T @ R, np.eye(3)) and np.isclose(np.linalg.det(R), 1)
    assert np.allclose(R.T @ dmu, components)
    assert np.allclose(Sigma @ R, R @ D)

    longitude, latitude = np.meshgrid(np.linspace(0, 2*np.pi, 100), np.linspace(0, np.pi, 60))
    unit = np.array([np.cos(longitude)*np.sin(latitude),
                     np.sin(longitude)*np.sin(latitude), np.cos(latitude)])
    surface = np.einsum('ij,jkl->ikl', R @ np.diag(sigma), unit)
    assert np.allclose(np.einsum('ikl,ij,jkl->kl', surface, np.linalg.inv(Sigma), surface), 1)

    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_axes([.015, .09, .66, .79], projection='3d', computed_zorder=False)
    ax.view_init(elev=25, azim=-30)
    ax.set_proj_type('ortho')
    ax.plot_surface(*surface, color='#8eafcb', alpha=.18, linewidth=0,
                    shade=False, zorder=1)
    # Three great ellipses clarify the surface without a dense mesh.
    angle = np.linspace(0, 2*np.pi, 300)
    for i, j in [(0, 1), (0, 2), (1, 2)]:
        curve = sigma[i]*R[:, i, None]*np.cos(angle) + sigma[j]*R[:, j, None]*np.sin(angle)
        ax.plot(*curve, color='#6589a6', alpha=.45, lw=.9, zorder=2)
    colors = ['#326a9b', '#b74b55', '#278273']
    for i, color in enumerate(colors):
        direction = R[:, i]
        ax.plot(*np.column_stack([-sigma[i]*direction, np.zeros(3)]),
                color=color, lw=1.3, alpha=.55, zorder=3)
        endpoint = sigma[i]*direction
        ax.quiver(*mu, *endpoint, color=color, linewidth=2.5,
                  arrow_length_ratio=.10, zorder=5)
        label = endpoint + .20*direction
        ax.text(*label, rf'$e_{i+1}$', color=color, fontsize=17, zorder=8)
        midpoint = .57*endpoint + np.array([0, 0, -.19 if i < 2 else .05])
        ax.text(*midpoint, rf'$\sigma_{i+1}$', color=color, fontsize=16,
                bbox=dict(fc='white', ec='none', alpha=.7, pad=.5), zorder=8)
    ax.scatter(*mu, s=22, color='#263746', depthshade=False, zorder=8)
    ax.text(*np.array([-.15, -.20, -.15]), r'$\mu$', fontsize=17, zorder=9)
    ax.quiver(*mu, *dmu, color='#bb830b', linewidth=3.0,
              arrow_length_ratio=.075, zorder=7)
    ax.text(*(dmu+np.array([.12, .02, .13])), r'$d\mu$', color='#a97509', fontsize=19, zorder=9)
    start = mu.copy()
    for i, color in enumerate(colors):
        end = start + components[i]*R[:, i]
        ax.plot(*np.column_stack([start, end]), ls='--', lw=2.1, color=color, zorder=6)
        ax.scatter(*end, s=16, color=color, depthshade=False, zorder=7)
        start = end
    assert np.allclose(start, dmu)
    ax.set(xlim=(-2.65, 2.65), ylim=(-2.65, 2.65), zlim=(-2.1, 3.2),
           xlabel=r'$x_1$', ylabel=r'$x_2$', zlabel=r'$x_3$')
    ax.set_box_aspect((1, 1, 1))
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.pane.fill = False
        axis._axinfo['grid']['color'] = (0.75, .78, .8, .25)
    ax.set_xticks([-2, 0, 2]); ax.set_yticks([-2, 0, 2]); ax.set_zticks([-2, 0, 2])
    ax.tick_params(labelsize=10)
    fig.suptitle('A Gaussian covariance: principal directions and scales', y=.96, fontsize=20)
    fig.text(.69, .81, 'Covariance in its principal basis', fontsize=14, weight='bold')
    fig.text(.69, .745, r'$R=(e_1,e_2,e_3),\qquad R^TR=I$', fontsize=18)
    fig.text(.69, .682, r'$D=\mathrm{diag}(\sigma_1^2,\sigma_2^2,\sigma_3^2)$', fontsize=18)
    fig.text(.69, .608, r'$\Sigma=RDR^T$', fontsize=23, bbox=dict(fc='#edf3f8', ec='none', pad=8))
    fig.text(.69, .49, 'The same mean tangent, in this basis', fontsize=14, weight='bold')
    fig.text(.69, .426, r'$R^T d\mu=(e_1^T d\mu,\ e_2^T d\mu,\ e_3^T d\mu)^T$', fontsize=16)
    fig.text(.69, .36, r'$d\mu=\sum_{i=1}^{3}(e_i^T d\mu)e_i$', fontsize=19)
    for i, color in enumerate(colors):
        fig.text(.70, .285-.042*i, rf'Dashed segment {i+1}:  $(e_{i+1}^T d\mu)e_{i+1}$',
                 color=color, fontsize=12)
    fig.text(.69, .11, 'Components in the current eigenvector basis.\nThe vector itself is unchanged.', fontsize=12, color='.35')
    fig.text(.34, .045, r'Constant-density surface: $(x-\mu)^T\Sigma^{-1}(x-\mu)=1$',
             ha='center', fontsize=14, color='#41576a')
    folder = Path(__file__).resolve().parent
    out = folder/'output'
    out.mkdir(exist_ok=True)
    for extension in ['png', 'pdf']:
        fig.savefig(out/f'gaussian_spectral_decomposition.{extension}', dpi=220)
    plt.close(fig)
    for filename in ['gaussian_spectral_decomposition_caption.txt', 'caption.txt']:
        (folder/filename).write_text(CAPTION+'\n', encoding='utf-8')
    print('Saved PNG, PDF, and captions. Verified orthonormality, eigenpairs, density level, and tangent components.')


if __name__ == '__main__':
    main()
