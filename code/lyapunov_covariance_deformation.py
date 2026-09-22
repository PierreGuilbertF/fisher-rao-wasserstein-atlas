"""Geometric interpretation of the inverse Lyapunov operator for Gaussians.

Run: python visualization_scripts/lyapunov_covariance_deformation.py
Requires numpy and matplotlib. Outputs PNG/PDF and caption files.
The paper defines L_Sigma(dSigma)=A by Sigma A + A Sigma = dSigma.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

CAPTION = r"""The Lyapunov operator turns a covariance variation into a linear gradient velocity field. Left: the solid blue ellipse is the Gaussian density level (x−μ)ᵀΣ⁻¹(x−μ)=1, and the dashed red ellipse has covariance Σ+εdΣ, with the mean fixed. Dotted principal axes indicate the change of orientation and scales. Right: the unique symmetric solution A=L_Σ(dΣ) of ΣA+AΣ=dΣ gives v(x)=A(x−μ)=∇φ(x), where φ(x)=½(x−μ)ᵀA(x−μ). Gold arrows show the finite display displacement εv(x) from selected points of the original ellipse. Their endpoints lie on the green ellipse, whose covariance is (I+εA)Σ(I+εA)ᵀ. This equals Σ+εdΣ+ε²AΣA, so the transported ellipse agrees with the prescribed covariance change to first order, rather than exactly for a finite step. The right panel also repeats the red target ellipse to expose this second-order difference. In the principal basis of Σ, each entry of RᵀdΣR is divided by σᵢ²+σⱼ² to obtain RᵀAR. A symmetric gradient deformation can change an anisotropic ellipse's orientation without being a rigid rotation. Here μ=0, Σ=diag(4,1), dΣ=((1.2,2),(2,−0.3)), and ε=0.4; the ellipse levels do not specify an enclosed probability."""


def ellipse(covariance, angles):
    values, vectors = np.linalg.eigh(covariance)
    return (vectors * np.sqrt(values)) @ np.array([np.cos(angles), np.sin(angles)])


def main():
    plt.rcParams.update({'font.family': 'serif', 'mathtext.fontset': 'cm',
                         'font.size': 12, 'axes.titlesize': 16,
                         'savefig.facecolor': 'white'})
    Sigma = np.diag([4., 1.])
    dSigma = np.array([[1.2, 2.], [2., -.3]])
    epsilon = .4
    # Solve in the principal basis, without assuming dSigma commutes with Sigma.
    values, R = np.linalg.eigh(Sigma)
    principal_variation = R.T @ dSigma @ R
    A = R @ (principal_variation / (values[:, None]+values[None, :])) @ R.T
    target = Sigma + epsilon*dSigma
    transport = np.eye(2) + epsilon*A
    transported = transport @ Sigma @ transport.T
    assert np.allclose(A, A.T)
    assert np.allclose(Sigma @ A + A @ Sigma, dSigma)
    assert np.linalg.eigvalsh(target).min() > 0
    assert np.linalg.eigvalsh(transport).min() > 0
    assert np.allclose(transported-target, epsilon**2*A @ Sigma @ A)
    # The covariance error is genuinely second order in the display step.
    half_transport = np.eye(2) + epsilon/2*A
    half_error = half_transport @ Sigma @ half_transport.T - (Sigma+epsilon/2*dSigma)
    assert np.allclose(half_error, (transported-target)/4)
    angles = np.linspace(0, 2*np.pi, 600)
    original = ellipse(Sigma, angles)
    prescribed = ellipse(target, angles)
    moved = transport @ original
    assert np.allclose(np.einsum('ij,ij->j', moved, np.linalg.solve(transported, moved)), 1)
    blue, red, green, gold = '#326a9b', '#bb4f59', '#278273', '#bf8b19'
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 8.1))
    fig.subplots_adjust(left=.065, right=.975, top=.79, bottom=.30, wspace=.20)
    fig.suptitle('From covariance variation to a velocity field', fontsize=21, y=.96)
    fig.text(.5, .885, r'$d\Sigma\quad\longmapsto\quad A=\mathcal{L}_{\Sigma}(d\Sigma),'
             r'\qquad \Sigma A+A\Sigma=d\Sigma$', ha='center', fontsize=21)
    for ax in axes:
        ax.set_aspect('equal')
        ax.axhline(0, color='.87', lw=.8, zorder=0)
        ax.axvline(0, color='.87', lw=.8, zorder=0)
        ax.fill(*original, color=blue, alpha=.10)
        ax.plot(*original, color=blue, lw=2.1, zorder=3)
        ax.plot(*prescribed, color=red, ls='--', lw=2.2, zorder=5)
        ax.scatter(0, 0, s=20, color='#273747', zorder=7)
        ax.text(.07, -.22, r'$\mu$', fontsize=16)
        ax.set(xlim=(-2.7, 2.7), ylim=(-1.65, 1.65), xlabel=r'$x_1$', ylabel=r'$x_2$')
        ax.set_xticks([-2, -1, 0, 1, 2]); ax.set_yticks([-1, 0, 1])
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(labelsize=10)
    axes[0].set_title('(a) Prescribe a covariance change', loc='left', pad=18)
    for covariance, color in [(Sigma, blue), (target, red)]:
        eigvals, eigvecs = np.linalg.eigh(covariance)
        for i in range(2):
            endpoint = np.sqrt(eigvals[i])*eigvecs[:, i]
            axes[0].plot([-endpoint[0], endpoint[0]], [-endpoint[1], endpoint[1]],
                         color=color, ls=':', alpha=.7, lw=1.2)
    axes[0].legend(handles=[Line2D([], [], color=blue, lw=2, label=r'$\Sigma$'),
                            Line2D([], [], color=red, lw=2, ls='--', label=r'$\Sigma+\varepsilon d\Sigma$')],
                   loc='upper left', bbox_to_anchor=(0, -.13), frameon=False, ncol=2)
    axes[1].set_title('(b) Recover the gradient velocity', loc='left', pad=18)
    axes[1].plot(*moved, color=green, lw=1.9, zorder=4)
    boundary = ellipse(Sigma, np.linspace(0, 2*np.pi, 16, endpoint=False))
    displacement = epsilon*A @ boundary
    axes[1].quiver(*boundary, *displacement, angles='xy', scale_units='xy', scale=1,
                   color=gold, width=.0055, headwidth=3.8, headlength=4.5, zorder=8)
    axes[1].scatter(*(boundary+displacement), color=green, s=12, zorder=7)
    axes[1].legend(handles=[Line2D([], [], color=gold, lw=2, label=r'$\varepsilon A(x-\mu)$'),
                            Line2D([], [], color=green, lw=2, label='Transported ellipse')],
                   loc='upper left', bbox_to_anchor=(0, -.13), frameon=False, ncol=2)
    fig.text(.5, .17,
             r'$(I+\varepsilon A)\Sigma(I+\varepsilon A)^T'
             r'=\Sigma+\varepsilon d\Sigma+\varepsilon^2 A\Sigma A$',
             fontsize=20, ha='center')
    fig.text(.5, .12, 'The prescribed change and the transported covariance agree to first order.',
             ha='center', color='.35', fontsize=12)
    fig.text(.5, .045,
             r'In the principal basis:  $(R^TAR)_{ij}'
             r'=\dfrac{(R^Td\Sigma R)_{ij}}{\sigma_i^2+\sigma_j^2}$'
             r'    ($\varepsilon=0.4$ for display)',
             ha='center', fontsize=15,
             bbox=dict(fc='#f0f4f7', ec='none', pad=9))
    folder = Path(__file__).resolve().parent
    out = folder/'output'
    out.mkdir(exist_ok=True)
    for extension in ['png', 'pdf']:
        fig.savefig(out/f'lyapunov_covariance_deformation.{extension}', dpi=220)
    plt.close(fig)
    for name in ['lyapunov_covariance_deformation_caption.txt', 'caption.txt']:
        (folder/name).write_text(CAPTION+'\n', encoding='utf-8')
    print('Saved figure and captions. Verified Lyapunov equation, ellipse transport, and second-order error.')
    print(f'A = {A.tolist()}')


if __name__ == '__main__':
    main()
