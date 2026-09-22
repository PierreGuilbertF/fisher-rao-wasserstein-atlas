"""Full 2D Gaussian Fisher--Rao and Wasserstein--Otto geodesics.

Run: python visualization_scripts/gaussian_2d_geodesic_comparison.py
Requires numpy, scipy and matplotlib. Saves PNG, PDF and numerical diagnostics
in output/, plus a named caption and caption.txt next to this script.

FR: solve the coupled mean/covariance geodesic boundary-value problem in matrix
coordinates, not by independently interpolating covariance and mean.
WO: exact Gaussian displacement interpolation. Spectral coordinates are used
only for display, with sigma_1 > sigma_2 and a continuous angle modulo pi.

Reference for the FR equations: Lawson, Burrage, Mengersen & dos Santos (2023),
The Fisher Geometry and Geodesics of the Multivariate Normals, without
Differential Geometry, Section 2.2, https://arxiv.org/abs/2306.01278.
WO: Malago, Montrucchio & Pistone (2018), Wasserstein Riemannian geometry
of Gaussian densities, Sections 3--4, doi:10.1007/s41884-018-0014-4.
"""
from pathlib import Path
import json
import numpy as np
from scipy.integrate import solve_bvp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from mpl_toolkits.mplot3d.art3d import Line3DCollection


def pack(matrix):
    return matrix[..., [0, 0, 1], [0, 1, 1]]


def unpack(entries):
    matrix = np.empty(entries.shape[:-1]+(2, 2))
    matrix[..., 0, 0] = entries[..., 0]
    matrix[..., 0, 1] = matrix[..., 1, 0] = entries[..., 1]
    matrix[..., 1, 1] = entries[..., 2]
    return matrix


def covariance(scales, angle_degrees):
    angle = np.deg2rad(angle_degrees)
    R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    return R @ np.diag(np.asarray(scales)**2) @ R.T


def matrix_power(matrix, exponent):
    values, vectors = np.linalg.eigh(matrix)
    return (vectors*values**exponent) @ vectors.T


def fisher_rhs(t, state):
    Sigma = unpack(state[2:5].T)
    Sigma_dot = unpack(state[7:10].T)
    mu_dot = state[5:7].T
    inverse = np.linalg.inv(Sigma)
    mu_ddot = np.einsum('nij,njk,nk->ni', Sigma_dot, inverse, mu_dot)
    Sigma_ddot = Sigma_dot @ inverse @ Sigma_dot - mu_dot[:, :, None]*mu_dot[:, None, :]
    return np.vstack([state[5:], mu_ddot.T, pack(Sigma_ddot).T])


def solve_fisher(mu0, Sigma0, mu1, Sigma1, perturbation=0.):
    q0, q1 = np.r_[mu0, pack(Sigma0)], np.r_[mu1, pack(Sigma1)]
    mesh = np.linspace(0, 1, 81)
    q = q0[:, None] + (q1-q0)[:, None]*mesh
    velocity = np.repeat((q1-q0)[:, None], mesh.size, axis=1)
    # A second initial guess provides a check against initial-guess sensitivity.
    direction = np.array([.1, -.15, .8, .1, .4])*perturbation
    q += direction[:, None]*np.sin(np.pi*mesh)
    velocity += direction[:, None]*np.pi*np.cos(np.pi*mesh)
    solution = solve_bvp(fisher_rhs,
                         lambda a, b: np.r_[a[:5]-q0, b[:5]-q1],
                         mesh, np.vstack([q, velocity]), tol=1e-8, max_nodes=6000)
    if not solution.success:
        raise RuntimeError(solution.message)
    return solution


def wasserstein_path(t, mu0, Sigma0, mu1, Sigma1):
    root = matrix_power(Sigma0, .5)
    invroot = matrix_power(Sigma0, -.5)
    T = invroot @ matrix_power(root @ Sigma1 @ root, .5) @ invroot
    M = np.eye(2)[None, :, :] + t[:, None, None]*(T-np.eye(2))
    S = M @ Sigma0 @ M.transpose(0, 2, 1)
    mu = mu0 + t[:, None]*(mu1-mu0)
    Sdot = (T-np.eye(2)) @ Sigma0 @ M.transpose(0, 2, 1) + M @ Sigma0 @ (T-np.eye(2))
    return mu, S, Sdot


def spectral_coordinates(Sigma):
    eigenvalues = np.linalg.eigvalsh(Sigma)
    # Analytic angle of the major eigenvector, insensitive to eigenvector signs.
    theta = .5*np.unwrap(np.arctan2(2*Sigma[:, 0, 1], Sigma[:, 0, 0]-Sigma[:, 1, 1]))
    return np.column_stack([np.sqrt(eigenvalues[:, 1]), np.sqrt(eigenvalues[:, 0]), np.rad2deg(theta)])


def fisher_speed_squared(mu_dot, Sigma, Sigma_dot):
    inv = np.linalg.inv(Sigma)
    return (np.einsum('ni,nij,nj->n', mu_dot, inv, mu_dot)
            + .5*np.trace(inv @ Sigma_dot @ inv @ Sigma_dot, axis1=1, axis2=2))


def main():
    mu0, mu1 = np.array([-1.5, -.6]), np.array([1.7, .9])
    Sigma0 = covariance([1.45, .55], -20)
    Sigma1 = covariance([1.15, .70], 35)
    time = np.linspace(0, 1, 2001)
    times = np.linspace(0, 1, 6)
    sample_indices = np.rint(times*(time.size-1)).astype(int)
    fr = solve_fisher(mu0, Sigma0, mu1, Sigma1)
    alternative = solve_fisher(mu0, Sigma0, mu1, Sigma1, perturbation=1.)
    state = fr.sol(time)
    assert np.max(np.abs(state-alternative.sol(time))) < 2e-6
    fr_mu, fr_S = state[:2].T, unpack(state[2:5].T)
    wo_mu, wo_S, wo_Sdot = wasserstein_path(time, mu0, Sigma0, mu1, Sigma1)
    fr_Sdot = unpack(state[7:10].T)
    fr_speed = fisher_speed_squared(state[5:7].T, fr_S, fr_Sdot)
    assert np.ptp(fr_speed)/fr_speed.mean() < 1e-7
    assert np.max(np.abs(fr.sol(time, 1)-fisher_rhs(time, state))) < 2e-6
    wo_values, wo_vectors = np.linalg.eigh(wo_S)
    principal_dS = wo_vectors.transpose(0, 2, 1) @ wo_Sdot @ wo_vectors
    A = wo_vectors @ (principal_dS/(wo_values[:, :, None]+wo_values[:, None, :])) @ wo_vectors.transpose(0, 2, 1)
    wo_speed = np.sum((mu1-mu0)**2)+.5*np.trace(A @ wo_Sdot, axis1=1, axis2=2)
    assert np.ptp(wo_speed)/wo_speed.mean() < 1e-10
    fr_length = np.trapezoid(np.sqrt(fr_speed), time)
    wo_length = np.trapezoid(np.sqrt(wo_speed), time)
    # A competing full Gaussian path has no smaller FR length in this check.
    wo_fr_length = np.trapezoid(np.sqrt(fisher_speed_squared(np.broadcast_to(mu1-mu0, wo_mu.shape), wo_S, wo_Sdot)), time)
    assert fr_length < wo_fr_length
    curves = [spectral_coordinates(fr_S), spectral_coordinates(wo_S)]
    diagnostics = {'mu0': mu0.tolist(), 'mu1': mu1.tolist(), 'Sigma0': Sigma0.tolist(),
                   'Sigma1': Sigma1.tolist(), 'FR_length': float(fr_length),
                   'WO_length': float(wo_length), 'WO_path_FR_length': float(wo_fr_length),
                   'FR_max_BVP_residual': float(fr.rms_residuals.max()),
                   'FR_relative_speed_variation': float(np.ptp(fr_speed)/fr_speed.mean()),
                   'initial_guess_agreement': float(np.max(np.abs(state-alternative.sol(time))))}
    for name, means, covs, curve in zip(['FR', 'WO'], [fr_mu, wo_mu], [fr_S, wo_S], curves):
        assert np.allclose(means[[0, -1]], [mu0, mu1])
        assert np.allclose(covs[[0, -1]], [Sigma0, Sigma1])
        eigenvalues = np.linalg.eigvalsh(covs)
        gap = eigenvalues[:, 1]-eigenvalues[:, 0]
        assert eigenvalues.min() > .2 and gap.min() > .25
        assert curve[:, 2].min() > -80 and curve[:, 2].max() < 80
        diagnostics[name+'_minimum_eigenvalue'] = float(eigenvalues.min())
        diagnostics[name+'_minimum_eigenvalue_gap'] = float(gap.min())
        diagnostics[name+'_angle_range_degrees'] = [float(curve[:, 2].min()), float(curve[:, 2].max())]
    # Bound variation between knots, so spectral checks do not rely only on samples.
    # For symmetric matrices, eigenvalues change by at most the spectral norm
    # of the matrix change (Weyl); the Frobenius norm is an upper bound.
    widths = np.diff(fr.sol.x)
    coeff = fr.sol.c[:, :, 2:5].transpose(0, 2, 1)
    entry_bound = (3*np.abs(coeff[0])*widths**2
                   + 2*np.abs(coeff[1])*widths + np.abs(coeff[2]))
    derivative_bound = np.sqrt(entry_bound[0]**2+2*entry_bound[1]**2+entry_bound[2]**2)
    knots = unpack(fr.sol(fr.sol.x)[2:5].T)
    for name, matrices, step, bound in [
        ('FR', knots, widths, derivative_bound),
        ('WO', wo_S, np.diff(time),
         np.maximum(np.linalg.norm(wo_Sdot[:-1], axis=(1, 2)),
                    np.linalg.norm(wo_Sdot[1:], axis=(1, 2))))]:
        vals = np.linalg.eigvalsh(matrices)
        gaps = vals[:, 1]-vals[:, 0]
        lower_gap = np.min(np.minimum(gaps[:-1], gaps[1:])-bound*step)
        lower_eigenvalue = np.min(np.minimum(vals[:-1, 0], vals[1:, 0])-bound*step/2)
        diagonal_difference = matrices[:, 0, 0]-matrices[:, 1, 1]
        lower_chart = np.min(np.minimum(diagonal_difference[:-1], diagonal_difference[1:])-bound*step)
        assert lower_gap > .25 and lower_eigenvalue > .2 and lower_chart > 0
        diagnostics[name+'_interval_eigenvalue_gap_lower_bound'] = float(lower_gap)
        diagnostics[name+'_interval_minimum_eigenvalue_lower_bound'] = float(lower_eigenvalue)
        diagnostics[name+'_interval_S11_minus_S22_lower_bound'] = float(lower_chart)

    diagnostics['FR_max_mean_departure_from_line'] = float(np.max(np.abs((mu1-mu0)[0]*(fr_mu-mu0)[:, 1]-(mu1-mu0)[1]*(fr_mu-mu0)[:, 0]))/np.linalg.norm(mu1-mu0))

    plt.rcParams.update({'font.family': 'serif', 'mathtext.fontset': 'cm', 'font.size': 11,
                         'axes.titlesize': 14, 'savefig.facecolor': 'white'})
    fig = plt.figure(figsize=(16, 13.5))
    fig.suptitle('Two Gaussian geodesics: covariance shape and density evolution', fontsize=22, y=.974)
    fig.text(.5, .934, r'$\Sigma=R(\theta)\,\mathrm{diag}(\sigma_1^2,\sigma_2^2)\,R(\theta)^T,'
             r'\qquad \sigma_1>\sigma_2>0$', ha='center', fontsize=18)
    fig.text(.5, .904, 'Covariance projections of full Gaussian geodesics; means are included in the density snapshots.',
             ha='center', fontsize=12, color='.35')
    grid = fig.add_gridspec(2, 2, left=.055, right=.915, bottom=.125, top=.875,
                           height_ratios=[1, 1.12], hspace=.20, wspace=.12)
    cmap = plt.get_cmap('coolwarm')
    all_coordinates = np.concatenate(curves)
    limits = []
    for column in range(3):
        low, high = all_coordinates[:, column].min(), all_coordinates[:, column].max()
        pad = (high-low)*.15
        limits.append((low-pad, high+pad))
    # Physical density uses a single linear scale for every snapshot.
    density_cmap = LinearSegmentedColormap.from_list('transparent_blue',
                    [(0., (.88, .95, 1., 0.)), (.2, (.55, .77, .92, .3)),
                     (.55, (.18, .46, .72, .75)), (1., (.025, .16, .36, 1.))])
    density_max = max(1/(2*np.pi*np.sqrt(np.linalg.det(s))) for s in np.concatenate([fr_S, wo_S]))
    density_norm = Normalize(0, density_max)
    x = np.linspace(-5.1, 5.1, 230)
    y = np.linspace(-3.7, 3.7, 180)
    X, Y = np.meshgrid(x, y)
    xy = np.stack([X, Y], axis=-1)
    for col, (name, means, covs, coordinates) in enumerate(zip(
            ['Fisher–Rao', 'Wasserstein–Otto'], [fr_mu, wo_mu], [fr_S, wo_S], curves)):
        ax = fig.add_subplot(grid[0, col], projection='3d')
        points = coordinates[::4]
        segments = np.stack([points[:-1], points[1:]], axis=1)
        line = Line3DCollection(segments, cmap=cmap, norm=Normalize(0, 1), linewidth=3)
        line.set_array(time[::4][:-1])
        ax.add_collection3d(line)
        samples = coordinates[sample_indices]
        ax.scatter(*samples.T, c=times, cmap=cmap, s=46, edgecolor='white', depthshade=False, zorder=6)
        for endpoint, label in [(samples[0], r'$p_0$'), (samples[-1], r'$p_1$')]:
            ax.text(*(endpoint+np.array([.012, 0, 1.])), label, fontsize=14)
        ax.set(xlim=limits[0], ylim=limits[1], zlim=limits[2],
               xlabel=r'$\sigma_1$', ylabel=r'$\sigma_2$', zlabel=r'$\theta$ (degrees)')
        ax.set_title(name + (' — numerical Gaussian geodesic' if col == 0 else ' — Gaussian geodesic'), pad=8, fontsize=15)
        ax.view_init(elev=22, azim=-57)
        ax.set_box_aspect((1.2, 1, 1))
        ax.tick_params(labelsize=9, pad=1)
        for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
            axis.pane.fill = False
            axis._axinfo['grid']['color'] = (.7, .74, .78, .3)
        snapshots = grid[1, col].subgridspec(2, 3, hspace=.20, wspace=.08)
        for index, (t, sample_index) in enumerate(zip(times, sample_indices)):
            axd = fig.add_subplot(snapshots[index//3, index%3])
            mean, S = means[sample_index], covs[sample_index]
            centered = xy-mean
            mahalanobis = np.einsum('...i,ij,...j->...', centered, np.linalg.inv(S), centered)
            density = np.exp(-mahalanobis/2)/(2*np.pi*np.sqrt(np.linalg.det(S)))
            axd.imshow(density, extent=[x[0], x[-1], y[0], y[-1]], origin='lower',
                       cmap=density_cmap, norm=density_norm, interpolation='bilinear', rasterized=True)
            # Identical absolute density levels, rather than per-panel peak normalization.
            axd.contour(X, Y, density, levels=density_max*np.array([.08, .22, .45, .70]),
                        colors=['#47789c'], linewidths=.55, alpha=.7)
            axd.plot(means[:, 0], means[:, 1], color='.45', ls=':', lw=.85, alpha=.7)
            axd.scatter(*mean, s=23, color=cmap(t), edgecolor='white', linewidth=.65, zorder=5)
            axd.set_title(rf'$t={t:.1f}$', fontsize=12, color='.2', pad=5)
            axd.set(xlim=(x[0], x[-1]), ylim=(y[0], y[-1]), aspect='equal')
            axd.set_xticks([-4, 0, 4]); axd.set_yticks([-3, 0, 3]); axd.tick_params(labelsize=8, length=2)
            if index//3 == 1:
                axd.set_xlabel(r'$x_1$', labelpad=1)
            else:
                axd.set_xticklabels([])
            if index%3 == 0:
                axd.set_ylabel(r'$x_2$', labelpad=1)
            else:
                axd.set_yticklabels([])
            axd.spines[['top', 'right']].set_visible(False)
    bar_ax = fig.add_axes([.937, .17, .009, .26])
    bar = fig.colorbar(plt.cm.ScalarMappable(norm=density_norm, cmap=density_cmap), cax=bar_ax)
    bar.set_label('Probability density', labelpad=7, fontsize=10)
    bar.ax.tick_params(labelsize=9)
    fig.text(.5, .073, 'Dots and curve colors indicate geodesic time. Dotted traces in the snapshots show the mean paths.',
             ha='center', fontsize=12)
    fig.text(.5, .048, 'Fisher–Rao couples mean and covariance; the Wasserstein–Otto mean moves on a straight segment.',
             ha='center', fontsize=12)
    fig.text(.5, .023, 'Shared axes and density scale. Both covariance paths keep distinct eigenvalues and a continuous principal-axis angle.',
             ha='center', fontsize=10, color='.35')
    folder = Path(__file__).resolve().parent
    out = folder/'output'
    out.mkdir(exist_ok=True)
    for ext in ['png', 'pdf']:
        fig.savefig(out/f'gaussian_2d_geodesic_comparison.{ext}', dpi=200)
    plt.close(fig)
    np.savez(out/'gaussian_2d_geodesic_comparison_paths.npz', t=time,
             fr_mu=fr_mu, fr_covariance=fr_S, wo_mu=wo_mu, wo_covariance=wo_S,
             fr_spectral=curves[0], wo_spectral=curves[1])
    (out/'gaussian_2d_geodesic_comparison_diagnostics.json').write_text(json.dumps(diagnostics, indent=2)+'\n')
    caption = (
        'Fisher–Rao and Wasserstein–Otto geodesics between two bivariate Gaussian densities. '
        'The endpoints have means μ₀=(−1.5,−0.6)ᵀ and μ₁=(1.7,0.9)ᵀ, principal standard deviations '
        '(σ₁,σ₂)=(1.45,0.55) and (1.15,0.70), and major-axis angles θ₀=−20° and θ₁=35°. '
        'Here Σ=R(θ)diag(σ₁²,σ₂²)R(θ)ᵀ, with σ₁>σ₂ and R(θ) the counterclockwise planar rotation. '
        'Top: the covariance projections of the full Gaussian geodesics in (σ₁,σ₂,θ) coordinates, '
        'shown with identical axis limits. These are not geodesics computed with the mean held fixed, '
        'and Euclidean lengths in this plot do not represent either metric. '
        'The Fisher–Rao path is computed numerically by solving the coupled equations '
        'μ̈=Σ̇Σ⁻¹μ̇ and Σ̈=Σ̇Σ⁻¹Σ̇−μ̇μ̇ᵀ with both endpoint conditions; '
        'the Wasserstein–Otto path uses exact Gaussian displacement interpolation. '
        'Bottom: six density snapshots per metric at equal constant-speed times t=0,0.2,0.4,0.6,0.8,1, '
        'read left to right and then top to bottom within each column. Blue shading and contours use '
        'the same absolute probability-density scale in every panel; low densities fade to transparent. '
        'Colored dots mark the current means, and dotted traces show their complete paths. '
        'The Fisher–Rao mean path curves through its coupling with covariance; the Wasserstein–Otto mean path is straight. '
        f'On a 2001-point verification grid, the minimum covariance eigenvalue gaps are '
        f'{diagnostics["FR_minimum_eigenvalue_gap"]:.3f} (Fisher–Rao) and '
        f'{diagnostics["WO_minimum_eigenvalue_gap"]:.3f} (Wasserstein–Otto). '
        'Derivative bounds between verification points also keep both paths away from repeated eigenvalues, zero eigenvalues and the angle-chart boundary. '
        'The numerical Fisher–Rao solution was checked for endpoint agreement, geodesic-equation residuals, '
        'constant speed and agreement from two initial guesses. '
        'For the Fisher–Rao equations see Lawson et al., The Fisher Geometry and Geodesics of the Multivariate Normals, '
        'without Differential Geometry (2023), Section 2.2, https://arxiv.org/abs/2306.01278; '
        'for Gaussian Wasserstein interpolation see Malagò, Montrucchio and Pistone, '
        'Wasserstein Riemannian geometry of Gaussian densities (2018), Sections 3–4, '
        'https://doi.org/10.1007/s41884-018-0014-4.'
    )
    for name in ['gaussian_2d_geodesic_comparison_caption.txt', 'caption.txt']:
        (folder/name).write_text(caption+'\n', encoding='utf-8')
    print(json.dumps(diagnostics, indent=2))
    print('Saved figure, captions, paths, and diagnostics.')


if __name__ == '__main__':
    main()
