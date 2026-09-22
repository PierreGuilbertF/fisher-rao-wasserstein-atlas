"""Intrinsic Gaussian Fisher--Rao and Wasserstein--Otto geodesics.

Endpoints: N(-2,1) and N(2,1), with variance 1 at both ends.
Run: python visualization_scripts/gaussian_geodesic_interpolation.py
Requires numpy and matplotlib. Saves PNG/PDF beside this script in output/.
Six densities are evaluated at equal constant-speed geodesic times.
"""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np


def fisher_path(t):
    # In (mu/sqrt(2), sigma), the hyperbolic geodesic is a radius-sqrt(3)
    # semicircle. Hyperbolic arclength is affine in the argument below.
    argument = (2*np.asarray(t)-1)*np.arctanh(np.sqrt(2/3))
    return np.sqrt(6)*np.tanh(argument), np.sqrt(3)/np.cosh(argument)


def wasserstein_path(t):
    t = np.asarray(t)
    return -2+4*t, np.ones_like(t)


def main():
    plt.rcParams.update({
        'font.family': 'serif', 'mathtext.fontset': 'cm', 'font.size': 11,
        'axes.titlesize': 14, 'axes.labelsize': 13, 'savefig.facecolor': 'white',
    })
    times = np.linspace(0, 1, 6)
    smooth_times = np.linspace(0, 1, 600)
    x = np.linspace(-7.5, 7.5, 2000)
    cmap = plt.get_cmap('coolwarm')
    colors = cmap(times)
    fr_mu, fr_sigma = fisher_path(times)
    wo_mu, wo_sigma = wasserstein_path(times)
    assert np.allclose(fr_mu[[0, -1]], [-2, 2])
    assert np.allclose(fr_sigma[[0, -1]], 1)
    assert np.allclose(fr_mu**2/2+fr_sigma**2, 3)
    # Check constant Fisher speed from analytic derivatives, not linear angle.
    half_span = np.arctanh(np.sqrt(2/3))
    argument = (2*smooth_times-1)*half_span
    mu_dot = 2*half_span*np.sqrt(6)/np.cosh(argument)**2
    sigma_dot = -2*half_span*np.sqrt(3)*np.tanh(argument)/np.cosh(argument)
    _, smooth_sigma = fisher_path(smooth_times)
    speed_squared = (mu_dot**2+2*sigma_dot**2)/smooth_sigma**2
    assert np.allclose(speed_squared, 8*half_span**2)
    assert np.allclose(wo_mu[[0,-1]], [-2,2]) and np.allclose(wo_sigma, 1)

    fig, axes = plt.subplots(2, 2, figsize=(13.8, 9.4))
    fig.subplots_adjust(left=.07, right=.895, bottom=.14, top=.84,
                        wspace=.20, hspace=.38)
    fig.suptitle('Gaussian geodesics: moving from '+r'$\mathcal{N}(-2,1)$'+
                 ' to '+r'$\mathcal{N}(2,1)$', y=.975, fontsize=19)
    fig.text(.485, .921,
             r'Six distributions at $t=0,\ 0.2,\ 0.4,\ 0.6,\ 0.8,\ 1$'
             ' along constant-speed geodesics', ha='center', fontsize=12)
    for col, (path, title, length) in enumerate([
        (fisher_path, 'Fisher–Rao within the Gaussian family', 2*np.sqrt(2)*half_span),
        (wasserstein_path, 'Wasserstein–Otto', 4.0),
    ]):
        top, bottom = axes[:, col]
        mu_curve, sigma_curve = path(smooth_times)
        mu_points, sigma_points = path(times)
        top.plot(mu_curve, sigma_curve, color='.35', lw=1.5, zorder=1)
        for i in range(len(smooth_times)-1):
            top.plot(mu_curve[i:i+2], sigma_curve[i:i+2],
                     color=cmap((smooth_times[i]+smooth_times[i+1])/2), lw=2.6)
        top.scatter(mu_points, sigma_points, c=colors, edgecolor='white',
                    linewidth=.9, s=66, zorder=4)
        top.set(xlim=(-2.6, 2.6), ylim=(.76, 1.95),
                xlabel=r'Mean $\mu$', ylabel=r'Standard deviation $\sigma$')
        top.set_title(title, pad=14)
        top.set_yticks([.8, 1, 1.2, 1.4, 1.6, 1.8])
        top.text(.04, .91, rf'$d={length:.4f}$', transform=top.transAxes, fontsize=12)
        for mu_end, label in [(-2, r'$p_0$'), (2, r'$p_1$')]:
            top.annotate(label, (mu_end, 1), xytext=(0,-20),
                         textcoords='offset points', ha='center', fontsize=12)
        for mu_value, sigma_value, color in zip(mu_points, sigma_points, colors):
            density = np.exp(-.5*((x-mu_value)/sigma_value)**2)/(sigma_value*np.sqrt(2*np.pi))
            bottom.plot(x, density, color=color, lw=2.2)
            bottom.fill_between(x, 0, density, color=color, alpha=.035)
        bottom.set(xlim=(-7.5, 7.5), ylim=(0, .435), xlabel=r'$x$',
                   ylabel=r'$p_{\mu(t),\sigma(t)}(x)$')
        bottom.set_title('Densities along the geodesic', fontsize=13, pad=10)
        for ax in (top, bottom):
            ax.spines[['top', 'right']].set_visible(False)
            ax.grid(alpha=.13, lw=.6)
            ax.tick_params(labelsize=10)
    axes[0, 0].text(.5, .06, r'$\mu(t)^2/2+\sigma(t)^2=3$',
                    transform=axes[0,0].transAxes, ha='center', fontsize=12)
    axes[0, 1].text(.5, .06, r'$\mu(t)=-2+4t,\quad\sigma(t)=1$',
                    transform=axes[0,1].transAxes, ha='center', fontsize=12)
    color_ax = fig.add_axes([.925, .22, .016, .53])
    colorbar = fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(0,1), cmap=cmap),
                           cax=color_ax, ticks=times)
    colorbar.set_label(r'Geodesic time $t$', labelpad=12)
    fig.text(.485, .077,
             'Fisher–Rao broadens the Gaussian during the transition; '
             'Wasserstein–Otto translates it at fixed variance.',
             ha='center', fontsize=12)
    fig.text(.485, .033,
             'Colors match the six parameter points and their densities. '
             'Both columns use identical axis limits.',
             ha='center', fontsize=10, color='.35')
    out = Path(__file__).resolve().parent/'output'
    out.mkdir(exist_ok=True)
    fig.savefig(out / "gaussian_geodesic_interpolation.pdf")
    fig.savefig(out / "gaussian_geodesic_interpolation.png", dpi=220)
    plt.close(fig)
    print('Checked endpoints, Fisher geodesic ellipse, and constant speed.')
    print('Six times:', times)
    print('Fisher standard deviations:', fr_sigma)


if __name__ == '__main__':
    main()
