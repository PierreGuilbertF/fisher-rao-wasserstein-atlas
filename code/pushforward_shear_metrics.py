"""Shear of a standard 2D Gaussian: velocity projection and FR/WO costs.
Run: python visualization_scripts/pushforward_shear_metrics.py
Requires numpy and matplotlib; writes PNG, PDF, captions and diagnostics.
All tangent fields and costs are evaluated at theta=0, dtheta=1.
"""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm

DTHETA = 1.0
EPSILON = .65
ARROW_SCALE = .28


def density(y1, y2, theta=0.):
    return np.exp(-((y1-theta*y2)**2+y2**2)/2)/(2*np.pi)


def fields(y1, y2):
    p = density(y1, y2)
    u = DTHETA*y1*y2*p
    fr = DTHETA**2*y1**2*y2**2*p
    wo = DTHETA**2*(y1**2+y2**2)*p/4
    raw = DTHETA**2*y2**2*p
    return p, u, fr, wo, raw


def verify():
    axis = np.linspace(-9, 9, 1001)
    y1, y2 = np.meshgrid(axis, axis)
    p, u, fr, wo, raw = fields(y1, y2)
    integral = lambda a: float(np.trapezoid(np.trapezoid(a, axis, axis=1), axis))
    values = list(map(integral, (p, u, fr, wo, raw)))
    assert np.allclose(values, [1, 0, DTHETA**2, DTHETA**2/2, DTHETA**2], atol=1e-10)
    x, y = np.array([-.8, .3, 1.2]), np.array([.4, -.7, 1.1])
    h = 1e-5
    assert np.allclose((density(x,y,h*DTHETA)-density(x,y,-h*DTHETA))/(2*h),
                       fields(x,y)[1], atol=1e-10)
    for kind in ('raw', 'gradient', 'rotation'):
        def flux(x,y):
            p = density(x,y)
            if kind == 'raw': return p*DTHETA*y, np.zeros_like(p)
            if kind == 'gradient': return p*DTHETA*y/2, p*DTHETA*x/2
            return p*DTHETA*y/2, -p*DTHETA*x/2
        div = ((flux(x+h,y)[0]-flux(x-h,y)[0]) +
               (flux(x,y+h)[1]-flux(x,y-h)[1]))/(2*h)
        expected = np.zeros_like(x) if kind == 'rotation' else -fields(x,y)[1]
        assert np.allclose(div, expected, atol=1e-10)
    return dict(zip(('mass','integral_u','FR_squared_norm','WO_squared_norm','raw_velocity_energy'), values))


CAPTION = r'''Shear of a Gaussian: the density variation and the velocity used by each metric. The reference is q = N(0,I_2), and phi_theta(x_1,x_2) = (x_1+theta x_2,x_2). All infinitesimal fields and costs are evaluated at theta = 0 with dtheta = 1, so p_0 = q. (a) Circular contours of q (blue) and their sheared images (red) for the finite display increment epsilon dtheta = 0.65, together with the sheared grid. (b) The induced velocity v = dtheta(y_2,0). (c) Its weighted gradient projection grad psi = (dtheta/2)(y_2,y_1), where psi = dtheta y_1 y_2/2. (d) The remaining rotation w = (dtheta/2)(y_2,-y_1), tangent to the Gaussian contours and satisfying div(qw) = 0. Thus v = grad psi + w, and v and grad psi generate the same density variation. All arrows are multiplied by 0.28 for display. (e) The signed density differential u = -div(qv) = dtheta y_1 y_2 q: red shows increasing density and blue decreasing density. Ordinary divergence of the shear velocity is zero, but weighted divergence is not. (f) The Fisher–Rao cost density u^2/q, whose whole-space integral is (dtheta)^2 = 1. (g) The Wasserstein–Otto cost density q||grad psi||^2, whose integral is (dtheta)^2/2 = 0.5. (h) The energy density q||v||^2 of the prescribed shear, whose integral is (dtheta)^2 = 1; this is not the WO cost. The discarded rotation has integrated energy 0.5 and is orthogonal to grad psi in L^2(q), explaining the energy difference after integration. This orthogonality does not assert pointwise additivity of the energy densities. The equality between FR cost and raw shear energy in this example is incidental. Panels (f–h) share one numerical color scale, and all panels use identical spatial limits. Faint contours in (b–h) show the current density q. The integrals are squared tangent norms or velocity energies, not endpoint distances, and do not include either display factor. Only panel (a) shows a finite sheared density; the metric calculation is at the undeformed Gaussian.'''


def main():
    diagnostics = verify()
    plt.rcParams.update({'font.family':'serif', 'mathtext.fontset':'dejavuserif',
                         'font.size':10, 'axes.spines.top':False, 'axes.spines.right':False})
    axis = np.linspace(-3.3,3.3,401)
    y1,y2 = np.meshgrid(axis,axis)
    p,u,fr,wo,raw = fields(y1,y2)
    fig,axs = plt.subplots(2,4,figsize=(16,8.6),layout='constrained')
    fig.suptitle('Shear: the prescribed motion and the motion WO needs\n'
                 r'$\phi_\theta(x)=(x_1+\theta x_2,x_2),\quad \theta=0,\quad d\theta=1$',fontsize=18)
    extent=[axis[0],axis[-1],axis[0],axis[-1]]
    titles=['(a) A finite shear', '(b) Induced velocity', '(c) Gradient projection', '(d) Density-preserving rotation',
            '(e) Density variation', '(f) Fisher–Rao', '(g) Wasserstein–Otto', '(h) Prescribed-motion energy']
    for ax,title in zip(axs.flat,titles):
        ax.set(title=title,xlim=(-3.3,3.3),ylim=(-3.3,3.3),aspect='equal',xlabel='$y_1$',ylabel='$y_2$')
        ax.set_xticks([-3,0,3]); ax.set_yticks([-3,0,3])
        ax.set_title(title,pad=12,fontsize=12)
    levels = [.015,.055,.12]
    angles=np.linspace(0,2*np.pi,401)
    for r in (.75,1.5,2.3):
        x,y=r*np.cos(angles),r*np.sin(angles)
        axs[0,0].plot(x,y,color='#2867a3',lw=1.4,label='$q$' if r==.75 else None)
        axs[0,0].plot(x+EPSILON*DTHETA*y,y,color='#c95548',lw=1.4,
                      label=r'$p_{\varepsilon d\theta}$' if r==.75 else None)
    line=np.linspace(-2.3,2.3,100)
    for tick in np.linspace(-2,2,9):
        axs[0,0].plot(line+EPSILON*tick,np.full_like(line,tick),color='.6',alpha=.4,lw=.6)
        axs[0,0].plot(tick+EPSILON*line,line,color='.6',alpha=.4,lw=.6)
    axs[0,0].legend(loc='upper left',frameon=False,fontsize=9)
    axs[0,0].text(.5, -.24,r'$\varepsilon d\theta=0.65$ (display only)',transform=axs[0,0].transAxes,ha='center')
    a,b=np.meshgrid(np.linspace(-2.4,2.4,9),np.linspace(-2.4,2.4,9))
    vectors=[(DTHETA*b,np.zeros_like(a)),(DTHETA*b/2,DTHETA*a/2),(DTHETA*b/2,-DTHETA*a/2)]
    labels=[r'$v=(y_2,0)$',r'$\nabla\psi=\frac{1}{2}(y_2,y_1)$',r'$w=\frac{1}{2}(y_2,-y_1)$']
    for ax,(vx,vy),label,color in zip(axs[0,1:],vectors,labels,['#a27612','#16877e','#8350a0']):
        ax.contour(y1,y2,p,levels=levels,colors='#2867a3',linewidths=.8,alpha=.45)
        ax.quiver(a,b,ARROW_SCALE*vx,ARROW_SCALE*vy,angles='xy',scale_units='xy',scale=1,color=color,width=.005)
        ax.text(.5,-.24,label,transform=ax.transAxes,ha='center',fontsize=12)
    signed=axs[1,0].imshow(u,origin='lower',extent=extent,cmap='coolwarm',norm=TwoSlopeNorm(0,-abs(u).max(),abs(u).max()))
    fig.colorbar(signed,ax=axs[1,0],orientation='horizontal',fraction=.045,pad=.13,label='Signed density rate')
    cmap=LinearSegmentedColormap.from_list('energy',['#ffffff','#b3cfe3','#487ca7','#ad5361','#c73132'])
    norm=Normalize(0,max(fr.max(),wo.max(),raw.max()))
    for ax,values in zip(axs[1,1:],[fr,wo,raw]):
        cost=ax.imshow(values,origin='lower',extent=extent,cmap=cmap,norm=norm)
    fig.colorbar(cost,ax=list(axs[1,1:]),orientation='horizontal',fraction=.045,pad=.13,label='Cost / energy density (shared scale)')
    formulas=[r'$u=y_1y_2q$',r'$u^2/q$'+'\nIntegral = 1.00',r'$q\|\nabla\psi\|^2$'+'\nIntegral = 0.50',r'$q\|v\|^2$'+'\nIntegral = 1.00']
    for ax,label in zip(axs[1],formulas):
        ax.contour(y1,y2,p,levels=levels[:2],colors='#455b70',linewidths=.65,alpha=.35)
        ax.text(.5,.98,label,transform=ax.transAxes,ha='center',va='top',fontsize=11,
                bbox=dict(facecolor='white',edgecolor='none',alpha=.88,pad=2))
    base=Path(__file__).resolve().parent
    output=base/'output'; output.mkdir(exist_ok=True)
    for ext in ('png','pdf'):
        fig.savefig(output/f'pushforward_shear_metrics.{ext}',dpi=220)
    plt.close(fig)
    for name in ('pushforward_shear_metrics_caption.txt','caption.txt'):
        (base/name).write_text(CAPTION+'\n',encoding='utf-8')
    (output/'pushforward_shear_metrics_diagnostics.json').write_text(json.dumps(diagnostics,indent=2)+'\n')
    print(json.dumps(diagnostics,indent=2))


if __name__=='__main__':
    main()
