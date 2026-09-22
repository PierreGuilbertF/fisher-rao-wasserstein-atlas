"""Spatial frequency versus FR/WO cost for a one-dimensional push-forward.
Run: python visualization_scripts/pushforward_oscillating_metrics.py
Requires numpy and matplotlib. Outputs PNG/PDF and caption files.
"""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

FREQUENCIES = (1., 4.)
DTHETA = 1.
BLUE, RED, PURPLE, TEAL = '#2867a3', '#c95548', '#8350a0', '#16877e'


def fields(y, k):
    q = np.exp(-y*y/2)/np.sqrt(2*np.pi)
    v = DTHETA*np.sin(k*y)
    derivative = DTHETA*k*np.cos(k*y)
    u = q*(y*v-derivative)
    return q, v, u, q*(y*v-derivative)**2, q*v*v


def costs(k):
    wo = DTHETA**2*(1-np.exp(-2*np.asarray(k)**2))/2
    fr = wo+DTHETA**2*np.asarray(k)**2*(1+np.exp(-2*np.asarray(k)**2))/2
    return fr, wo


def verify():
    y = np.linspace(-10, 10, 40001)
    results=[]
    for k in FREQUENCIES:
        q,v,u,fr,wo=fields(y,k)
        actual=np.array([np.trapezoid(a,y) for a in (q,u,fr,wo)])
        expected=[1,0,*costs(k)]
        assert np.allclose(actual,expected,atol=1e-10)
        h=1e-5
        def flux(y):
            a=fields(y,k)
            return a[0]*a[1]
        probes=np.linspace(-3,3,71)
        assert np.allclose(-(flux(probes+h)-flux(probes-h))/(2*h),fields(probes,k)[2],atol=1e-8)
        # Independent density derivative via inversion of the perturbed map.
        def pushed_density(theta):
            x=probes.copy()
            for _ in range(8):
                x-=(x+theta*np.sin(k*x)-probes)/(1+theta*k*np.cos(k*x))
            return np.exp(-x*x/2)/(np.sqrt(2*np.pi)*(1+theta*k*np.cos(k*x)))
        derivative=(pushed_density(h*DTHETA)-pushed_density(-h*DTHETA))/(2*h)
        assert np.allclose(derivative,fields(probes,k)[2],atol=1e-8)
        results.append(dict(k=k,integral_u=float(actual[1]),FR_squared_norm=float(actual[2]),WO_squared_norm=float(actual[3])))
    return results


CAPTION = r'''Rapid spatial variation at fixed velocity amplitude. For each fixed frequency k, consider the one-parameter push-forward family generated from q = N(0,1) by phi_theta(x) = x + theta sin(kx), with |theta|k < 1. The figure evaluates the density differential and both metric costs at theta = 0, dtheta = 1, so p_0 = q. The two rows show k = 1 and k = 4; k indexes different families and is not the tangent parameter. Columns show the induced velocity v(y) = sin(ky), the density variation u = -partial_y(qv) = q[y sin(ky) - k cos(ky)], the Fisher–Rao local squared-cost density u^2/q, and the Wasserstein–Otto local squared-cost density q v^2. Positive and negative velocity or density variation are colored red and blue. Since the one-dimensional velocity is a gradient, it is already the minimum-energy velocity realizing u, so q v^2 is the WO integrand. For standard Gaussian q, integration by parts gives integral(u^2/q) = integral q[(v')^2 + v^2]; this is an equality of integrals, not of pointwise integrands. Thus WO measures velocity amplitude, while FR also detects rapid spatial changes of velocity that compress and expand the density. The lower plot shows the whole-space squared costs versus frequency: g_WO = (1-exp(-2k^2))/2 and g_FR = g_WO + k^2(1+exp(-2k^2))/2. The WO cost approaches 1/2, while the FR cost grows like k^2/2. Markers correspond to the two displayed rows. Spatial axes and vertical scales are shared within each column, and the two cost columns share the same vertical scale. All printed integrals are whole-space squared tangent norms, not endpoint distances. This comparison fixes velocity amplitude, not the norm of the density variation. No finite parameter increment is shown: the admissible range |theta| < 1/k shrinks with k, but dtheta remains a valid tangent direction at theta = 0 for each fixed k.'''


def main():
    diagnostics=verify()
    plt.rcParams.update({'font.family':'serif','mathtext.fontset':'dejavuserif','font.size':11,
                         'axes.spines.top':False,'axes.spines.right':False})
    fig=plt.figure(figsize=(16,10))
    grid=fig.add_gridspec(3,4,height_ratios=[1,1,.8],left=.07,right=.985,top=.82,bottom=.08,wspace=.25,hspace=.48)
    axes=np.array([[fig.add_subplot(grid[i,j]) for j in range(4)] for i in range(2)])
    y=np.linspace(-4,4,4001)
    titles=['Velocity: same amplitude','Density variation','Fisher–Rao cost density','Wasserstein–Otto cost density']
    formulas=[r'$v=\sin(ky)$',r'$u=q[y\sin(ky)-k\cos(ky)]$',r'$u^2/q$',r'$qv^2$']
    for i,k in enumerate(FREQUENCIES):
        q,v,u,fr,wo=fields(y,k)
        for j,values in enumerate((v,u,fr,wo)):
            ax=axes[i,j]
            ax.set(xlim=(-4,4),xlabel='$y$')
            ax.set_xticks([-4,-2,0,2,4]); ax.grid(alpha=.13)
            if j<2:
                ax.axhline(0,color='.6',lw=.6)
                ax.plot(y,values,color=RED,lw=1.3)
                ax.fill_between(y,0,values,where=values>=0,color=RED,alpha=.2)
                ax.fill_between(y,0,values,where=values<0,color=BLUE,alpha=.2)
                ax.set_ylim((-1.4,1.7) if j==0 else (-1.9,2.5))
            else:
                color=PURPLE if j==2 else TEAL
                ax.plot(y,values,color=color,lw=1.5)
                ax.fill_between(y,0,values,color=color,alpha=.25)
                ax.set_ylim(0,8.4)
                total=costs(k)[j-2]
                ax.text(.5,.83,f'Integral = {total:.3f}',transform=ax.transAxes,ha='center',color=color,fontsize=12)
            ax.text(.5,.98,formulas[j],transform=ax.transAxes,ha='center',va='top',fontsize=11)
            if i==0: ax.set_title(titles[j],pad=17,fontsize=13)
        axes[i,0].set_ylabel(rf'$k={k:g}$',fontsize=16)
    ax=fig.add_subplot(grid[2,:2])
    k=np.linspace(0,5,600)
    fr,wo=costs(k)
    ax.plot(k,fr,color=PURPLE,lw=2,label='FR')
    ax.plot(k,wo,color=TEAL,lw=2,label='WO')
    for freq in FREQUENCIES:
        f,w=costs(freq)
        ax.scatter([freq],[f],color=PURPLE,zorder=4)
        ax.scatter([freq],[w],color=TEAL,zorder=4)
        ax.axvline(freq,color='.6',lw=.7,ls=':',zorder=0)
    ax.axhline(.5,color=TEAL,lw=.7,ls='--',alpha=.65)
    ax.set(xlabel='Spatial frequency $k$',ylabel='Squared tangent norm',xlim=(0,5),ylim=(0,14))
    ax.legend(frameon=False);ax.grid(alpha=.13)
    explanation=fig.add_subplot(grid[2,2:]);explanation.axis('off')
    explanation.text(0,.9,r'$g^{\mathrm{WO}}_0(d\theta,d\theta)=\int qv^2\,dy\ \longrightarrow\ \frac{1}{2}$',fontsize=16)
    explanation.text(0,.55,r'$g^{\mathrm{FR}}_0(d\theta,d\theta)=\int q\left[(\partial_y v)^2+v^2\right]dy\ \sim\ \frac{k^2}{2}$',fontsize=16)
    explanation.text(0,.1,'Opposing velocities over shorter spatial intervals\nproduce stronger alternating density changes.',fontsize=13,linespacing=1.6)
    fig.suptitle('Rapid spatial variation: bounded speed, increasing Fisher–Rao cost',fontsize=20,y=.97)
    fig.text(.5,.935,r'$q=\mathcal{N}(0,1),\quad \phi_\theta(x)=x+\theta\sin(kx),\quad \theta=0,\quad d\theta=1$'
             '\nEach row: a fixed frequency; all costs evaluated at the same reference density.',ha='center',va='top',fontsize=12,linespacing=1.7)
    base=Path(__file__).resolve().parent;out=base/'output';out.mkdir(exist_ok=True)
    for ext in ('png','pdf'):fig.savefig(out/f'pushforward_oscillating_metrics.{ext}',dpi=220)
    plt.close(fig)
    for name in ('pushforward_oscillating_metrics_caption.txt','caption.txt'):(base/name).write_text(CAPTION+'\n',encoding='utf-8')
    (out/'pushforward_oscillating_metrics_diagnostics.json').write_text(json.dumps(diagnostics,indent=2)+'\n')
    print(json.dumps(diagnostics,indent=2))


if __name__=='__main__':
    main()
