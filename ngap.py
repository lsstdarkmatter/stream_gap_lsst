from __future__ import print_function, division

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import astropy.units as u
from astropy.constants import G
from astropy.table import Table
import scipy

from mass_size import *

def nencounter(bmax, length, t, sigma, nsub):
    """Calculate the number of encounters for a given stream"""
    
    n = np.int64((np.sqrt(0.5*np.pi) * bmax * length * nsub * sigma * t).decompose())
    
    return n

def sample_encounters(N=100, seed=134698, fcut=0.75):
    """"""
    
    length = 9 * u.kpc
    tmax = 7.7 * u.Gyr
    r = 19 * u.kpc
    vcirc = 220 * u.km/u.s
    nsub = 8.66e-4 * (u.kpc)**-3

    # subhalo mass parameters
    xmin = 10**5
    xmax = 10**9
    n = 1.9
    Nmass = 50000
    
    # impact parameters
    bmax = 5 * u.kpc
    
    # velocity parameters
    sigma = 180 * u.km/u.s
    vmin = -1000*u.km/u.s
    vmax = 1000*u.km/u.s
    
    
    np.random.seed(seed)
    # sample encounter times
    t = np.random.triangular(0, tmax.value, tmax.value, size=N) * tmax.unit
    
    # sample impact parameters
    b = np.random.rand(N) * bmax
    
    # sample subhalo masses
    m = sample_pdf(pdf_mass, xmin, xmax, size=N, args=[n, xmin, xmax], N=Nmass)*u.Msun
    rs = rs_nfw(m)
    
    # sample velocities
    wpar = sample_pdf(pdf_wpar, vmin, vmax, size=N, args=[vcirc, sigma])
    wperp = np.abs(sample_pdf(pdf_wperp, vmin, vmax, size=N, args=[sigma]))
    w = np.sqrt(wpar**2 + wperp**2)
    
    tcaustic = time_caustic(m, rs, t, r, w, wperp, b)
    flag_caustic = t>tcaustic
    
    delta = gap_size(m, rs, t, r, w, wperp, b, caustic=flag_caustic)
    f = gap_depth(m, rs, t, r, w, wperp, b)
    
    gap = f<fcut
    nexp = nencounter(bmax, length, tmax, sigma, nsub)
    print(np.sum(gap), np.sum(gap)*nexp/np.size(gap))
    
    plt.close()
    fig, ax = plt.subplots(1,2, figsize=(10,5))
    
    plt.sca(ax[0])
    plt.hist(f)
    plt.xlim(0,1)
    
    plt.sca(ax[1])
    plt.hist(delta.value)
    plt.xlim(0,100)
    
    plt.tight_layout()

def rs_nfw(m):
    """"""
    
    rs = 1.62*u.kpc * np.sqrt(m.to(u.Msun).value*1e-8)
    
    return rs


def pdf_const(x):
    """"""
    
    return np.zeros_like(x)+1

def pdf_norm(x, mu, sigma):
    """"""
    
    f = (2*np.pi*sigma**2)**-0.5 * np.exp(-(x - mu)**2/(2*sigma**2))
    
    return f

def pdf_wpar(x, mu, sigma):
    """"""
    
    f = (2*np.pi*sigma**2)**-0.5 * np.exp(-(x + mu)**2/(2*sigma**2))
    
    return f

def pdf_wperp(x, sigma):
    """"""
    
    f = np.sqrt(2/np.pi) * x**2 * sigma**-3 * np.exp(-x**2/(2*sigma**2))
    
    return f

def pdf_mass(x, n, xmin, xmax):
    """"""
    
    f = (1-n) / (xmax**(1-n) - xmin**(1-n)) * x**-n
    
    return f

def sample_pdf(pdf, xmin, xmax, size=1, N=10000, args=[]):
    """"""
    
    n = np.linspace(0,N,N)
    h = (xmax-xmin)/N
    v = np.cumsum(pdf(n*h + xmin, *args)*h)
    
    r = np.random.rand(size)
    
    vr = v[np.newaxis,:] - r[:,np.newaxis]
    x = np.argmin(np.abs(vr), axis=1)*h + xmin
    
    return x


def test_velocities(seed=39862):
    """"""

    xmin = -1000
    xmax = 1000
    mu = 220
    sigma = 180
    
    N = 10000
    np.random.seed(seed)
    vpar = sample_pdf(pdf_wpar, xmin, xmax, size=N, args=[mu, sigma])
    vperp = sample_pdf(pdf_wperp, xmin, xmax, size=N, args=[sigma])
    vtot = np.sqrt(vpar**2 + vperp**2)
    
    varray = np.linspace(xmin, xmax, N)
    
    plt.close()
    plt.hist(vpar, bins=20, histtype='step', lw=2, normed=True)
    plt.hist(vperp, bins=20, histtype='step', lw=2, normed=True)
    plt.hist(vtot, bins=20, histtype='step', lw=2, normed=True)
    
    plt.plot(varray, pdf_wpar(varray, mu, sigma), 'k-')
    plt.plot(varray, pdf_wperp(varray, sigma), 'k-')

def test_mass(seed=2348):
    """"""
    
    xmin = 10**5
    xmax = 10**9
    a0 = 1.77e-5
    m0 = 2.52e7
    n = 1.9
    m0 = 1e5
    
    N = 10000
    np.random.seed(seed)
    
    m = sample_pdf(pdf_mass, xmin, xmax, size=N, args=[n, xmin, xmax], N=50000)
    
    m_array = np.logspace(5,9,100)
    
    plt.close()
    
    plt.hist(m, bins=20, normed=True, log=True)
    plt.plot(m_array, pdf_mass(m_array, n, xmin, xmax))
    
    plt.gca().set_xscale('log')
    plt.gca().set_yscale('log')
    
    plt.tight_layout()

def test_sampling():
    """"""
    
    N = 10000
    n = np.linspace(0,N,N)
    xmin = -5
    xmax = 5
    h = (xmax-xmin)/N
    v = np.cumsum(pdf_norm(n*h + xmin, 0, 1)*h)
    
    seed = 543
    r = np.random.rand(10000)
    
    vr = v[np.newaxis,:] - r[:,np.newaxis]
    x = np.argmin(np.abs(vr), axis=1)*h + xmin
    
    plt.close()
    plt.hist(x, normed=True, bins=20)
    
    x_ = np.linspace(-5,5,1000)
    y_ = pdf_norm(x_, 0, 1)
    
    plt.plot(x_, y_, '-')


def input_legacy():
    """"""
    
    streams = ['gd1', 'pal5', 'tri', 'atlas', 'phoenix', 'styx']
    length = np.array([9, 9, 12, 6, 4, 50]) * u.kpc
    tmax = np.array([7.7, 3.4, 9.3, 2.1, 1.8, 13]) * u.Gyr
    r = np.array([19, 13, 40, 22, 19, 45]) * u.kpc
    vcirc = np.array([220, 220, 190, 220, 220, 190]) * u.km/u.s
    nsub = np.array([8.66e-4, 1.01e-3, 5.51e-4, 8.06e-4, 8.66e-4, 5.01e-4]) * (u.kpc)**-3
    fcut = np.ones(np.size(nsub)) * 0.75
    
    t = Table([streams, length, tmax, r, vcirc, nsub, fcut], names=('name', 'length', 'age', 'rgal', 'vcirc', 'nsub', 'fcut'))
    t.pprint()
    
    t.write('legacy_streams.fits', overwrite=True)

def ngaps_perstream(length, tmax, r, vcirc, nsub, fcut, xmin=10**5, xmax=10**9, n=1.9, Nmass=50000, bmax=5*u.kpc, sigma=180*u.km/u.s, vmin=-1000*u.km/u.s, vmax=1000*u.km/u.s, seed=542734, N=1000):
    """"""
    
    #length = 9 * u.kpc
    #tmax = 7.7 * u.Gyr
    #r = 19 * u.kpc
    #vcirc = 220 * u.km/u.s
    #nsub = 8.66e-4 * (u.kpc)**-3

    ## subhalo mass parameters
    #xmin = 10**5
    #xmax = 10**9
    #n = 1.9
    #Nmass = 50000
    
    ## impact parameters
    #bmax = 5 * u.kpc
    
    ## velocity parameters
    #sigma = 180 * u.km/u.s
    #vmin = -1000*u.km/u.s
    #vmax = 1000*u.km/u.s
    
    np.random.seed(seed)
    # sample encounter times
    t = np.random.triangular(0, tmax.value, tmax.value, size=N) * tmax.unit
    
    # sample impact parameters
    b = np.random.rand(N) * bmax
    
    # sample subhalo masses
    m = sample_pdf(pdf_mass, xmin, xmax, size=N, args=[n, xmin, xmax], N=Nmass)*u.Msun
    rs = rs_nfw(m)
    
    # sample velocities
    wpar = sample_pdf(pdf_wpar, vmin, vmax, size=N, args=[vcirc, sigma])
    wperp = np.abs(sample_pdf(pdf_wperp, vmin, vmax, size=N, args=[sigma]))
    w = np.sqrt(wpar**2 + wperp**2)
    
    tcaustic = time_caustic(m, rs, t, r, w, wperp, b)
    flag_caustic = t>tcaustic
    
    delta = gap_size(m, rs, t, r, w, wperp, b, caustic=flag_caustic)
    f = gap_depth(m, rs, t, r, w, wperp, b)
    
    gap = f<fcut
    nexp = nencounter(bmax, length, tmax, sigma, nsub)
    ngap_tot = np.sum(gap)
    ngap_obs = ngap_tot * nexp / N
    
    return ngap_obs

def get_gaps(streams='legacy', N=1000):
    """Get the number of detactable gaps per stream"""
    
    t = Table.read('{}_streams.fits'.format(streams))
    Nstream = len(t)
    ngaps = np.zeros(Nstream)
    
    for i in range(Nstream):
        t_ = t[i]
        ngaps[i] = ngaps_perstream(t_['length']*t['length'].unit, t_['age']*t['age'].unit, t_['rgal']*t['rgal'].unit, t_['vcirc']*t['vcirc'].unit, t_['nsub']*t['nsub'].unit, t_['fcut'], N=N)
    
    tout = Table([t['name'], ngaps], names=('name', 'ngaps'))
    tout.pprint()
    
    tout.write('ngaps_{}_lcdm.fits'.format(streams))

def lcdm_limits(streams='legacy'):
    """Show consistency with LCDM as a function of the total number of detected gaps in a set of streams"""
    
    t = Table.read('ngaps_{}_lcdm.fits'.format(streams))
    
    Ntot = np.int64(np.sum(t['ngaps']))
    x = np.int64(np.linspace(0, 2*Ntot, 100))
    
    q = [0.001, 0.05, 0.95, 0.999]
    levels = scipy.stats.poisson.ppf(q, Ntot)
    
    ytop = scipy.stats.poisson.pmf(Ntot, Ntot)
    yhalf = 0.5 * ytop
    
    plt.close()
    plt.figure(figsize=(8.5,5))
    
    plt.plot(x, scipy.stats.poisson.pmf(x, Ntot), '-', color=mpl.cm.Blues(0.75), lw=5, alpha=0.7)
    
    for e, l in enumerate(levels):
        plt.axvline(l, ls=':', lw=1.5, color='0.4', alpha=0.8)
        txt = plt.text(l, yhalf, '{:g}%'.format(q[e]*100), rotation=90, va='center', ha='center', fontsize='small')
        txt.set_bbox(dict(facecolor='w', alpha=1, ec='none'))
    
    #plt.text(Ntot, ytop*1.1, 'Consistent with $\Lambda$CDM', ha='center')
    plt.title('Consistent with $\Lambda$CDM', fontsize='medium')
    
    for s in ['left', 'top', 'right']:
        plt.gca().spines[s].set_visible(False)
    plt.tick_params(axis='both', which='both', top='off', right='off', left='off', labelleft='off')

    plt.xlabel('Number of gaps in {} streams with LSST'.format(streams))
    
    plt.tight_layout()




