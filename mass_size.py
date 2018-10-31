from __future__ import print_function, division

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import astropy.units as u
from astropy.constants import G

import scipy.stats

def gap_size(M, rs, t, r, w, wperp, b, caustic=True):
    """"""

    delta_c = 4 * np.sqrt(2*G*M*t/(w*r**2))
    brs = np.sqrt(b**2 + rs**2)
    delta_nonc = 2*w/wperp * brs/r + 2*G*M*wperp*t/(w**2*r*brs)
    
    delta = np.empty_like(delta_c)
    delta[caustic] = delta_c[caustic]
    delta[~caustic] = delta_nonc[~caustic]
    
    return (delta.decompose()*u.radian).to(u.deg)

def gap_depth(M, rs, t, r, w, wperp, b):
    """"""
    
    f = (1 + (wperp**2/w**3 * 2*G*M*t/(b**2+rs**2)).decompose())**-1
    
    return f

def time_caustic(M, rs, t, r, w, wperp, b):
    """"""
    
    tc = 4*w**3/wperp**2 * (b**2 + rs**2)/(G*M)
    
    return tc.to(u.Gyr)

def test():
    """"""
    
    M = 10**6 * u.Msun
    rs = 100*u.pc
    t = 0.5*u.Gyr
    r = 20*u.kpc
    w = 100 * u.km/u.s
    wperp = 100 * u.km/u.s
    b = rs
    
    tcaustic = time_caustic(M, rs, t, r, w, wperp, b)
    flag_caustic = t>tcaustic
    
    f = gap_depth(M, rs, t, r, w, wperp, b)
    delta = gap_size(M, rs, t, r, w, wperp, b, caustic=flag_caustic)
    
    print(flag_caustic, f, delta)

def get_mass(delta, f, t, r, w, wperp, b):
    """"""
    
    finv = f**-1
    m = r**2*w*(delta.to(u.radian).value)**2*(finv - 1) / (2*G*t*(finv +1)**2)
    
    return m.to(u.Msun)

def get_rs(delta, f, t, r, w, wperp, b):
    """"""
    
    rs = np.sqrt((r*wperp*delta.to(u.radian).value/(w*(f**-1 + 1)))**2 - b**2)
    
    return rs.to(u.pc)

def test_gap_inference():
    """"""
    
    N = 1000
    t = 0.5*u.Gyr
    r = 20*u.kpc
    w = 100 * u.km/u.s
    wperp = 100 * u.km/u.s
    b = 100*u.pc
    delta = 2 * u.deg
    f = 0.5
    
    m = get_mass(delta, f, t, r, w, wperp, b)
    rs = get_rs(delta, f, t, r, w, wperp, b)
    
    print(m, rs)

def gap_inference(seed=4356):
    """"""
    
    delta = 4*u.deg
    f = 0.5

    r = 20*u.kpc
    sigma = 180 * u.km/u.s
    bmax = 500 * u.pc
    
    N = 1000
    np.random.seed(seed)
    t = np.random.triangular(0,7,7, size=N) * u.Gyr
    b = np.random.rand(N) * bmax
    
    w = scipy.stats.maxwell.rvs(scale=sigma, size=N) * sigma.unit
    wperp = w
    
    m = get_mass(delta, f, t, r, w, wperp, b)
    rs = get_rs(delta, f, t, r, w, wperp, b)

    plt.close()
    plt.figure()
    
    plt.plot(rs, m, 'k.')
    
    plt.gca().set_xscale('log')
    plt.gca().set_yscale('log')
    plt.xlabel('$r_s$ [pc]')
    plt.ylabel('M / M$_\odot$')
    
    plt.tight_layout()
