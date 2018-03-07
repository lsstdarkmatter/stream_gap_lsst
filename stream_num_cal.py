import astropy.io.fits as pyfits
import numpy as np

data = pyfits.open('simulated_dwarf_1e6.fits')[1].data

def lum_cal(Mv):
    msun = 4.83
    return 10.**(0.4*(msun - Mv))


def nstar_cal(mu, distance, maglim_g, maglim_r, frac = 0.6):
    '''
    input:
    mu: surface brightness mag/arcsec^2
    distance: kpc
    maglim: limiting magnitude of the survey
    frac : fraction of the member stars get selected

    output:
    nstar: num of star per sq deg
    '''

    dm = 5*np.log10(distance*1000)-5
    lum_arcsq = mu - dm
    lstar = lum_cal(lum_arcsq) * 3600**2
    mtl = 1.4 # mass to light ratio is 1.4 for this isochrone
    mstar = lstar * mtl

    N1  = (((data['g']+dm) < maglim_g) & ((data['r']+dm)<maglim_r)).sum()
    nstar = N1 * (mstar/1e6) * frac

    return nstar


