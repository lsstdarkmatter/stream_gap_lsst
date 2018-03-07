import astropy.table as atpy
import read_girardi
import numpy as np
import scipy.spatial
import astropy.units as auni
import stream_num_cal as snc
import simple_stream_model as sss
import matplotlib.pyplot as plt


def betw(x, x1, x2): return (x >= x1) & (x <= x2)


def getMagErr(mag, filt, survey='LSST'):
    """
    Parameters
    ----------
    mag: float
        Magnitude
    filt: str
        Filter
    survey: str
        Survey
    Returns:
    -------
        err: float
        Magnitude uncertainty
    """
    if survey == 'LSST':
        import photErrorModel as pem
        lem = pem.LSSTErrorModel()
        #minmagerr = 0.01
        magerr = lem.getMagError(mag, 'LSST_'+filt)
        return magerr


def getMagErrVec(mag, filt, survey='LSST'):
    """ 
    Parameters:
    ----------
    mag: numpy
        The array of magnitudes
    filt: str
        The filter of observations
    Returns:
    err: numpy array
        The magnitude uncertainty
    
    """
    maggrid = np.linspace(15, 28, 1000)
    res = [getMagErr(m, filt, survey) for m in maggrid]
    res = scipy.interpolate.UnivariateSpline(maggrid, res, s=0)(mag)
    return res


def getMagLimit(filt, survey='LSST'):
    "A sophisticated calculation of LSST magntude limit"
    return 27


def getIsoCurve(iso, magstep = 0.01):
    """
    Returns the list of points sampling along the isochrone
    
    Arguments:
    ---------
    iso: dict
        Dictionary with the Girardi isochrone
    magstep: float(optional)
        The step in magntidues along the isochrone
    Returns:
    -------
    gcurve,rcurve: Tuple of numpy arrays
        The tupe of arrays of magnitudes in g and r going along the isochrone
    """
    mini = iso['M_ini']
    g = iso['DES-g']
    r = iso['DES-r']
    res_g, res_r = [], []
    for i in range(len(mini)-1):
        l_g, l_r = g[i], r[i]
        r_g, r_r = g[i+1], r[i+1]
        npt = max(abs(r_g-l_g), abs(r_r-l_r))/magstep+2
        ggrid = np.linspace(l_g, r_g, npt)
        rgrid = np.linspace(l_r, r_r, npt)
        res_g.append(ggrid)
        res_r.append(rgrid)
    res_g, res_r = [np.concatenate(_) for _ in [res_g, res_r]]
    return res_g, res_r


def get_mock_density(distance, isoname, survey,
                     mockfile='stream_gap_mock.fits', mockarea=100,
                     minerr=0.01):
    """
    Compute the density of background stars within the isocrhone mask 
    of the stellar population at a given distance

    Arguments:
    ---------
    distance: float
        Distance to the stream in kpc
    isoname: str
        The filename with the isochrone
    survey: str
        The name of the survey (currently only LSST)
    mockfile: str
        The name of the file with the mock galaxia data
    mockarea: float
        The area in square degrees used for the mockfile generation
    Returns:
    --------
    Stellar denstiy in stars/sq. deg

    """
    minerr = 0.02
    dm = 5*np.log10(distance*1e3)-5
    iso = read_girardi.read_girardi(isoname)
    xind = iso['stage'] <= 3 # cut the horizontal branch 
    for k,v in iso.items():
        iso[k] = v[xind]

    r_mag_limit = getMagLimit('r', survey)

    gcurve, rcurve = getIsoCurve(iso)
    gcurve, rcurve = [_ + dm for _ in [gcurve, rcurve]]

    mincol, maxcol = -0.3, 1.2
    minmag, maxmag = 17, 28
    colbin = 0.01
    magbin = 0.01

    grgrid, rgrid = np.mgrid[mincol:maxcol:colbin, minmag:maxmag:magbin]
    ggrid = grgrid + rgrid

    colbins = np.arange(mincol, maxcol, colbin)
    magbins = np.arange(minmag, maxmag, magbin)

    arr0 = np.array([(ggrid).flatten(), rgrid.flatten()]).T
    arr = np.array([gcurve, rcurve]).T
    tree = scipy.spatial.KDTree(arr)
    D, xind = tree.query(arr0)
    gerr = getMagErrVec(ggrid.flatten(), 'g', survey).reshape(ggrid.shape)
    rerr = getMagErrVec(rgrid.flatten(), 'r', survey).reshape(rgrid.shape)
    gerr, rerr = [np.maximum(_, minerr) for _ in [gerr, rerr]]

    dg = ggrid - gcurve[xind].reshape(ggrid.shape)
    dr = rgrid - rcurve[xind].reshape(rgrid.shape)

    thresh = 2  # how many sigma away from the isochrone we select

    mask = (np.abs(dg/gerr) < thresh) & (np.abs(dr/rerr) < thresh)
    dat = atpy.Table().read(mockfile)
    g, r = dat['g'], dat['r']
    colid = np.digitize(g - r, colbins)-1
    magid = np.digitize(r, magbins)-1
    xind = betw(colid, 0, grgrid.shape[0]-1) & betw(magid, 0, grgrid.shape[1]) & (
        r < r_mag_limit)
    nbgstars = xind.sum()
    bgdens = nbgstars/mockarea
    return bgdens

def predict_gap_depths(mu, distance_kpc, survey, width_pc, maglim=None):
    """
    Arguments:
    ---------
    mu: real
        Surface brightness of the stream in mag/sq.arcsec^2
    distance_kpc: real
        Distance to the stream in kpc
    survey: str
        Name of the survey
    width_pc: real
        The width of the stream in pc
    Returns:
    ---
    (masses,tdepths,odepths): Tuple of 3 numpy arrays
        The array of halo masses
        The array of theoretically predicted gap depths
        The array of potentially observable gap depths
    """
    isoname = 'iso_a12.0_z0.00020.dat'
    mockarea = 100
    mockfile = 'stream_gap_mock.fits'
    width_deg = np.rad2deg(width_pc/distance_kpc/1e3)
    mgrid = 10**np.linspace(5.5, 8.5, 10)
    mgrid7 = mgrid / 1e7
    gap_depths = np.array([1 - sss.gap_depth(_) for _ in mgrid7])
    # We do 1-gap_depth() because sss_gap_depth returns the height of 
    # the gap from zero rather than from 1.
    
    gap_sizes_deg = np.array(
        [sss.gap_size(_, dist=distance_kpc*auni.kpc)/auni.deg for _ in mgrid7])
    
    if maglim is None:
        maglim = getMagLimit('r', survey)
    dens_stream = snc.nstar_cal(mu, distance_kpc, maglim)
    dens_bg = get_mock_density(distance_kpc, isoname, survey,
                               mockfile=mockfile, mockarea=mockarea)
    print('Background/stream density [stars/sq.deg]', dens_bg, dens_stream)
    max_gap = 10 # this the maximum gap length that we consider reasonable
    N = len(gap_sizes_deg)
    detfracs = np.zeros(N)
    for i in range(N):
        area = 2 * width_deg * gap_sizes_deg[i]
        # twice the width and the length of the gap 
        nbg = dens_bg * area
        nstr = dens_stream * area
        print('Nstream', nstr, 'Nbg', nbg)
        detfrac = 3 * np.sqrt(nbg + nstr) / nstr
        # this is smallest gap depth that we could detect 
        # we the poisson noise on density is sqrt(nbg+nstr) 
        # and the stream density (per bin) is nstr 
        detfracs[i] = detfrac
        if gap_sizes_deg[i]>max_gap:
            detfracs[i]=np.nan
    return (mgrid, gap_depths, detfracs)
