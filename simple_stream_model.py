# simple_stream_model.py: (very!) simple prescriptions for the expected
#                         amplitude and size of gaps in stellar streams
#                         from different subhalo masses
import numpy
import astropy.units as u
from astropy.constants import G
from galpy.potential import MWPotential2014, evaluateR2derivs


def rs(Mhalo):
    """
    NAME:
       rs
    PURPOSE:
       return the scale radius of a halo of a given mass using the relation from Erkal et al. (2016) / Bovy et al. (2017)
    INPUT:
       Mhalo - Mass of the subhalo (in units of 10^7 Msun or as an astropy Quantity)
    OUTPUT:
       scale radius
    HISTORY:
       2018-03-06 - Written - Bovy (UofT)
    """
    if isinstance(Mhalo, u.Quantity):
        Mhalo = Mhalo.to(u.Msun).value / 10.**7.
    return 1.05 * u.kpc * (Mhalo / 10.)**0.5 / 1.05 * 0.625


def gamma(R, z):
    if not isinstance(R, u.Quantity):
        R = R * u.kpc
    if not isinstance(z, u.Quantity):
        z = z * u.kpc
    r = numpy.sqrt(R**2. + z**2.)
    return numpy.sqrt(3. + r**2. / (220. * u.km / u.s)**2.
                      * evaluateR2derivs(MWPotential2014, R, z, ro=8., vo=220.)) / 1.9 * 1.5


def gap_size(Mhalo, dist=10. * u.kpc,
             gap_amp=None, X=1., wwperpfac=numpy.sqrt(3. / 2.),
             w=150., timpact=1., R=0. * u.kpc, z=20. * u.kpc, scale_radius=None):
    """
    NAME:
       gap_size
    PURPOSE:
       return the typical gap size for a given subhalo mass
    INPUT:
       Mhalo - Mass of the subhalo (in units of 10^7 Msun or as an astropy Quantity)
       dist= (10.) distance of the stream (in units of kpc or as an astropy Quantity)
       gap_amp= (default: use gap_depth function with defaults) amplitude (depth) of the gap
       X= (1) ratio of impact parameter to scale radius
       wwperpfac= (sqrt[3/2]) factor w/wperp of relative impact velocity
       timpact= (1.) time since impact (Gyr or astropy Quantity)
       w= (150.) relative velocity of the fly-by (km/s or astropy Quantity)
       R= (0) Galactocentric cylindrical radius of stream (kpc or Quantity)
       z= (20) Galactocentric height of stream (kpc of Quantity)
    OUTPUT:
       size of the gap
    HISTORY:
       2018-02-06 - Written - Bovy (UofT)
    """
    if not isinstance(Mhalo, u.Quantity):
        Mhalo = Mhalo * u.Msun * 10.**7.
    if not isinstance(dist, u.Quantity):
        dist = dist * u.kpc
    if not isinstance(timpact, u.Quantity):
        timpact = timpact * u.Gyr
    if not isinstance(w, u.Quantity):
        w = w * u.km / u.s
    if not isinstance(R, u.Quantity):
        R = R * u.km / u.s
    if not isinstance(z, u.Quantity):
        z = z * u.km / u.s
    if gap_amp is None:
        gap_amp = gap_depth(Mhalo, wwperpfac=wwperpfac, timpact=timpact,
                            w=w, X=X, R=R, z=z)
    # Using Erkal et al. (2016), Eqn. (21) and (22)
    tgamma = gamma(R, z)

    if scale_radius == None:
        scale_radius = rs(Mhalo)

    tcaustic = 4.**tgamma**2. / (4. - tgamma**2.) * wwperpfac**2. * w * (X**2. + 1.)\
        * scale_radius**2. / G / Mhalo
    B = numpy.sqrt(X**2. + 1.) * scale_radius * wwperpfac / dist
    if timpact < tcaustic:
        return (B * (1. + 1. / gap_amp))\
            .to(u.deg, equivalencies=u.dimensionless_angles())
    else:
        return (4. * B * numpy.sqrt(1. / gap_amp - 1.))\
            .to(u.deg, equivalencies=u.dimensionless_angles())


def gap_depth(Mhalo, wwperpfac=numpy.sqrt(3. / 2.),
              w=150., X=1., timpact=1., R=0. * u.kpc, z=20. * u.kpc, , scale_radius=None, **kwargs):
    """
    NAME:
       gap_depth
    PURPOSE:
       return the typical gap depth for a given subhalo mass
    INPUT:
       Mhalo - Mass of the subhalo (in units of 10^7 Msun or as an astropy Quantity)
       wwperpfac= (sqrt[3/2]) factor w/wperp of relative impact velocity      
       w= (150.) relative velocity of the fly-by (km/s or astropy Quantity)
       X= (1) ratio of impact parameter to scale radius
       timpact= (1.) time since impact (Gyr or astropy Quantity)
       R= (0) Galactocentric cylindrical radius of stream (kpc or Quantity)
       z= (20) Galactocentric height of stream (kpc of Quantity)
    OUTPUT:
       fractional gap depth
    HISTORY:
       2018-03-06 - Written - Bovy (UofT)
    """
    if not isinstance(Mhalo, u.Quantity):
        Mhalo = Mhalo * u.Msun * 10.**7.
    if not isinstance(w, u.Quantity):
        w = w * u.km / u.s
    if not isinstance(timpact, u.Quantity):
        timpact = timpact * u.Gyr
    if not isinstance(R, u.Quantity):
        R = R * u.km / u.s
    if not isinstance(z, u.Quantity):
        z = z * u.km / u.s
    tgamma = gamma(R, z)
    if scale_radius == None:
        scale_radius = rs(Mhalo)

    return 1. / (1. + (4. - tgamma**2.) / tgamma**2. / wwperpfac**2. / w * 2. * G * Mhalo / (X**2. + 1.) / scale_radius**2. * timpact).value
