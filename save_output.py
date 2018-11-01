from __future__ import division

import os
import matplotlib.pyplot as plt

import astropy.table as atpy
import read_girardi
import numpy as np
import scipy.spatial
import astropy.units as auni
import stream_num_cal as snc
import simple_stream_model as sss
import matplotlib.pyplot as plt
import scipy.interpolate
import scipy.optimize

import mock_sim
import simple_stream_model


def save_output(filename='output.txt', mus=[30.], distances=[20.], velocities=[150.], impact_parameters=[1.], maglims=[None], latitudes=[60.], surveys=['LSST'], gap_fill=True, **kwargs):
    if os.path.exists(filename):
        pass
    else:
        output_file = open(filename, 'w')
        output_file.write('# distance (kpc), flyby_velocity (km/s), impact_parameter (r_s), magnitude_limit (mag), latitude (deg), gap_fill (True/False), survey, surface_brightness (mag/arcsec^2), minimum_mass (M_sun)\n')
        output_file.close()

    output = np.genfromtxt(filename, unpack=True, delimiter=', ', dtype=None, names=['dist', 'w', 'b', 'maglim', 'lat', 'gap_fill', 'survey', 'mu', 'mass'], encoding='bytes')

    for survey in surveys:
        for lat in latitudes:
            for maglim in maglims:
                for b in impact_parameters:
                    for w in velocities:
                        for distance in distances:

                            ret = []
                            if maglim == None:
                                maglim_label = mock_sim.getMagLimit('g', survey)

                            for mu in mus:
                                # print mu, distance, w, b, maglim, lat, survey
                                # check if output already saved for these params
                                try:
                                    idx = (output['dist'] == distance) & (output['w'] == w) & (output['b'] == b) & (output['maglim'] == np.around(
                                        maglim_label, 2)) & (output['survey'] == survey) & (output['gap_fill'] == gap_fill) & (output['mu'] == mu)
                                    if np.sum(idx) > 0:
                                        print 'Output exists'
                                        continue
                                except:
                                    pass
                                mockfile = '%d_deg_mock.fits' % lat
                                mass, gapt, gapo = mock_sim.predict_gap_depths(mu, distance, survey, width_pc=20, maglim=maglim,
                                                                               timpact=0.5, gap_fill=gap_fill, w=w, X=b, mockfile=mockfile, **kwargs)
                                xind = np.isfinite(gapo / gapt)
                                try:
                                    II1 = scipy.interpolate.UnivariateSpline(np.log10(mass)[xind], (gapo / gapt - 1)[xind], s=0)
                                except:
                                    print 'Spline error'
                                    ret.append(np.nan)
                                    continue
                                R = scipy.optimize.root(II1, 6)
                                ret.append(10**R['x'])

                                with open(filename, 'a') as output_file:
                                    output_file.write('%.2f, %.2f, %.2f, %.2f, %.2f, %d, %s, %.2f, %.2f\n' % (distance, w, b, maglim_label, lat, gap_fill, survey, mu, ret[-1]))


if __name__ == "__main__":
    save_output(mus=[30, 31, 32, 33], distances=[10, 20, 30, 40], surveys=['SDSS', 'LSST10'])
