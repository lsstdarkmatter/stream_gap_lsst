from __future__ import division

import os
import matplotlib.pyplot as plt
from collections import OrderedDict as odict

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

STREAMS = odict([
    ('Tucana_III',
     dict(distance=25.1, surface_brightness=32.0, width=32.0
          )),
    ('ATLAS',
     dict(distance=22.9, surface_brightness=33.0, width=96.0
          )),
    ('Phoenix',
     dict(distance=19.1, surface_brightness=32.6, width=53.0
          )),
    ('Indus',
     dict(distance=16.6, surface_brightness=31.9, width=240.0
          )),
    ('Jhelum',
     dict(distance=13.2, surface_brightness=33.3, width=267.0
          )),
    ('Ravi',
     dict(distance=22.9, surface_brightness=33.4, width=288.0
          )),
    ('Chenab',
     dict(distance=39.8, surface_brightness=34.1, width=493.0
          )),
    ('Elqui',
     dict(distance=50.1, surface_brightness=34.3, width=472.0
          )),
    ('Aliqa_Uma',
     dict(distance=28.8, surface_brightness=33.8, width=131.0
          )),
    ('Turbio',
     dict(distance=16.6, surface_brightness=32.6, width=72.0
          )),
    ('Willka_Yaku',
     dict(distance=34.7, surface_brightness=32.9, width=127.0
          )),
    ('Turranburra',
     dict(distance=27.5, surface_brightness=34.0, width=288.0
          )),
    ('Wambelong',
     dict(distance=15.1, surface_brightness=33.7, width=106.0
          )),
])


def min_gap_depth(distance_kpc, mu, width_pc, maglim_g, maglim_r, survey='LSST10', lat=60, gap_size=5.0):
    isoname = 'iso_a12.0_z0.00020.dat'
    mockfile = '%d_deg_mock.fits' % lat
    mockarea = 100

    width_deg = np.rad2deg(width_pc / distance_kpc / 1e3)

    dens_stream = mock_sim.snc.nstar_cal(mu, distance_kpc, maglim_g=maglim_g, maglim_r=maglim_r)
    dens_bg = mock_sim.get_mock_density(distance_kpc, isoname, survey, mockfile=mockfile, mockarea=mockarea, maglim_g=maglim_g, maglim_r=maglim_r)
    # print('Background/stream density [stars/sq.deg]', dens_bg, dens_stream)

    area = 2 * width_deg * gap_size  # twice the width and the length of the gap

    nbg = dens_bg * area
    nstr = dens_stream * area
    # print('Nstream', nstr, 'Nbg', nbg)

    detfrac = 5 * np.sqrt(nbg + nstr) / nstr  # this is smallest gap depth that we could detect
    return detfrac


def des_stream_min_depths(survey='LSST10', maglim=None):
    if maglim is None:
        maglim_g = mock_sim.getMagLimit('g', survey)
        maglim_r = mock_sim.getMagLimit('r', survey)
    else:
        maglim_g, maglim_r = [maglim] * 2

    depths = []
    for stream in STREAMS.keys():
        min_detectable_depth = min_gap_depth(STREAMS[stream]['distance'], STREAMS[stream]['surface_brightness'], STREAMS[stream]['width'], maglim_g, maglim_r, survey)
        depths.append(min_detectable_depth)
        print stream, min_detectable_depth
    output = np.vstack([STREAMS.keys(), depths]).T
    try:
        np.savetxt('min_depths_%s.txt' % survey, output, delimiter=', ', fmt='%s')
    except:
        print 'saving error'
    return depths

if __name__ == "__main__":
    depths = des_stream_min_depths()
