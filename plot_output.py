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


def plot_pretty(dpi=175, fontsize=15):
    # import pyplot and set some parameters to make plots prettier
    plt.rc("savefig", dpi=dpi)
    plt.rc('text', usetex=True)
    plt.rc('font', size=fontsize)
    plt.rc('xtick.major', pad=5)
    plt.rc('xtick.minor', pad=5)
    plt.rc('ytick.major', pad=5)
    plt.rc('ytick.minor', pad=5)
    return


def plot_output(filename, mus=[30.], distances=[20.], velocities=[150.], impact_parameters=[1.], maglims=[None], latitudes=[60.], surveys=['LSST'], gap_fill=True):
    output = np.genfromtxt('output.txt', unpack=True, delimiter=', ', dtype=None, names=['dist', 'w', 'b', 'maglim', 'lat', 'gap_fill', 'survey', 'mu', 'mass'], encoding='bytes')

    plt.figure()
    for survey in surveys:
        for lat in latitudes:
            for maglim in maglims:
                for b in impact_parameters:
                    for w in velocities:
                        for distance in distances:

                            ret = []
                            if maglim == None:
                                maglim = mock_sim.getMagLimit('g', survey)

                            for mu in mus:
                                idx = (output['dist'] == distance) & (output['w'] == w) & (output['b'] == b) & (output['maglim'] == np.around(
                                    maglim, 2)) & (output['survey'] == survey) & (output['gap_fill'] == gap_fill) & (output['mu'] == mu)
                                # print idx
                                mass = output['mass'][idx][0]
                                ret.append(mass)

                            try:
                                label = ''
                                if len(distances) > 1:
                                    label += r'$ d = %d \mathrm{kpc}$' % distance
                                if len(velocities) > 1:
                                    label += r'$ w = %d \mathrm{km/s}$' % w
                                if len(impact_parameters) > 1:
                                    label += r'$ b =  %d \mathrm{r_s}$' % b
                                if len(maglims) > 1:
                                    label += r'$\mathrm{ maglim =} %.1$' % maglim
                                if len(latitudes) > 1:
                                    label += r'$\mathrm{ lat = %d}$' % lat
                                if len(surveys) > 1:
                                    label += r'$\mathrm{ %s}$' % survey
                                if label == '':
                                    label = r'$\mathrm{d=%d, w=%d, b=%d, mag=%d, lat=%d, %s}$' % (distance, w, b, maglim, lat, survey)
                            except:
                                label = r'$\mathrm{d=%d, w=%d, b=%d, mag=%d, lat=%d, %s}$' % (distance, w, b, maglim, lat, survey)
                            plt.semilogy(mus, ret, 'o-', label=label)  # label='d = %d, w = %d, b = %d' % (distance, w, b)

    plt.legend(loc='upper left', fontsize=10)
    plt.title(r'$\mathrm{Minimum\ Detectable\ Halo\ Mass}$')
    plt.xlabel(r'$\mu\ \mathrm{(mag/arcsec^2)}$',)
    plt.ylabel(r'$M_{\mathrm{halo}}\ \mathrm{(M_{\odot})}$',)
    plt.tight_layout()
    plt.savefig('%s.png' % filename)


def plot_output_distance(filename, mus=[30.], distances=[20.], velocities=[150.], impact_parameters=[1.], maglims=[None], latitudes=[60.], surveys=['LSST'], gap_fill=True):
    output = np.genfromtxt('output.txt', unpack=True, delimiter=', ', dtype=None, names=['dist', 'w', 'b', 'maglim', 'lat', 'gap_fill', 'survey', 'mu', 'mass'], encoding='bytes')

    plt.figure()
    for survey in surveys:
        for lat in latitudes:
            for maglim in maglims:
                for b in impact_parameters:
                    for w in velocities:
                        for mu in mus:

                            ret = []
                            if maglim == None:
                                maglim = mock_sim.getMagLimit('g', survey)

                            for distance in distances:
                                idx = (output['dist'] == distance) & (output['w'] == w) & (output['b'] == b) & (output['maglim'] == np.around(
                                    maglim, 2)) & (output['survey'] == survey) & (output['gap_fill'] == gap_fill) & (output['mu'] == mu)
                                # print idx
                                mass = output['mass'][idx][0]
                                ret.append(mass)

                            try:
                                label = ''
                                if len(mus) > 1:
                                    label += r'$ \mu = %d$' % mu  # \ \mathrm{mag/arcsec^2}
                                if len(velocities) > 1:
                                    label += r'$ w = %d \mathrm{km/s}$' % w
                                if len(impact_parameters) > 1:
                                    label += r'$ b =  %d \mathrm{r_s}$' % b
                                if len(maglims) > 1:
                                    label += r'$\mathrm{ maglim =} %d$' % maglim
                                if len(latitudes) > 1:
                                    label += r'$\mathrm{ lat = %d}$' % lat
                                if len(surveys) > 1:
                                    label += r'$\mathrm{ %s}$' % survey
                                if label == '':
                                    label = r'$\mathrm{mu=%d, w=%d, b=%d, mag=%d, lat=%d, %s}$' % (mu, w, b, maglim, lat, survey)
                            except:
                                label = r'$\mathrm{d=%d, w=%d, b=%d, mag=%d, lat=%d, %s}$' % (mu, w, b, maglim, lat, survey)
                            plt.semilogy(distances, ret, 'o-', label=label)  # label='d = %d, w = %d, b = %d' % (distance, w, b)

    plt.legend(loc='upper left', fontsize=10)
    plt.title(r'$\mathrm{Minimum\ Detectable\ Halo\ Mass}$')
    plt.xlabel(r'$\mathrm{Distance}\ \mathrm{(kpc)}$',)
    plt.ylabel(r'$M_{\mathrm{halo}}\ \mathrm{(M_{\odot})}$',)
    plt.tight_layout()
    plt.savefig('%s.png' % filename)

if __name__ == "__main__":
    plot_pretty()
    plot_output(filename='sample_plot', mus=[30, 31, 32, 33], surveys=['SDSS', 'LSST10'])
