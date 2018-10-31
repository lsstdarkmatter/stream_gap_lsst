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

dpi = 175
fontsize = 15
# import pyplot and set some parameters to make plots prettier
plt.rc("savefig", dpi=dpi)
plt.rc('text', usetex=True)
plt.rc('font', size=fontsize)
plt.rc('xtick.major', pad=5)
plt.rc('xtick.minor', pad=5)
plt.rc('ytick.major', pad=5)
plt.rc('ytick.minor', pad=5)


def wdm_mass(mhalo, h=0.7):
    # input solar mass, return keV
    # return (mhalo * (h / 1e11))**-0.25 # 1512.05349
    return (mhalo / 5.5e10) ** (1 / -3.33)  # 1707.04256


def final_plot(filename, mus=[30., 31., 32., 33.], surveys=['SDSS', 'LSST10'], w=150., b=1., maglim=None, lat=60., gap_fill=True):
    output = np.genfromtxt('output.txt', unpack=True, delimiter=', ', dtype=None, names=['dist', 'w', 'b', 'maglim', 'lat', 'gap_fill', 'survey', 'mu', 'mass'], encoding='bytes')

    colors = ['cornflowerblue', 'seagreen', 'darkslateblue']
    plt.figure()

    for i, survey in enumerate(surveys):
        maglim = mock_sim.getMagLimit('g', survey)
        ret10 = []
        ret20 = []
        ret40 = []
        for mu in mus:
            idx = (output['w'] == w) & (output['b'] == b) & (output['maglim'] == np.around(
                maglim, 2)) & (output['survey'] == survey) & (output['gap_fill'] == gap_fill) & (output['mu'] == mu)

            idx10 = idx & (output['dist'] == 10)
            idx20 = idx & (output['dist'] == 20)
            idx40 = idx & (output['dist'] == 40)

            mass10 = output['mass'][idx10][0]
            mass20 = output['mass'][idx20][0]
            mass40 = output['mass'][idx40][0]
            ret10.append(mass10)
            ret20.append(mass20)
            ret40.append(mass40)

        label = r'$\mathrm{%s}$' % survey
        plt.semilogy(mus, ret20, 'o-', label=label, c=colors[i])  # label='d = %d, w = %d, b = %d' % (distance, w, b)
        plt.fill_between(mus, ret10, ret40, alpha=0.2, color=colors[i])

    ax1 = plt.gca()
    ax2 = ax1.twinx()
    mn, mx = ax1.get_ylim()
    ax2.set_ylim(mn, mx)
    ax2.set_yscale('log')

    plt.fill_between([29.5,33.5],[2.95,2.95],[mx,mx], facecolor= 'none', edgecolor='k', alpha = 0.7, hatch ='/') # MW satellite constraint
    plt.fill_between([29.5,33.5],[5.30,5.30],[mx,mx], facecolor= 'none', edgecolor='k', alpha = 0.7, hatch='\\') # Lyman alpha constraint

    ticks = ax1.get_yticks()
    wdm = wdm_mass(np.asarray(ticks))
    labels = [r'$%.2f$' % t for t in wdm]
    ax2.set_yticklabels(labels)

    ax2.set_ylabel(r'$m_{\mathrm{WDM}\ \mathrm{(keV)}}$')

    plt.legend(loc='upper left', fontsize=15)
    plt.title(r'$\mathrm{Minimum\ Detectable\ Halo\ Mass}$')
    plt.xlabel(r'$\mu\ \mathrm{(mag/arcsec^2)}$',)
    plt.ylabel(r'$M_{\mathrm{halo}}\ \mathrm{(M_{\odot})}$',)
    plt.tight_layout()
    plt.savefig('%s.png' % filename)


if __name__ == "__main__":
    plot_pretty()
    plot_gap_fill_times()
