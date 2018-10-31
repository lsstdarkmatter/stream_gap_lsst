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


def halo_mass(mwdm, h=0.7):
    return 5.5e10 * (mwdm)**(-3.33)


def final_plot(filename=None, mus=[30., 31., 32., 33.], surveys=['SDSS', 'LSST10'], w=150., b=1., maglim=None, lat=60., gap_fill=True):
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

    plt.legend(loc='upper left', fontsize=15)
    plt.ylabel(r'$M_{\mathrm{halo}}\ \mathrm{(M_{\odot})}$',)
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    mn, mx = ax1.get_ylim()
    ax2.set_ylim(mn, mx)
    ax2.set_yscale('log')

    ticks = ax1.get_yticks()
    wdm = wdm_mass(np.asarray(ticks))
    labels = [r'$%.2f$' % t for t in wdm]
    ax2.set_yticklabels(labels)
    ax2.set_ylabel(r'$m_{\mathrm{WDM}\ \mathrm{(keV)}}$')
    mn2, mx2 = ax2.get_ylim()
    # ax2.fill_between([29.5,33.5],[halo_mass(2.95),halo_mass(2.95)],[mx,mx], facecolor= 'none', edgecolor='k', alpha = 0.3, hatch ='/') # MW satellite constraint
    # ax2.fill_between([29.5,33.5],[halo_mass(5.30),halo_mass(5.30)],[mx,mx], facecolor= 'none', edgecolor='k', alpha = 0.3, hatch='\\') # Lyman alpha constraint
    plt.plot([29.5, 33.5], [halo_mass(2.95), halo_mass(2.95)], c='0.5', lw=2, linestyle='-')#, label=r'$\mathrm{MW\ satellites}$')
    plt.plot([29.5, 33.5], [halo_mass(5.30), halo_mass(5.30)], c='0.5', lw=2, linestyle='-')#, label=r'$\mathrm{Lyman}\ \alpha$')
    plt.text(30.45,halo_mass(2.95),r'$\mathrm{MW\ satellites}$', horizontalalignment='center', verticalalignment='center', size=10,bbox=dict(facecolor='white', alpha=1, edgecolor=None))
    plt.text(30.45,halo_mass(5.30),r'$\mathrm{Lyman}\ \alpha$', horizontalalignment='center', verticalalignment='center', size=10,bbox=dict(facecolor='white', alpha=1, edgecolor=None))
 #   plt.plot([31.9,31.9], [mn2,mx2], c='0.5', lw=2, linestyle='--')#, label=r'$\mathrm{MW\ satellites}$')
 #   plt.plot([33.0,33.0], [mn2,mx2], c='0.5', lw=2, linestyle='--')#, label=r'$\mathrm{Lyman}\ \alpha$')
    plt.text(31.9,3.6e5,r'$\mathrm{Indus}$',rotation=90., horizontalalignment='center', verticalalignment='center', size=10,bbox=dict(facecolor='white', alpha=1, edgecolor=None))
    plt.text(33.0,5e5,r'$\mathrm{ATLAS}$',rotation=90., horizontalalignment='center', verticalalignment='center', size=10,bbox=dict(facecolor='white', alpha=1, edgecolor=None))


    plt.xlim(29.9, 33.1)
    plt.title(r'$\mathrm{Minimum\ Detectable\ Halo\ Mass}$')
    plt.xlabel(r'$\mu\ \mathrm{(mag/arcsec^2)}$',)
    plt.tight_layout()
    if filename is not None:
        plt.savefig('%s.png' % filename)
    plt.show()

if __name__ == "__main__":
    final_plot()
