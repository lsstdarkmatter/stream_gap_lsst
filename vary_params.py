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


def plot_scale_radius(mass=1e6, dist=20, maxt=0.5):
    mass /= 1e7
    rs = mock_sim.sss.rs(mass)

    scale_radii = rs * np.linspace(0.25, 4., 100)

    depths = []
    widths = []

    for r in scale_radii:
        gap_width, gap_depth = mock_sim.find_gap_size_depth(mass=mass, dist=dist, maxt=maxt, scale_radius=r)
        depths.append(gap_depth)
        widths.append(gap_width)

    # plt.figure()
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))

    ax[0].plot(scale_radii / rs, widths, lw=2)
    ax[0].set_ylabel(r'$\mathrm{Gap\ Width\ (deg)}$')
    ax[0].set_xlabel(r'$r_s\ \mathrm{(kpc)}$')

    ax[1].plot(scale_radii / rs, depths, lw=2)
    ax[1].set_ylabel(r'$\mathrm{Gap\ Depth}$')
    ax[1].set_xlabel(r'$r_s\ \mathrm{(kpc)}$')

    plt.tight_layout()
    plt.savefig('scale_radius.png')


def plot_gap_fill_times(dist=20):
    mass_array = 10**np.linspace(3, 7, 100) / 1e7
    times = []
    for m in mass_array:
        fill_time = mock_sim.find_gap_fill_time(mass=m, dist=dist)
        times.append(fill_time)

    plt.figure()
    plt.plot(mass_array * 1e7, times, lw=2)

    plt.axhline(0.5, ls='--', c='forestgreen', lw=2)

    plt.ylabel(r'$\mathrm{Fill\ Time\ (Gyr)}$')
    plt.xlabel(r'$\mathrm{Mass\ (Msun)}$')
    plt.xscale('log')

    plt.tight_layout()
    plt.savefig('fill_time.png')


def plot_gap_sizes(dist=20, maxt=0.5):
    mass_array = 10**np.linspace(3, 7, 50) / 1e7
    depths = []
    widths = []
    for m in mass_array:
        gap_width, gap_depth = mock_sim.find_gap_size_depth(mass=m, dist=dist, maxt=maxt)
        depths.append(gap_depth)
        widths.append(gap_width)

    # plt.figure()
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))

    ax[0].plot(mass_array * 1e7, widths, lw=2)
    ax[0].set_ylabel(r'$\mathrm{Gap\ Width\ (deg)}$')
    ax[0].set_xlabel(r'$\mathrm{Mass\ (Msun)}$')

    ax[1].plot(mass_array * 1e7, depths, lw=2)
    ax[1].set_ylabel(r'$\mathrm{Gap\ Depth}$')
    ax[1].set_xlabel(r'$\mathrm{Mass\ (Msun)}$')

    plt.tight_layout()
    plt.savefig('gap_width_depth.png')


def plot_imact_parameter(mass=1e6, dist=20, maxt=0.5):
    mass /= 1e7
    impact_params = np.linspace(0.1, 10.0, 100)  # units of subhalo scale radius

    depths = []
    widths = []

    for b in impact_params:
        gap_width, gap_depth = mock_sim.find_gap_size_depth(mass=mass, dist=dist, maxt=maxt, X=b)
        depths.append(gap_depth)
        widths.append(gap_width)

    # plt.figure()
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))

    ax[0].plot(impact_params, widths, lw=2)
    ax[0].set_ylabel(r'$\mathrm{Gap\ Width\ (deg)}$')
    ax[0].set_xlabel(r'$b\ \mathrm{(r_s)}$')

    ax[1].plot(impact_params, depths, lw=2)
    ax[1].set_ylabel(r'$\mathrm{Gap\ Depth}$')
    ax[1].set_xlabel(r'$b\ \mathrm{(r_s)}$')

    plt.tight_layout()
    plt.savefig('impact_parameter.png')


def plot_flyby_velocity(mass=1e6, dist=20, maxt=0.5):
    mass /= 1e7
    vels = np.linspace(10, 400, 100)  # km/s

    depths = []
    widths = []

    for w in vels:
        gap_width, gap_depth = mock_sim.find_gap_size_depth(mass=mass, dist=dist, maxt=maxt, w=w)
        depths.append(gap_depth)
        widths.append(gap_width)

    # plt.figure()
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))

    ax[0].plot(vels, widths, lw=2)
    ax[0].set_ylabel(r'$\mathrm{Gap\ Width\ (deg)}$')
    ax[0].set_xlabel(r'$v_{\mathrm{flyby}}\ \mathrm{(km/s)}$')

    ax[1].plot(vels, depths, lw=2)
    ax[1].set_ylabel(r'$\mathrm{Gap\ Depth}$')
    ax[1].set_xlabel(r'$v_{\mathrm{flyby}}\ \mathrm{(km/s)}$')

    plt.tight_layout()
    plt.savefig('flyby_velocity.png')


if __name__ == "__main__":
    plot_pretty()
    plot_gap_fill_times()
