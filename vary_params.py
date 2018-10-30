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


def plot_imact_parameter(mass=1e6, dist=20, maxt=0.5):
    mass /= 1e7
    rs = mock_sim.sss.rs(mass)

    scale_radii = np.linspace(rs / 4., rs * 4., 100)

    depths = []
    widths = []

    for r in scale_radii:
        gap_width, gap_depth = mock_sim.find_gap_size_depth(mass=mass, dist=dist, maxt=maxt, scale_radius=r)
        depths.append(gap_depth)
        widths.append(gap_width)

    # plt.figure()
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))

    ax[0].plot(scale_radii, widths, lw=2)
    ax[0].set_ylabel(r'$\mathrm{Gap\ Width\ (deg)}$')
    ax[0].set_xlabel(r'$r_s\ \mathrm{(kpc)}$')

    ax[1].plot(scale_radii, depths, lw=2)
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


def save_output(mus=[30.], distances=[20.], velocities=[150.], impact_parameters=[1.], maglims=[None], latitudes=[60.], surveys=['LSST'], gap_fill=True, **kwargs):
    if os.path.exists('output.txt'):
        pass
    else:
        output_file = open('output.txt', 'w')
        output_file.write('# distance (kpc), flyby_velocity (km/s), impact_parameter (r_s), magnitude_limit (mag), latitude (deg), gap_fill (True/False), survey, surface_brightness (mag/arcsec^2), minimum_mass (M_sun)\n')
        output_file.close()

    output = np.genfromtxt('output.txt', unpack=True, delimiter=', ', dtype=None, names=['dist', 'w', 'b', 'maglim', 'lat', 'gap_fill', 'survey', 'mu', 'mass'], encoding='bytes')

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
                                print mu, distance, w, b, maglim, lat, survey
                                # check if output already saved for these params
                                idx = (output['dist'] == distance) & (output['w'] == w) & (output['b'] == b) & (output['maglim'] == np.around(
                                    maglim, 2)) & (output['survey'] == survey) & (output['gap_fill'] == gap_fill) & (output['mu'] == mu)
                                if np.sum(idx) > 0:
                                    print 'Output exists'
                                    continue

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

                                with open('output.txt', 'a') as output_file:
                                    output_file.write('%.2f, %.2f, %.2f, %.2f, %.2f, %d, %s, %.2f, %.2f\n' % (distance, w, b, maglim, lat, gap_fill, survey, mu, ret[-1]))


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
                                print idx
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
                                    label += r'$\mathrm{ maglim =} %d$' % maglim
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


if __name__ == "__main__":
    plot_pretty()
    plot_gap_fill_times()
