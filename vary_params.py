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


def make_plot(filename, mus=[30.], distances=[20.], velocities=[150.], impact_parameters=[1.], maglims=[None], latitudes=[60.], gap_fill=True, **kwargs):
    plt.figure()
    for lat in latitudes:
        for maglim in maglims:
            for b in impact_parameters:
                for w in velocities:
                    for distance in distances:
                        ret = []
                        for mu in mus:
                            mockfile = '%d_deg_mock.fits' % lat
                            mass, gapt, gapo = mock_sim.predict_gap_depths(mu, distance, 'LSST', width_pc=20, maglim=maglim,
                                                                           timpact=0.5, gap_fill=gap_fill, w=w, X=b, mockfile=mockfile, **kwargs)
                            xind = np.isfinite(gapo / gapt)
                            II1 = scipy.interpolate.UnivariateSpline(
                                np.log10(mass)[xind], (gapo / gapt - 1)[xind], s=0)
                            R = scipy.optimize.root(II1, 6)
                            ret.append(10**R['x'])

                        try:
                            label = ''
                            if len(distances) > 1:
                                label += ' d = %d kpc' % distance
                            if len(velocities) > 1:
                                label += ' w = %d km/s' % w
                            if len(impact_parameters) > 1:
                                label += ' b =  %d rs' % b
                            if len(maglims) > 1:
                                if maglim == None:
                                    maglim = getMagLimit('g', 'LSST')
                                label += 'maglim = %d' % maglim
                            if len(latitudes) > 1:
                                label += 'lat = %d' % lat
                        except:
                            label = 'd=%d, w=%d, b=%d, mag=%d' % (distance, w, b, maglim)
                        plt.semilogy(mus, ret, 'o-', label=label)  # label='d = %d, w = %d, b = %d' % (distance, w, b)

    plt.legend()
    plt.title('Minimum Detectable halo mass from a single stream impact')
    plt.xlabel(r'$\mu$ [mag/sq.arcsec]',)
    plt.ylabel(r'$M_{halo}$ [M$_{\odot}$]',)
    plt.savefig(filename)


if __name__ == "__main__":
    plot_pretty()
    plot_gap_fill_times()
