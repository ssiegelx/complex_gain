#!/usr/bin/python
import os
import glob
import datetime
import argparse

import numpy as np
import stability

# Set up plotting
# ---------------
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm

from general_tools import OutputPlot

fontsize = 16
plt.rc('font', family='sans-serif', weight='normal', size=fontsize)

COLORMAP = matplotlib.cm.__dict__["inferno"]
COLORMAP.set_under("black")
COLORMAP.set_bad("lightgray")


# Auxiliary functions
# -------------------

def plot_versus_input(data, flag, color='b', ylabel='RMS [rad]', yrng=None,
                      label=None, title=None, no_extra=False, no_fill=False):
    
    if label is None:
        label = ''
    
    if data.ndim > 1:
        nfreq, ninput = data.shape
        inp = np.arange(ninput)
        perc = np.nanpercentile(np.where(flag, data, np.nan), [2.5, 16.0, 50.0, 84.0, 97.5], axis=0)
        if not no_fill:
            plt.fill_between(inp, perc[0, :], perc[4, :], color='gray', alpha=1.0)
            plt.fill_between(inp, perc[1, :], perc[3, :], color='darkgray', alpha=0.5)

        trace = perc[2, :]
    else:
        ninput = data.size
        inp = np.arange(ninput)
        trace = np.where(flag, data, np.nan)

    plt.plot(inp, trace, color=color, linestyle='-', linewidth=1.0, label=label)

    if not no_extra:

        plt.xlabel("Feed Number")
        plt.ylabel(ylabel)
        plt.xlim(0, ninput)
        plt.xticks(np.arange(ninput // 256 + 1) * 256)
        if yrng is not None:
            plt.ylim(yrng)
        plt.grid()
        if title is not None:
            plt.title(title)


def plot_versus_freq(data, flag, freq=None, color='b', ylabel='RMS [rad]', yrng=None,
                     label=None, title=None, no_extra=False, no_fill=False):

    if freq is None:
        freq = np.linspace(800.0, 400.0, 1024, endpoint=False)

    if label is None:
        label = ''
    
    if data.ndim > 1:
        nfreq, ninput = data.shape
        perc = np.nanpercentile(np.where(flag, data, np.nan), [2.5, 16.0, 50.0, 84.0, 97.5], axis=1)
        if not no_fill:
            plt.fill_between(freq, perc[0, :], perc[4, :], color='gray', alpha=1.0)
            plt.fill_between(freq, perc[1, :], perc[3, :], color='darkgray', alpha=0.5)

        trace = perc[2, :]
    else:
        ninput = data.size
        trace = np.where(flag, data, np.nan)

    plt.plot(freq, trace, color=color, linestyle='-', linewidth=1.0, label=label)

    if not no_extra:

        plt.xlabel("Frequency [MHz]")
        plt.ylabel(ylabel)
        plt.xlim(freq[-1], freq[0])
        if yrng is not None:
            plt.ylim(yrng)
        plt.grid()
        if title is not None:
            plt.title(title)


# Main functions
# --------------

def plot_stats(input_file, output_file=None, plot_dir=None,
               amp_range=None, phi_range=None, tau_range=None):

    if output_file is None:
        output_dir = os.path.dirname(input_file)
        if plot_dir is not None:
            output_dir = os.path.join(output_dir, plot_dir)

        output_file = os.path.join(output_dir, os.path.splitext(os.path.basename(input_file))[0] + '.pdf')

    keys = [('std_amp', 'stat', amp_range, r'$\sigma( \delta a )$', 'r'), 
            ('mad_amp', 'stat', amp_range, r'$MAD( \delta a )$', 'm'),
            ('std_phi', 'stat', phi_range, r'$\sigma( \delta \phi )$ [rad]', 'b'),
            ('std_resid_phi', 'stat', phi_range, r'$\sigma( \delta \phi - \omega \delta \tau )$ [rad]', 'b'),
            ('std_tau', 'tau', tau_range, r'$\sigma( \delta \tau )$', 'b'),
            ('mad_phi', 'stat', phi_range, r'$MAD( \delta \phi )$ [rad]', 'c'),
            ('mad_resid_phi', 'stat', phi_range, r'$MAD( \delta \phi - \omega \delta \tau )$ [rad]', 'c'),
            ('mad_tau', 'tau', tau_range, r'$MAD( \delta \tau )$', 'c')]

    dsets = [key[0] for key in keys]
    flags = ['flags/%s' % key[1] for key in keys]

    data = stability.StabilityData.from_acq_h5(input_file, datasets=dsets + flags + ['timestamp'])

    timescale, counts = np.unique(np.round(np.abs(np.diff(data.datasets['timestamp'], axis=0)) / 360.0),
                                  return_counts=True)
    ttl = r"$\Delta t$ = %s" % ', '.join(["%0.1f hr (%d)" % ts for ts in zip(timescale / 10.0, counts)])


    out = OutputPlot(output_file, dpi=400)

    for (dkey, fkey, yrng, ylbl, clr) in keys:

        fig = plt.figure(num=1, figsize=(20, 10), dpi=400)
        
        val = data.datasets[dkey][:]
        flag = data.flags[fkey][:]
        
        if (flag.ndim - val.ndim) > 0:
            flag = np.any(flag, axis=-1)

        plot_versus_input(val, flag, title=ttl, ylabel=ylbl, yrng=yrng, color=clr)
        
        out.save()
        plt.close(fig)

    out.close()


def plot_stats_by_pair(input_file, output_file=None, plot_dir=None,
                       amp_range=None, phi_range=None, tau_range=None):

    if output_file is None:
        output_dir = os.path.dirname(input_file)
        if plot_dir is not None:
            output_dir = os.path.join(output_dir, plot_dir)

        output_file = os.path.join(output_dir, os.path.splitext(os.path.basename(input_file))[0] + '_by_pair.pdf')

    keys = [('std_amp', 'stat', amp_range, r'$\sigma( \delta a )$'), 
            ('mad_amp', 'stat', amp_range, r'$MAD( \delta a )$'),
            ('std_phi', 'stat', phi_range, r'$\sigma( \delta \phi )$ [rad]'),
            ('std_resid_phi', 'stat', phi_range, r'$\sigma( \delta \phi - \omega \delta \tau )$ [rad]'),
            ('std_tau', 'tau_stat', tau_range, r'$\sigma( \delta \tau )$'),
            ('mad_phi', 'stat', phi_range, r'$MAD( \delta \phi )$ [rad]'),
            ('mad_resid_phi', 'stat', phi_range, r'$MAD( \delta \phi - \omega \delta \tau )$ [rad]'),
            ('mad_tau', 'tau_stat', tau_range, r'$MAD( \delta \tau )$')]

    keys = [(key[0] + '_by_pair', key[1] + '_by_pair') + key[2:] for key in keys]

    dsets = [key[0] for key in keys]
    flags = ['flags/%s' % key[1] for key in keys]

    data = stability.StabilityData.from_acq_h5(input_file, datasets=dsets + flags + ['timestamp', 'pair_map'])

    label = []
    counts = []
    for dd, name in enumerate(data.index_map['pair'][:]):
        print data.datasets['pair_map'][:].shape
        this_pair = np.flatnonzero(data.datasets['pair_map'][:] == dd)
        timescale = np.median(np.round(np.abs(np.diff(data.datasets['timestamp'][:, this_pair], axis=0)) / 360.0)) / 10.0
        label.append('%s - %0.1f hr - %d Transits' % (name, timescale, this_pair.size))
        counts.append(this_pair.size)

    order = np.argsort(counts)[::-1]
    color = ['b', 'r', 'forestgreen']

    out = OutputPlot(output_file, dpi=400)

    for (dkey, fkey, yrng, ylbl) in keys:

        fig = plt.figure(num=1, figsize=(20, 10), dpi=400)
        
        val = data.datasets[dkey][:]
        flag = data.flags[fkey][:]
        
        print dkey, val.shape, fkey, flag.shape, order

        for ii, dd in enumerate(order):

            plot_versus_input(val[..., dd], flag[..., dd], label=label[dd],
                              ylabel=ylbl, yrng=yrng, color=color[ii],
                              no_extra=(ii > 0), no_fill=(ii > 0))
        
        plt.legend()
        out.save()
        plt.close(fig)

    out.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('filename',   help='Name of the file containing the stability analysis.', type=str)

    parser.add_argument('--pdf',   help='Name of the output pdf file.',
                                   type=str, default=None)

    parser.add_argument('--pdf_dir',   help='Name of the directory for output pdf files.',
                                       type=str, default=None)

    parser.add_argument('--by_pair',   help='Plot each pair separately.',
                                       action='store_true')

    parser.add_argument('--amp_range',   help='Range for amplitude data.',
                                         type=float, nargs='+', default=None)

    parser.add_argument('--phi_range',   help='Range for phase data.',
                                         type=float, nargs='+', default=None)

    parser.add_argument('--tau_range',   help='Range for delay data.',
                                         type=float, nargs='+', default=None)

    args = parser.parse_args()

    # Call main routine
    method = plot_stats_by_pair if args.by_pair else plot_stats
    
    method(args.filename, output_file=args.pdf, plot_dir=args.pdf_dir,
           amp_range=args.amp_range, phi_range=args.phi_range, tau_range=args.tau_range)

