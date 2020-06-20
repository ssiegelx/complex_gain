#!/usr/bin/env
import os
import sys
import glob
import time
import datetime
import argparse
import random
import pytz
import json
import logging
import gc

import wtl.log as log
from wtl.namespace import NameSpace
from wtl.config import load_yaml_config

import scipy.interpolate
from scipy.constants import c as speed_of_light
import numpy as np
import h5py

from ch_util.fluxcat import FluxCatalog
from ch_util import ephemeris
from ch_util import andata
from ch_util import tools
from ch_util import cal_utils
from ch_util import timing
from ch_util import rfi

from complex_gain import sutil
from complex_gain.temps import TempData

###################################################
# default variables
###################################################

DEFAULTS = NameSpace(load_yaml_config(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                                   'defaults.yaml') + ':joint_regression'))

LOG_FILE = os.environ.get('JOINT_REGRESSION_LOG_FILE',
           os.path.join(os.path.dirname(os.path.realpath(__file__)), 'joint_regression.log'))

DEFAULT_LOGGING = {
    'formatters': {
         'std': {
             'format': "%(asctime)s %(levelname)s %(name)s: %(message)s",
             'datefmt': "%m/%d %H:%M:%S"},
          },
    'handlers': {
        'stderr': {'class': 'logging.StreamHandler', 'formatter': 'std', 'level': 'DEBUG'}
        },
    'loggers': {
        '': {'handlers': ['stderr'], 'level': 'INFO'}  # root logger

        }
    }

###################################################
# ancillary routines
###################################################

class StabilityData(andata.BaseData):

    @property
    def time(self):
        return self.index_map['time']

    @property
    def freq(self):
        return self.index_map['freq']


def _concatenate(xdist, xtemp, xcable, xtiming,
                 name_xdist=None, name_xtemp=None,
                 name_xcable=None, name_xtiming=None):

    combine = []
    name = []

    if xdist is not None:
        combine.append(xdist)
        if name_xdist is not None and len(name_xdist) == xdist.shape[-1]:
            name += list(name_xdist)
        else:
            name += ['dist%d' % dd for dd in range(xdist.shape[-1])]

    if xtemp is not None:
        combine.append(xtemp)
        if name_xtemp is not None and len(name_xtemp) == xtemp.shape[-1]:
            name += list(name_xtemp)
        else:
            name += ['temp%d' % tt for tt in range(xtemp.shape[-1])]

    if xcable is not None:
        combine.append(xcable)
        if name_xcable is not None and len(name_xcable) == xcable.shape[-1]:
            name += list(name_xcable)
        else:
            name += ['cable%d' % tt for tt in range(xcable.shape[-1])]

    if xtiming is not None:
        combine.append(xtiming)
        if name_xtiming is not None and len(name_xtiming) == xtiming.shape[-1]:
            name += list(name_xtiming)
        else:
            name += ['time%d' % tt for tt in range(xtiming.shape[-1])]

    if combine:
        x = np.concatenate(tuple(combine), axis=-1)
        return x, name
    else:
        raise RuntimeError("Did not provide any datasets to concatenate.")


class Timer(object):

    def __init__(self, logger, minutes=False):

        self.logger = logger
        self.msg = None
        self.t0 = time.time()

        self.minutes = minutes
        self.label = 'minutes' if minutes else 'seconds'

    def start(self, msg=None):

        self.msg = msg
        self.t0 = time.time()

        if msg is not None:
            self.logger.info("Start: %s" % msg)

    def stop(self, msg=None):

        if msg is None:
            msg = self.msg

        elapsed_time = time.time() - self.t0
        if self.minutes:
            elapsed_time /= 60.0

        self.logger.info("Finished: %s [took %0.2f %s]" % (msg, elapsed_time, self.label))

        self.msg = None
        self.t0 = time.time()


###################################################
# main routine
###################################################

def main(config_file=None, logging_params=DEFAULT_LOGGING):

    # Load config
    config = DEFAULTS.deepcopy()
    if config_file is not None:
        print(config_file)
        config.merge(NameSpace(load_yaml_config(config_file)))

    # Setup logging
    log.setup_logging(logging_params)
    logger = log.get_logger(__name__)

    timer = Timer(logger)

    # Load data
    sfile = config.data.filename if os.path.isabs(config.data.filename) else os.path.join(config.directory, config.data.filename)
    sdata = StabilityData.from_file(sfile)

    ninput, ntime = sdata['tau'].shape

    # Load temperature data
    tfile = (config.temperature.filename if os.path.isabs(config.temperature.filename) else
             os.path.join(config.directory, config.temperature.filename))

    tkeys = ['flag', 'data_flag', 'outlier']
    if config.temperature.load:
        tkeys += config.temperature.load

    tdata = TempData.from_acq_h5(tfile, datasets=tkeys)

    # Query layout database
    inputmap = tools.get_correlator_inputs(ephemeris.unix_to_datetime(np.median(sdata.time[:])), correlator='chime')

    good_input = np.flatnonzero(np.any(sdata['flags']['tau'][:], axis=-1))
    pol = sutil.get_pol(sdata, inputmap)
    npol = len(pol)

    mezz_index, crate_index = sutil.get_mezz_and_crate(sdata, inputmap)

    if config.mezz_ref.enable:
        phase_ref = [ipol[mezz_index[ipol] == iref] for ipol, iref in zip(pol, config.mezz_ref.mezz)]
    else:
        phase_ref = config.data.phase_ref

    # Load timing
    if config.timing.enable:

        # Extract filenames from config
        timing_files = [tf if os.path.isabs(tf) else os.path.join(config.directory, tf) for tf in config.timing.files]
        timing_files_hpf = [os.path.join(os.path.dirname(tf), 'hpf', os.path.basename(tf)) for tf in timing_files]
        timing_files_lpf = [os.path.join(os.path.dirname(tf), 'lpf', os.path.basename(tf)) for tf in timing_files]

        # If requested, add the timing data back into the delay data
        if config.timing.add.enable:

            timer.start("Adding timing data to delay measurements.")

            ns_tau, _, ns_flag, ns_inputs = sutil.get_timing_correction(sdata, timing_files, **config.timing.add.kwargs)

            index = timing.map_input_to_noise_source(sdata.index_map['input'], ns_inputs)

            timing_tau = ns_tau[index, :]
            timing_flag = ns_flag[index, :]
            for ipol, iref in zip(pol, config.data.phase_ref):
                timing_tau[ipol, :] = timing_tau[ipol, :] - timing_tau[iref, np.newaxis, :]
                timing_flag[ipol, :] = timing_flag[ipol, :] & timing_flag[iref, np.newaxis, :]

            sdata['tau'][:] = sdata['tau'][:] + timing_tau
            sdata['flags']['tau'][:] = sdata['flags']['tau'][:] & timing_flag

            timer.stop()

        # Extract the dependent variables from the timing dataset
        timer.start("Calculating timing dependence.")

        if config.timing.sep_delay:
            logger.info("Fitting HPF and LPF timing correction separately.")
            files = timing_files_hpf
            files2 = timing_files_lpf
        else:
            files2 = None
            if config.timing.hpf_delay:
                logger.info("Using HPF timing correction for delay.")
                files = timing_files_hpf
            elif config.timing.lpf_delay:
                logger.info("Using LPF timing correction for delay.")
                files = timing_files_lpf
            else:
                logger.info("Using full timing correction for delay.")
                files = timing_files

        kwargs = {}
        if config.timing.lpf_amp:
            logger.info("Using LPF timing correction for amplitude.")
            kwargs['afiles'] = timing_files_lpf
        elif config.timing.hpf_amp:
            logger.info("Using HPF timing correction for amplitude.")
            kwargs['afiles'] = timing_files_hpf
        else:
            logger.info("Using full timing correction for amplitude.")
            kwargs['afiles'] = timing_files

        for key in ['ns_ref', 'inter_cmn', 'fit_amp', 'ref_amp', 'cmn_amp']:
            if key in config.timing:
                kwargs[key] = config.timing[key]

        xtiming, xtiming_flag, xtiming_group = sutil.timing_dependence(sdata, files, inputmap, **kwargs)

        if files2 is not None:
            logger.info("Calculating second timing dependence.")
            kwargs['fit_amp'] = False
            xtiming2, xtiming2_flag, xtiming2_group = sutil.timing_dependence(sdata, files2, inputmap, **kwargs)

            xtiming = np.concatenate((xtiming, xtiming2), axis=-1)
            xtiming_flag = np.concatenate((xtiming_flag, xtiming2_flag), axis=-1)
            xtiming_group = np.concatenate((xtiming_group, xtiming2_group), axis=-1)

        timer.stop()

    else:
        xtiming = None
        xtiming_flag = None
        xtiming_group = None

    # Reference delay data to mezzanine
    if config.mezz_ref.enable:

        timer.start("Referencing delay measurements to mezzanine.")

        for ipol, iref in zip(pol, config.mezz_ref.mezz):

            this_mezz = ipol[mezz_index[ipol] == iref]

            wmezz = sdata['flags']['tau'][this_mezz, :].astype(np.float32)

            norm = np.sum(wmezz, axis=0)

            taut_mezz = np.sum(wmezz * sdata['tau'][this_mezz, :], axis=0) * tools.invert_no_zero(norm)
            flagt_mezz = norm > 0.0

            sdata['tau'][ipol, :] = sdata['tau'][ipol, :] - taut_mezz[np.newaxis, :]
            sdata['flags']['tau'][ipol, :] = sdata['flags']['tau'][ipol, :] & flagt_mezz[np.newaxis, :]

        timer.stop()

    # Load cable monitor
    if config.cable_monitor.enable:

        timer.start("Calculating cable monitor dependence.")

        cbl = timing.TimingCorrection.from_acq_h5(config.cable_monitor.filename)

        kwargs = {'include_diff': config.cable_monitor.include_diff}

        xcable, xcable_flag, xcable_group = sutil.cable_monitor_dependence(sdata, cbl, **kwargs)

        timer.stop()

    else:
        xcable = None
        xcable_flag = None
        xcable_group = None

    # Load NS distance
    if config.ns_distance.enable:

        timer.start("Calculating NS distance dependence.")

        kwargs = {}
        kwargs['phase_ref'] = phase_ref

        for key in ['sensor', 'temp_field', 'sep_cyl', 'include_offset', 'include_ha']:
            if key in config.ns_distance:
                kwargs[key] = config.ns_distance[key]

        if config.ns_distance.use_cable_monitor:
            kwargs['is_cable_monitor'] = True
            kwargs['use_alpha'] = config.ns_distance.use_alpha
            nsx = timing.TimingCorrection.from_acq_h5(config.cable_monitor.filename)
        else:
            kwargs['is_cable_monitor'] = False
            nsx = tdata

        xdist, xdist_flag, xdist_group = sutil.ns_distance_dependence(sdata, nsx, inputmap, **kwargs)

        if (config.ns_distance.deriv is not None) and (config.ns_distance.deriv > 0):

            for dd in range(1, config.ns_distance.deriv+1):

                d_xdist, d_xdist_flag, d_xdist_group = sutil.ns_distance_dependence(sdata, tdata, inputmap,
                                                                                    deriv=dd, **kwargs)

                tind = np.atleast_1d(1)
                xdist = np.concatenate((xdist, d_xdist[:, :, tind]), axis=-1)
                xdist_flag = xnp.concatenate((xdist_flag, d_xdist_flag[:, :, tind]), axis=-1)
                xdist_group = np.concatenate((xdist_group, d_xdist_group[:, tind]), axis=-1)

        timer.stop()

    else:
        xdist = None
        xdist_flag = None
        xdist_group = None

    # Load temperatures
    if config.temperature.enable:

        timer.start("Calculating temperature dependence.")

        xtemp, xtemp_flag, xtemp_group, xtemp_name = sutil.temperature_dependence(sdata, tdata, config.temperature.sensor,
                                                                                  field=config.temperature.temp_field,
                                                                                  inputmap=inputmap,
                                                                                  phase_ref=phase_ref,
                                                                                  check_hut=config.temperature.check_hut)

        if (config.temperature.deriv is not None) and (config.temperature.deriv > 0):

            for dd in range(1, config.temperature.deriv+1):

                d_xtemp, d_xtemp_flag, d_xtemp_group, d_xtemp_name = sutil.temperature_dependence(sdata, tdata, config.temperature.sensor,
                                                                                                  field=config.temperature.temp_field,
                                                                                                  deriv=dd, inputmap=inputmap,
                                                                                                  phase_ref=phase_ref,
                                                                                                  check_hut=config.temperature.check_hut)

                xtemp = np.concatenate((xtemp, d_xtemp), axis=-1)
                xtemp_flag = xnp.concatenate((xtemp_flag, d_xtemp_flag), axis=-1)
                xtemp_group = np.concatenate((xtemp_group, d_xtemp_group), axis=-1)
                xtemp_name += d_xtemp_name

        timer.stop()

    else:
        xtemp = None
        xtemp_flag = None
        xtemp_group = None

    # Combine into single feature matrix
    x, coeff_name = _concatenate(xdist, xtemp, xcable, xtiming, name_xtemp=xtemp_name)

    x_group, _ = _concatenate(xdist_group, xtemp_group, xcable_group, xtiming_group)

    x_flag, _ =  _concatenate(xdist_flag, xtemp_flag, xcable_flag, xtiming_flag)
    x_flag = np.all(x_flag, axis=-1) & sdata.flags['tau'][:]

    nfeature = x.shape[-1]

    logger.info("Fitting %d features." % nfeature)

    # Save data
    if config.preliminary_save.enable:

        if config.preliminary_save.filename is not None:
            ofile = (config.preliminary_save.filename if os.path.isabs(config.preliminary_save.filename) else
                     os.path.join(config.directory, config.preliminary_save.filename))
        else:
            ofile = os.path.splitext(sfile)[0] + '_%s.h5' % config.preliminary_save.suffix

        sdata.save(ofile, mode='w')

    # Subtract mean
    if config.mean_subtract:
        timer.start("Subtracting mean value.")

        tau, mu_tau, mu_tau_flag = sutil.mean_subtract(sdata, sdata['tau'][:], x_flag, use_calibrator=True)

        mu_x = np.zeros(mu_tau.shape + (nfeature,), dtype=x.dtype)
        mu_x_flag = np.zeros(mu_tau.shape + (nfeature,), dtype=np.bool)
        x_no_mu = x.copy()
        for ff in range(nfeature):
            x_no_mu[..., ff], mu_x[..., ff], mu_x_flag[..., ff] = sutil.mean_subtract(sdata, x[:, :, ff], x_flag,
                                                                                      use_calibrator=True)
        timer.stop()

    else:
        x_no_mu = x.copy()
        tau = sdata['tau'][:].copy()

    # Calculate unique days
    csd_uniq, bmap = np.unique(sdata['csd'][:], return_inverse=True)
    ncsd = csd_uniq.size

    # Prepare unique sources
    classification = np.char.add(np.char.add(sdata['calibrator'][:], '/'), sdata['source'][:])

    # If requested, set up boot strapping
    if config.bootstrap.enable:

        nboot = config.bootstrap.number
        nchoices = ncsd if config.bootstrap.by_transit else ntime
        nsample = int(config.bootstrap.fraction * nchoices)

        bindex = np.zeros((nboot, nsample), dtype=np.int)
        for roll in range(nboot):
            bindex[roll, :]= np.sort(np.random.choice(nchoices, size=nsample, replace=config.bootstrap.replace))

    else:

        nboot = 1
        bindex = np.arange(ntime, dtype=np.int)[np.newaxis, :]

    # Prepare output
    if config.output.directory is not None:
        output_dir = config.output.directory
    else:
        output_dir = config.data.directory

    if config.output.suffix is not None:
        output_suffix = config.output.suffix
    else:
        output_suffix = os.path.splitext(os.path.basename(config.data.filename))[0]

    # Perform joint fit
    for bb, bind in enumerate(bindex):

        if config.bootstrap.enable and config.bootstrap.by_transit:
            tind = np.concatenate(tuple([np.flatnonzero(bmap == ii) for ii in bind]))
        else:
            tind = bind

        ntime = tind.size

        if config.jackknife.enable:
            start = int(config.jackknife.start * ncsd) if config.jackknife.start <= 1.0 else config.jackknife.start
            end = int(config.jackknife.end * ncsd) if config.jackknife.end <= 1.0 else config.jackknife.end

            time_flag_fit = (bmap >= start) & (bmap < end)

            if config.jackknife.restrict_stat:
                time_flag_stat = np.logical_not(time_flag_fit)
            else:
                time_flag_stat = np.ones(ntime, dtype=np.bool)

        else:
            time_flag_fit = np.ones(ntime, dtype=np.bool)
            time_flag_stat = np.ones(ntime, dtype=np.bool)

        logger.info("Fitting data between %s (CSD %d) and %s (CSD %d)" %
                    (ephemeris.unix_to_datetime(np.min(sdata.time[tind[time_flag_fit]])).strftime("%Y-%m-%d"),
                     np.min(sdata['csd'][:][tind[time_flag_fit]]),
                     ephemeris.unix_to_datetime(np.max(sdata.time[tind[time_flag_fit]])).strftime("%Y-%m-%d"),
                     np.max(sdata['csd'][:][tind[time_flag_fit]])))

        logger.info("Calculating statistics from data between %s (CSD %d) and %s (CSD %d)" %
                    (ephemeris.unix_to_datetime(np.min(sdata.time[tind[time_flag_stat]])).strftime("%Y-%m-%d"),
                     np.min(sdata['csd'][:][tind[time_flag_stat]]),
                     ephemeris.unix_to_datetime(np.max(sdata.time[tind[time_flag_stat]])).strftime("%Y-%m-%d"),
                     np.max(sdata['csd'][:][tind[time_flag_stat]])))

        timer.start("Setting up fit.  Bootstrap %d of %d." % (bb+1, nboot))
        fitter = sutil.JointTempRegression(x_no_mu[:, tind, :], tau[:, tind], x_group, flag=x_flag[:, tind],
                                           classification=classification[tind],
                                           coeff_name=coeff_name)
        timer.stop()

        timer.start("Performing fit.  Bootstrap %d of %d." % (bb+1, nboot))
        fitter.fit_temp(time_flag=time_flag_fit, **config.fit_options)
        timer.stop()

        # If bootstrapping, append counter to filename
        if config.bootstrap.enable:
            output_suffix_bb = output_suffix + "_bootstrap_%04d" % (config.bootstrap.index_start + bb,)

            with open(os.path.join(output_dir, "bootstrap_index_%s.json" % output_suffix_bb), 'w') as jhandler:
                json.dump({"bind": bind.tolist(), "tind": tind.tolist()}, jhandler)

        else:
            output_suffix_bb = output_suffix

        # Save statistics to file
        if config.output.stat:

            # If requested, break the model up into its various components for calculating statistics
            stat_key = ['data', 'model', 'resid']
            if config.refine_model.enable:
                stat_add = fitter.refine_model(config.refine_model.include)
                stat_key += stat_add

            # Redefine axes
            bdata = StabilityData()
            for dset in ["source", "csd", "calibrator", "calibrator_time"]:
                bdata.create_dataset(dset, data=sdata[dset][tind])

            bdata.create_index_map("time", sdata.index_map["time"][tind])
            bdata.create_index_map("input", sdata.index_map["input"][:])
            bdata.attrs["calibrator"] = sdata.attrs.get("calibrator", "CYG_A")

            # Calculate statistics
            stat = {}
            for statistic in ['std', 'mad']:
                for attr in stat_key:
                    for ref, ref_common in zip(['mezz', 'cmn'], [False, True]):
                        stat[(statistic, attr, ref)] = sutil.short_long_stat(bdata, getattr(fitter, attr),
                                                                             fitter._flag & time_flag_stat[np.newaxis, :],
                                                                             stat=statistic, ref_common=ref_common, pol=pol)

            output_filename = os.path.join(output_dir, "stat_%s.h5" % output_suffix_bb)

            write_stat(bdata, stat, fitter, output_filename)

        # Save coefficients to file
        if config.output.coeff:
            output_filename = os.path.join(output_dir, "coeff_%s.h5" % output_suffix_bb)

            write_coeff(sdata, fitter, output_filename)

        # Save residuals to file
        if config.output.resid:
            output_filename = os.path.join(output_dir, "resid_%s.h5" % output_suffix_bb)

            write_resid(sdata, fitter, output_filename)

        del fitter
        gc.collect()


def write_stat(sdata, stat, fitter, output_filename):

    axis = {'time_diff': ['source'],
            'short': ['input'],
            'short_by_source': ['input', 'source'],
            'long': ['input'],
            'long_by_source': ['input', 'source'],
            'num_short': ['input'],
            'num_short_by_source': ['input', 'source'],
            'num_long': ['input'],
            'num_long_by_source': ['input', 'source'],
            'mu': ['input', 'csd', 'source'],
            'mu_flag': ['input', 'csd', 'source']}

    # Save to hdf5 file
    with h5py.File(output_filename, 'w') as handler:
        for key, val in stat.iteritems():
            grp = handler.create_group('_'.join(key))
            for kk, vv in val.iteritems():
                if kk in axis:
                    dset = grp.create_dataset(kk, data=vv)
                    dset.attrs['axis'] = np.array(axis[kk], dtype=np.string_)

            if 'index_map' not in handler:
                index_map = handler.create_group('index_map')

                index_map.create_dataset('source',
                                         data=np.array(val['source_pair'][:], dtype=np.string_))

                index_map.create_dataset('csd',
                                         data=val['csd'][:])

                inp_dtype = [('chan_id', '<u2'), ('correlator_input', '<S32')]
                index_map.create_dataset('input',
                                         data=np.array(val['input'][:], dtype=inp_dtype))

        dset = handler.create_dataset('number', data=fitter.number)
        dset.attrs['axis'] = np.array(['input'], dtype=np.string_)


def write_coeff(sdata, fitter, output_filename):

    inputs = np.array(sdata.index_map['input'][:],
                      dtype=[('chan_id', '<u2'), ('correlator_input', '<S32')])

    with h5py.File(output_filename, 'w') as handler:
        dset = handler.create_dataset('coeff', data=fitter.coeff)
        dset.attrs['axis'] = np.array(['input', 'feature'], dtype=np.string_)

        dset = handler.create_dataset('intercept', data=fitter.intercept)
        dset.attrs['axis'] = np.array(['input', 'classification'], dtype=np.string_)

        grp = handler.create_group('index_map')
        grp.create_dataset('feature', data=np.array(fitter.coeff_name, dtype=np.string_))
        grp.create_dataset('classification', data=np.array(fitter.intercept_name, dtype=np.string_))
        grp.create_dataset('input', data=inputs)


def write_resid(sdata, fitter, output_filename):

    odata = StabilityData()

    for key, val in sdata.attrs.iteritems():
        odata.attrs[key] = val

    for key, val in sdata.index_map.iteritems():
        odata.create_index_map(key, val)

    for key, val in sdata.datasets.iteritems():
        dset = odata.create_dataset(key, data=val[:])
        for kk, vv in val.attrs.iteritems():
            dset.attrs[kk] = vv

    for key, val in sdata.flags.iteritems():
        dset = odata.create_flag(key, data=val[:])
        for kk, vv in val.attrs.iteritems():
            dset.attrs[kk] = vv

    odata['tau'][:] = fitter.resid[:]
    odata.flags['tau'][:] = fitter._flag[:]

    odata.save(output_filename, mode='w')


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--config',   help='Name of configuration file.',
                                      type=str, default=None)
    parser.add_argument('--log',      help='Name of log file.',
                                      type=str, default=LOG_FILE)

    args = parser.parse_args()

    # If calling from the command line, then send logging to log file instead of screen
    logging_params = DEFAULT_LOGGING

    if args.log != 'stdout':
        try:
            os.makedirs(os.path.dirname(args.log))
        except OSError:
            if not os.path.isdir(os.path.dirname(args.log)):
                raise

        logging_params['handlers'] = {'stderr': {'class': 'logging.handlers.WatchedFileHandler',
                                                 'filename': args.log, 'formatter': 'std', 'level': 'INFO'}}

    # Call main routine
    main(config_file=args.config, logging_params=logging_params)