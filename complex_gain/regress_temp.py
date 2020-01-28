import os
import glob
import datetime
import argparse

import scipy
import numpy as np
import h5py
from sklearn.linear_model import HuberRegressor

import wtl.log as log
from wtl.namespace import NameSpace
from wtl.config import load_yaml_config

from ch_util.fluxcat import FluxCatalog
from ch_util import ephemeris

from temps import TempData
from stability import StabilityData

###################################################
# default variables
###################################################

DEFAULTS = NameSpace(load_yaml_config(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                                   'defaults.yaml') + ':regress_temp'))

LOG_FILE = os.environ.get('REGRESS_TEMP_LOG_FILE',
           os.path.join(os.path.dirname(os.path.realpath(__file__)), 'regress_temp.log'))

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
# class for performing regression
###################################################

class TempRegression(object):

    def __init__(self, x, data, flag=None):

        self.log = log.get_logger(self)

        if flag is None:
            fshp = tuple([1] * (data.ndim - 1) + [data.shape[-1]])
            flag = np.ones(fshp, dtype=np.bool)

        self._flag = flag

        if x.ndim == 1:
            x = x[:, np.newaxis]

        if x.ndim != (data.ndim+1):
            expand = tuple([None] * (data.ndim+1 - x.ndim) + [slice(None)] * x.ndim)
            x = x[expand]

        self._x = x
        self.data = data

        self.N = data.shape[:-1]
        self.nsample = data.shape[-1]
        self.nfeature = x.shape[-1]

    def process(self):

        self.fit_temp()
        self.stats()

    def _resolve_index(self, ind, shp):
        return tuple([ii % ss for ii, ss in zip(ind, shp[0:len(ind)])])

    def flag(self, ind):
        return self._flag[self._resolve_index(ind, self._flag.shape)]

    def x(self, ind):
        return self._x[self._resolve_index(ind, self._x.shape)]

    def fit_data(self, ind):
        flag = np.flatnonzero(self.flag(ind))
        return self.data[ind][flag]

    def fit_x(self, ind):
        flag = np.flatnonzero(self.flag(ind))
        return self.x(ind)[flag, :]

    def count(self, ind):
        return np.sum(self.flag(ind))

    def fit_temp(self):

        self.intercept = np.zeros(self.N, dtype=np.float32)
        self.number = np.zeros(self.N, dtype=np.int)
        self.coeff = np.zeros(self.N + (self.nfeature,), dtype=np.float32)

        for counter, ii in enumerate(np.ndindex(*self.N)):

            if not (counter % 1000):
                self.log.info("Fitting index %d of %d." % (counter, np.prod(self.N)))

            nsample = self.count(ii)
            if nsample > (self.nfeature+1):

                try:
                    huber = HuberRegressor(fit_intercept=True).fit(self.fit_x(ii), self.fit_data(ii))

                except Exception as exc:
                    self.log.info("Fit failed for index %s:  %s" % (str(ii), exc))

                else:
                    self.intercept[ii] = huber.intercept_
                    self.coeff[ii] = huber.coef_[:]
                    self.number[ii] = nsample

        self.model = self.intercept[..., np.newaxis] + np.sum(self.coeff[..., np.newaxis, :] * self._x, axis=-1)
        self.resid = self.data - self.model

    def stats(self):

        for key in ['data', 'model', 'resid']:

            ts = np.where(self._flag, getattr(self, key), np.nan)

            mad = 1.46825 * np.nanmedian(np.abs(ts - np.nanmedian(ts, axis=-1, keepdims=True)), axis=-1)
            mad[~np.isfinite(mad)] = 0.0

            sigma = np.nanstd(ts, axis=-1)
            sigma[~np.isfinite(sigma)] = 0.0

            setattr(self, 'mad_' + key, mad)
            setattr(self, 'std_' + key, sigma)


###################################################
# main routine
###################################################

def main(filename, config_file=None, logging_params=DEFAULT_LOGGING):

    # Load config
    config = DEFAULTS.deepcopy()
    if config_file is not None:
        config.merge(NameSpace(load_yaml_config(config_file)))

    # Setup logging
    log.setup_logging(logging_params)
    logger = log.get_logger(__name__)

    # Load the data
    dsets = config.datasets + ['flags/%s' % name for name in config.flags] + ['timestamp', 'pair_map']

    logger.info("Requesting datasets: %s" % str(dsets))

    data = StabilityData.from_acq_h5(filename, datasets=dsets)

    logger.info("Loaded datasets: %s" % str(data.datasets.keys()))

    # Load the temperatures
    tdata = TempData.from_acq_h5(config.temp_filename)

    # Interpolate requested temperatures to time of transits
    if config.sensors is not None:
        index = np.sort(np.concatenate(tuple([tdata.search_sensors(name) for name in config.sensors])))
    else:
        index = slice(None)

    temp = tdata.datasets[config.temp_field][index]
    temp_func = scipy.interpolate.interp1d(tdata.time, temp, axis=-1, **config.interp)

    itemp = temp_func(data.datasets['timestamp'][:])

    # Difference temperatures between two transits
    feature = []
    dtemp = np.zeros((itemp.shape[-1], itemp.shape[0]), dtype=itemp.dtype)
    for isensor, stemp in enumerate(itemp):

        if config.is_ns_dist and config.is_ns_dist[isensor]:

            feature.append(tdata.sensor[index[isensor]] + '_ns_dist')

            coeff = np.array([[np.sin(np.radians(FluxCatalog[ss].dec - ephemeris.CHIMELATITUDE))
                               for ss in pair.decode("UTF-8").split('/')]
                               for pair in data.index_map['pair'][data.datasets['pair_map']]]).T

            dtemp[:, isensor] = coeff[0] * stemp[0] - coeff[1] * stemp[1]

        else:
            feature.append(tdata.sensor[index[isensor]])
            dtemp[:, isensor] = stemp[0] - stemp[1]

    # Generate flags for the temperature data
    flag_func = scipy.interpolate.interp1d(tdata.time, tdata.datasets['flag'][index].astype(np.float32),
                                           axis=-1, **config.interp)

    dtemp_flag = np.all(flag_func(data.datasets['timestamp'][:]) == 1.0, axis=(0, 1))

    # Add temperature information to data object
    data.create_index_map('feature', np.array(feature, dtype=np.string_))

    dset = data.create_dataset("temp", data=dtemp)
    dset.attrs['axis'] = np.array(['time', 'feature'], dtype=np.string_)

    dset = data.create_flag("temp", data=dtemp_flag)
    dset.attrs['axis'] = np.array(['time', 'feature'], dtype=np.string_)

    # Perform the fit
    for dkey, fkey in zip(config.datasets, config.flags):

        logger.info("Now fitting %s using %s flags" % (dkey, fkey))

        this_data = data.datasets[dkey][:]
        expand = tuple([None] * (this_data.ndim - 1) + [slice(None)])
        this_flag = data.flags[fkey][:] & dtemp_flag[expand]

        fitter = TempRegression(dtemp, this_data, flag=this_flag)
        fitter.process()

        # Save results
        for out in ['model', 'resid']:
            dset = data.create_dataset('_'.join([config.prefix, out, dkey]), data=getattr(fitter, out))
            dset.attrs['axis'] = data.datasets[dkey].attrs['axis'].copy()

            for stat in ['mad', 'std']:
                dset = data.create_dataset('_'.join([stat, config.prefix, out, dkey]),
                                           data=getattr(fitter, '_'.join([stat, out])))
                dset.attrs['axis'] = data.datasets[dkey].attrs['axis'][:-1].copy()

        for out in ['intercept', 'number']:
            dset = data.create_dataset('_'.join([config.prefix, out, dkey]), data=getattr(fitter, out))
            dset.attrs['axis'] = data.datasets[dkey].attrs['axis'][:-1].copy()

        dset = data.create_dataset('_'.join([config.prefix, 'coeff', dkey]), data=fitter.coeff)
        dset.attrs['axis'] = np.array(list(data.datasets[dkey].attrs['axis'][:-1]) + ['feature'], dtype=np.string_)


    # Save the results to disk
    output_filename = os.path.splitext(filename)[0] + '_' + config.output_suffix + '.h5'

    data.save(output_filename)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('filename',   help='Name of the file containing the stability analysis.', type=str)

    parser.add_argument('--config',   help='Name of configuration file.',
                                      type=str, default=None)
    parser.add_argument('--log',      help='Name of log file.',
                                      type=str, default=LOG_FILE)

    args = parser.parse_args()

    # If calling from the command line, then send logging to log file instead of screen
    logging_params = DEFAULT_LOGGING

    if args.log != 'stdout':
        log_dir = os.path.dirname(args.log)
        if log_dir:
            try:
                os.makedirs(log_dir)
            except OSError:
                if not os.path.isdir(log_dir):
                    raise

        logging_params['handlers'] = {'stderr': {'class': 'logging.handlers.WatchedFileHandler',
                                             'filename': args.log, 'formatter': 'std', 'level': 'INFO'}}

    # Call main routine
    main(args.filename, config_file=args.config, logging_params=logging_params)
