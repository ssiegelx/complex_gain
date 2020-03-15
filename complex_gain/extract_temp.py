import os
import glob
import datetime
import time
import argparse
from dateutil import relativedelta

import numpy as np
import h5py
import pandas as pd

from ch_util import andata
from ch_util import ephemeris

import wtl.log as log


CHIME_ARCHIVE = '/project/rpp-krs/chime/chime_archive'

###################################################
# Set up logging
###################################################

LOG_FILE = os.environ.get('HKP_LOG_FILE',
           os.path.join(os.path.dirname(os.path.realpath(__file__)), 'extract_temp.log'))

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

log.setup_logging(DEFAULT_LOGGING)


###################################################
# Define temperatures to extract
###################################################

# SPEC = {'ext_sensor_value': {'labels': ['hut', 'device'], 'ignore': ['Humidity'], 'scale': 0.10},
#         'weather_outTemp': {},
#         'fpga_motherboard_temp': {'labels':  ['sensor', 'crate_number', 'slot']},
#         'fpga_backplane_temp': {'labels':  ['crate_number', 'sensor']},
#         'chime_lna_temperature': {'labels': ['sensor_id']},
#         'fpga_power_supply_power': {'labels': ['name']}}

SPEC = {'fpga_mezzanine_voltage': {'labels':  ['sensor', 'crate_number', 'slot', 'mezzanine'], 'prefix': True},
        'fpga_mezzanine_current': {'labels':  ['sensor', 'crate_number', 'slot', 'mezzanine'], 'prefix': True}}


###################################################
# Routines
###################################################

def get_temperatures(year, month):

    logger = log.get_logger(__name__)

    metrics = sorted(SPEC.keys())

    this_date = datetime.datetime(year, month, 1).strftime("%Y%m%dT%H%M%SZ")

    filenames = sorted(glob.glob(os.path.join(CHIME_ARCHIVE, '%s_chime_hkp' % this_date, 'hkp_prom_*.h5')))
    nfiles = len(filenames)

    all_pdframe = {}

    for filename in filenames:

        logger.info("Now loading %s" % filename)

        # Load the housekeeping data for this day
        for mm in metrics:
            try:
                hkp = andata.HKPData.from_acq_h5([filename], metrics=[mm])

            except KeyError as kex:
                continue

            else:
                if mm not in all_pdframe:
                    all_pdframe[mm] = [ hkp.select(mm) ]
                else:
                    all_pdframe[mm].append( hkp.select(mm) )


    # Define function for interpolating the temperatures
    def _get_temp(stream, scale_factor=1.0):

        xtime = ephemeris.datetime_to_unix(stream.index.to_pydatetime())
        ytemp = stream.as_matrix()

        return xtime, ytemp * scale_factor


    # Concatenate the days
    all_names = []
    all_temps = []
    all_times = []

    for mm in sorted(all_pdframe.keys()):

        pdframe = pd.concat(all_pdframe[mm])

        labels = SPEC[mm].get('labels', None)
        scale = SPEC[mm].get('scale', 1.0)

        if (labels is None):

            this_time, this_temp = _get_temp(pdframe.value, scale_factor=scale)

            all_names.append(mm)
            all_temps.append(this_temp[:])
            all_times.append(this_time[:])

        else:

            # Extract list of devices
            devices = sorted(list(set(zip(*[pdframe[lbl] for lbl in labels]))))

            for dd, dev in enumerate(devices):

                if ('ignore' in SPEC[mm]) and any([xx in SPEC[mm]['ignore'] for xx in dev]):
                    continue

                name = '_'.join(['_'.join(xx.replace('-', '').split()) for xx in dev]).lower()

                if SPEC[mm].get('prefix', False):
                    name = '_'.join([mm, name])

                query = ' and '.join(['%s=="%s"' % (lbl, dev[ii]) for ii, lbl in enumerate(labels) ])

                stream = pdframe.query(query).value

                this_time, this_temp = _get_temp(stream, scale_factor=scale)

                all_names.append(name)
                all_temps.append(this_temp[:])
                all_times.append(this_time[:])

    return all_names, all_times, all_temps


def main(start_date, nmonth, output_dir):

    logger = log.get_logger(__name__)

    for mm in range(nmonth):

        t0 = time.time()

        this_date = start_date + relativedelta.relativedelta(months=mm)

        output_file = os.path.join(output_dir, '%s_chime_temperatures.h5' % this_date.strftime("%Y%m"))

        logger.info(this_date)
        logger.info(output_file)
        logger.info(''.join(['-'] * 80))

        res = get_temperatures(this_date.year, this_date.month)

        logger.info("Took %0.1f minutes to load temperatures." % ((time.time() - t0) / 60.0, ))
        t0 = time.time()

        with h5py.File(output_file, 'w') as handler:

            for sensor, times, temps in zip(*res):

                grp = handler.create_group(sensor)
                grp.create_dataset('time', data=times)
                grp.create_dataset('temp', data=temps)

        logger.info("Took %0.1f minutes to save to disk." % ((time.time() - t0) / 60.0, ))



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--date',     help='Date to start processing.',
                                      type=str, required=True)
    parser.add_argument('--out',      help='Name of output directory.',
                                      type=str, required=True)
    parser.add_argument('--nmonth',   help='Number of months to process.',
                                      type=int, default=6)
    parser.add_argument('--log',      help='Name of log file.',
                                      type=str, default=LOG_FILE)

    args = parser.parse_args()

    # If calling from the command line, then send logging to log file instead of screen
    try:
        os.makedirs(os.path.dirname(args.log))
    except OSError:
        if not os.path.isdir(os.path.dirname(args.log)):
            raise

    logging_params = DEFAULT_LOGGING
    logging_params['handlers'] = {'stderr': {'class': 'logging.handlers.WatchedFileHandler',
                                             'filename': args.log, 'formatter': 'std', 'level': 'INFO'}}

    log.setup_logging(logging_params)

    start_date = datetime.datetime(*[int(aa) for aa in args.date.replace(' ', '').split('_')])

    # Call main routine
    main(start_date, args.nmonth, args.out)
