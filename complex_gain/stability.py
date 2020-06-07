import os
import glob
import datetime
import argparse

import h5py
import scipy.interpolate
import numpy as np
import weighted as wq
from sklearn.linear_model import HuberRegressor

import wtl.log as log
from wtl.namespace import NameSpace
from wtl.config import load_yaml_config

from ch_util import ephemeris
from ch_util import andata
from ch_util import tools
from ch_util import timing
from ch_util import rfi
from ch_util import finder

from ch_pipeline.core import containers
from ch_pipeline.analysis.flagging import daytime_flag

from temps import TempData


###################################################
# default variables
###################################################

DEFAULTS = NameSpace(load_yaml_config(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                                   'defaults.yaml') + ':stability'))

LOG_FILE = os.environ.get('STABILITY_LOG_FILE',
           os.path.join(os.path.dirname(os.path.realpath(__file__)), 'stability.log'))

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

def reorder_inputs(inputmap, inputs):

    serials = inputs['correlator_input']

    isort = []
    remain = range(len(serials))

    for inp in inputmap:
        for rr in remain:
            if serials[rr] == inp.input_sn:
                isort.append(rr)
                break
        remain.remove(rr)

    return np.array(isort)


def construct_delay_template(omega, phase, flag, min_num_freq_for_delay_fit=100):

    nfreq, ninput, ntransit = phase.shape

    tau = np.zeros((ninput, ntransit), dtype=np.float32)
    tau_flag = np.zeros((ninput, ntransit), dtype=np.bool)

    for tt in range(ntransit):
        for ii in range(ninput):

            fflag = flag[:, ii, tt]

            if np.sum(fflag, dtype=np.int) > min_num_freq_for_delay_fit:

                x = omega[fflag]
                y = phase[fflag, ii, tt]

                try:
                    huber = HuberRegressor(fit_intercept=False).fit(x.reshape(-1, 1), y)

                except Exception:
                    continue

                else:
                    tau[ii, tt] = huber.coef_[0]
                    tau_flag[ii, tt] = True

    return tau, tau_flag


def compute_common_mode(y, flag, groups, median=True):

    nfreq, ninput, ntransit = y.shape
    ngroup = len(groups)

    shp = (nfreq, ngroup, ntransit)

    flg = np.zeros(shp, dtype=np.bool)
    cmn = np.zeros(shp, dtype=y.dtype)

    for ff in range(nfreq):

        for tt in range(ntransit):

            for gg, igroup in enumerate(groups):

                this_flag = flag[ff, igroup, tt].astype(np.float32)
                this_y = y[ff, igroup, tt]

                if np.any(this_flag):

                    flg[ff, gg, tt] = True

                    if median:
                        cmn[ff, gg, tt] = wq.median(this_y, this_flag)

                    else:
                        cmn[ff, gg, tt] = np.sum(this_flag * this_y) * tools.invert_no_zero(np.sum(this_flag))

    return cmn, flg

###################################################
# main routine
###################################################

def main(config_file=None, logging_params=DEFAULT_LOGGING):

    # Load config
    config = DEFAULTS.deepcopy()
    if config_file is not None:
        config.merge(NameSpace(load_yaml_config(config_file)))

    # Setup logging
    log.setup_logging(logging_params)
    logger = log.get_logger(__name__)

    ## Load data for flagging
    # Load fpga restarts
    time_fpga_restart = []
    if config.fpga_restart_file is not None:

        with open(config.fpga_restart_file, 'r') as handler:
            for line in handler:
                time_fpga_restart.append(ephemeris.datetime_to_unix(
                                            ephemeris.timestr_to_datetime(line.split('_')[0])))

    time_fpga_restart = np.array(time_fpga_restart)

    # Load housekeeping flag
    if config.housekeeping_file is not None:
        ftemp = TempData.from_acq_h5(config.housekeeping_file, datasets=["time_flag"])
    else:
        ftemp = None

    # Load jump data
    if config.jump_file is not None:
        with h5py.File(config.jump_file, 'r') as handler:
            jump_time = handler["time"][:]
            jump_size = handler["jump_size"][:]
    else:
        jump_time = None
        jump_size = None

    # Load rain data
    if config.rain_file is not None:
        with h5py.File(config.rain_file, 'r') as handler:
            rain_ranges = handler["time_range_conservative"][:]
    else:
        rain_ranges = []

    # Load data flags
    data_flags = {}
    if config.data_flags:
        finder.connect_database()
        flag_types = finder.DataFlagType.select()
        possible_data_flags = []
        for ft in flag_types:
            possible_data_flags.append(ft.name)
            if ft.name in config.data_flags:
                new_data_flags = finder.DataFlag.select().where(finder.DataFlag.type == ft)
                data_flags[ft.name] = list(new_data_flags)

    # Set desired range of time
    start_time = (ephemeris.datetime_to_unix(datetime.datetime(*config.start_date))
                  if config.start_date is not None else None)
    end_time = (ephemeris.datetime_to_unix(datetime.datetime(*config.end_date))
                if config.end_date is not None else None)


    ## Find gain files
    files = {}
    for src in config.sources:
        files[src] = sorted(glob.glob(os.path.join(config.directory, src.lower(),
                                                   "%s_%s_lsd_*.h5" % (config.prefix, src.lower(),))))
    csd = {}
    for src in config.sources:
        csd[src] = np.array([int(os.path.splitext(ff)[0][-4:]) for ff in files[src]])

    for src in config.sources:
        logger.info("%s:  %d files" % (src, len(csd[src])))


    ## Remove files that occur during flag
    csd_flag = {}
    for src in config.sources:

        body = ephemeris.source_dictionary[src]

        csd_flag[src] = np.ones(csd[src].size, dtype=np.bool)

        for ii, cc in enumerate(csd[src][:]):

            ttrans = ephemeris.transit_times(body, ephemeris.csd_to_unix(cc))[0]

            if (start_time is not None) and (ttrans < start_time):
                csd_flag[src][ii] = False
                continue

            if (end_time is not None) and (ttrans > end_time):
                csd_flag[src][ii] = False
                continue

            # If requested, remove daytime transits
            if not config.include_daytime.get(src, config.include_daytime.default) and daytime_flag(ttrans)[0]:
                logger.info("%s CSD %d:  daytime transit" % (src, cc))
                csd_flag[src][ii] = False
                continue

            # Remove transits during HKP drop out
            if ftemp is not None:
                itemp = np.flatnonzero((ftemp.time[:] >= (ttrans - config.transit_window)) &
                                       (ftemp.time[:] <= (ttrans + config.transit_window)))
                tempflg = ftemp['time_flag'][itemp]
                if (tempflg.size == 0) or ((np.sum(tempflg, dtype=np.float32) / float(tempflg.size)) < 0.50):
                    logger.info("%s CSD %d:  no housekeeping" % (src, cc))
                    csd_flag[src][ii] = False
                    continue

            # Remove transits near jumps
            if jump_time is not None:
                njump = np.sum((jump_size > config.min_jump_size) &
                               (jump_time > (ttrans - config.jump_window)) &
                               (jump_time < ttrans))
                if njump > config.max_njump:
                    logger.info("%s CSD %d:  %d jumps before" % (src, cc, njump))
                    csd_flag[src][ii] = False
                    continue

            # Remove transits near rain
            for rng in rain_ranges:
                if (((ttrans - config.transit_window) <= rng[1]) and
                    ((ttrans + config.transit_window) >= rng[0])):

                    logger.info("%s CSD %d:  during rain" % (src, cc))
                    csd_flag[src][ii] = False
                    break

            # Remove transits during data flag
            for name, flag_list in data_flags.items():

                if csd_flag[src][ii]:

                    for flg in flag_list:

                        if (((ttrans - config.transit_window) <= flg.finish_time) and
                            ((ttrans + config.transit_window) >= flg.start_time)):

                            logger.info("%s CSD %d:  %s flag" % (src, cc, name))
                            csd_flag[src][ii] = False
                            break

    # Print number of files left after flagging
    for src in config.sources:
        logger.info("%s:  %d files (after flagging)" % (src, np.sum(csd_flag[src])))


    ## Construct pair wise differences
    npair = len(config.diff_pair)
    shift = [nd * 24.0 * 3600.0 for nd in config.nday_shift]

    calmap = []
    calpair = []

    for (tsrc, csrc), sh in zip(config.diff_pair, shift):

        body_test = ephemeris.source_dictionary[tsrc]
        body_cal  = ephemeris.source_dictionary[csrc]

        for ii, cc in enumerate(csd[tsrc]):

            if csd_flag[tsrc][ii]:

                test_transit = ephemeris.transit_times(body_test, ephemeris.csd_to_unix(cc))[0]
                cal_transit = ephemeris.transit_times(body_cal, test_transit + sh)[0]
                cal_csd = int(np.fix(ephemeris.unix_to_csd(cal_transit)))

                ttrans = np.sort([test_transit, cal_transit])

                if cal_csd in csd[csrc]:
                    jj = list(csd[csrc]).index(cal_csd)

                    if csd_flag[csrc][jj] and not np.any((time_fpga_restart >= ttrans[0]) &
                                                         (time_fpga_restart <= ttrans[1])):
                        calmap.append([ii, jj])
                        calpair.append([tsrc, csrc])


    calmap = np.array(calmap)
    calpair = np.array(calpair)

    ntransit = calmap.shape[0]

    logger.info("%d total transit pairs" % ntransit)
    for ii in range(ntransit):

        t1 = ephemeris.transit_times(ephemeris.source_dictionary[calpair[ii, 0]],
                                     ephemeris.csd_to_unix(csd[calpair[ii, 0]][calmap[ii, 0]]))[0]
        t2 = ephemeris.transit_times(ephemeris.source_dictionary[calpair[ii, 1]],
                                     ephemeris.csd_to_unix(csd[calpair[ii, 1]][calmap[ii, 1]]))[0]

        logger.info("%s (%d) - %s (%d):  %0.1f hr" % (calpair[ii, 0], csd_flag[calpair[ii, 0]][calmap[ii, 0]],
                                                calpair[ii, 1], csd_flag[calpair[ii, 1]][calmap[ii, 1]],
                                                (t1 - t2) / 3600.0))

    # Determine unique diff pairs
    diff_name = np.array(['%s/%s' % tuple(cp) for cp in calpair])
    uniq_diff, lbl_diff, cnt_diff  = np.unique(diff_name, return_inverse=True, return_counts=True)
    ndiff = uniq_diff.size

    for ud, udcnt in zip(uniq_diff, cnt_diff):
        logger.info("%s:  %d transit pairs" % (ud, udcnt))

    ## Load gains
    inputmap = tools.get_correlator_inputs(datetime.datetime.utcnow(), correlator='chime')
    ninput = len(inputmap)
    nfreq = 1024

    # Set up gain arrays
    gain = np.zeros((2, nfreq, ninput, ntransit), dtype=np.complex64)
    weight = np.zeros((2, nfreq, ninput, ntransit), dtype=np.float32)
    input_sort = np.zeros((2, ninput, ntransit), dtype=np.int)

    kcsd = np.zeros((2, ntransit), dtype=np.float32)
    timestamp = np.zeros((2, ntransit), dtype=np.float64)
    is_daytime = np.zeros((2, ntransit), dtype=np.bool)

    for tt in range(ntransit):

        for kk, (src, ind) in enumerate(zip(calpair[tt], calmap[tt])):

            body = ephemeris.source_dictionary[src]
            filename = files[src][ind]

            logger.info("%s:  %s" % (src, filename))

            temp = containers.StaticGainData.from_file(filename)

            freq = temp.freq[:]
            inputs = temp.input[:]

            isort = reorder_inputs(inputmap, inputs)
            inputs = inputs[isort]

            gain[kk, :, :, tt] = temp.gain[:, isort]
            weight[kk, :, :, tt] = temp.weight[:, isort]
            input_sort[kk, :, tt] = isort

            kcsd[kk, tt] = temp.attrs['lsd']
            timestamp[kk, tt] = ephemeris.transit_times(body, ephemeris.csd_to_unix(kcsd[kk, tt]))[0]
            is_daytime[kk, tt] = daytime_flag(timestamp[kk, tt])[0]

            if np.any(isort != np.arange(isort.size)):
                logger.info("Input ordering has changed: %s" %
                            ephemeris.unix_to_datetime(timestamp[kk, tt]).strftime("%Y-%m-%d"))

        logger.info("")

    inputs = np.array([(inp.id, inp.input_sn) for inp in inputmap],
                      dtype=[('chan_id', 'u2'), ('correlator_input', 'S32')])

    ## Load input flags
    inpflg = np.ones((2, ninput, ntransit), dtype=np.bool)

    min_flag_time = np.min(timestamp) - 7.0 * 24.0 * 60.0 * 60.0
    max_flag_time = np.max(timestamp) + 7.0 * 24.0 * 60.0 * 60.0

    flaginput_files = sorted(glob.glob(os.path.join(config.flaginput_dir,
                                                    "*" + config.flaginput_suffix,
                                                    "*.h5")))

    if flaginput_files:
        logger.info("Found %d flaginput files." % len(flaginput_files))
        tmp = andata.FlagInputData.from_acq_h5(flaginput_files, datasets=())
        start, stop = [int(yy) for yy in
               np.percentile(np.flatnonzero((tmp.time[:] >= min_flag_time) & (tmp.time[:] <= max_flag_time)),
                             [0, 100])]

        cont = andata.FlagInputData.from_acq_h5(flaginput_files,
                                                start=start, stop=stop, datasets=['flag'])

        for kk in range(2):
            inpflg[kk, :, :] = cont.resample('flag', timestamp[kk], transpose=True)

            logger.info("Flaginput time offsets in minutes (pair %d):" % kk)
            logger.info(str(np.fix((
                cont.time[cont.search_update_time(timestamp[kk])] - timestamp[kk]) / 60.0).astype(np.int)))

    # Sort flags so they are in same order
    for tt in range(ntransit):
        for kk in range(2):
            inpflg[kk, :, tt] = inpflg[kk, input_sort[kk, :, tt], tt]

    # Do not apply input flag to phase reference
    for ii in config.index_phase_ref:
        inpflg[:, ii, :] = True

    ## Flag out gains with high uncertainty and frequencies with large fraction of data flagged
    frac_err = tools.invert_no_zero(np.sqrt(weight) * np.abs(gain))

    flag = np.all((weight > 0.0) & (np.abs(gain) > 0.0) &
                  (frac_err < config.max_uncertainty), axis=0)

    freq_flag = ((np.sum(flag, axis=(1, 2), dtype=np.float32) /
                  float(np.prod(flag.shape[1:]))) > config.freq_threshold)

    if config.apply_rfi_mask:
        freq_flag &= np.logical_not(rfi.frequency_mask(freq))

    flag = flag & freq_flag[:, np.newaxis, np.newaxis]

    good_freq = np.flatnonzero(freq_flag)

    logger.info("Number good frequencies %d" % good_freq.size)


    ## Generate flags with more conservative cuts on frequency
    c_flag = flag & np.all(frac_err < config.conservative.max_uncertainty, axis=0)

    c_freq_flag = ((np.sum(c_flag, axis=(1, 2), dtype=np.float32) /
                   float(np.prod(c_flag.shape[1:]))) > config.conservative.freq_threshold)

    if config.conservative.apply_rfi_mask:
        c_freq_flag &= np.logical_not(rfi.frequency_mask(freq))

    c_flag = c_flag & c_freq_flag[:, np.newaxis, np.newaxis]

    c_good_freq = np.flatnonzero(c_freq_flag)

    logger.info("Number good frequencies (conservative thresholds) %d" % c_good_freq.size)


    ## Apply input flags
    flag &= np.all(inpflg[:, np.newaxis, :, :], axis=0)


    ## Update flags based on beam flag
    if config.beam_flag_file is not None:

        dbeam = andata.BaseData.from_acq_h5(config.beam_flag_file)

        db_csd = np.floor(ephemeris.unix_to_csd(dbeam.index_map['time'][:]))

        for ii, name in enumerate(config.beam_flag_datasets):
            logger.info("Applying %s beam flag." % name)
            if not ii:
                db_flag = dbeam.flags[name][:]
            else:
                db_flag &= dbeam.flags[name][:]

        cnt = 0
        for ii, dbc in enumerate(db_csd):

            this_csd = np.flatnonzero(np.any(kcsd == dbc, axis=0))

            if this_csd.size > 0:

                logger.info("Beam flag for %d matches %s." % (dbc, str(kcsd[:, this_csd])))

                flag[:, :, this_csd] &= db_flag[np.newaxis, :, ii, np.newaxis]

                cnt += 1

        logger.info("Applied %0.1f percent of the beam flags" %
                    (100.0 * cnt / float(db_csd.size),))


    ## Flag inputs with large amount of missing data
    input_frac_flagged = (np.sum(flag[good_freq, :, :], axis=(0, 2), dtype=np.float32) /
                          float(good_freq.size * ntransit))
    input_flag = input_frac_flagged > config.input_threshold

    for ii in config.index_phase_ref:
        logger.info("Phase reference %d has %0.3f fraction of data flagged." % (ii, input_frac_flagged[ii]))
        input_flag[ii] = True

    good_input = np.flatnonzero(input_flag)

    flag = flag & input_flag[np.newaxis, :, np.newaxis]

    logger.info("Number good inputs %d" % good_input.size)

    ## Calibrate
    gaincal = gain[0] * tools.invert_no_zero(gain[1])

    frac_err_cal = np.sqrt(frac_err[0]**2 + frac_err[1]**2)

    count = np.sum(flag, axis=-1, dtype=np.int)
    stat_flag = count > config.min_num_transit

    ## Calculate phase
    amp = np.abs(gaincal)
    phi = np.angle(gaincal)

    ## Calculate polarisation groups
    pol_dict = {'E': 'X', 'S': 'Y'}
    cyl_dict = {2: 'A', 3: 'B', 4: 'C', 5: 'D'}

    if config.group_by_cyl:
        group_id = [(inp.pol, inp.cyl) if tools.is_chime(inp) and (ii in good_input) else None
                    for ii, inp in enumerate(inputmap)]
    else:
        group_id = [inp.pol if tools.is_chime(inp) and (ii in good_input) else None
                    for ii, inp in enumerate(inputmap)]

    ugroup_id = sorted([uidd for uidd in set(group_id) if uidd is not None])
    ngroup = len(ugroup_id)

    group_list_noref = [np.array([gg for gg, gid in enumerate(group_id) if (gid == ugid) and gg not in config.index_phase_ref])
                        for ugid in ugroup_id]

    group_list = [np.array([gg for gg, gid in enumerate(group_id) if gid == ugid])
                  for ugid in ugroup_id]

    if config.group_by_cyl:
        group_str = ["%s-%s" % (pol_dict[pol], cyl_dict[cyl]) for pol, cyl in ugroup_id]
    else:
        group_str = [pol_dict[pol] for pol in ugroup_id]

    index_phase_ref = []
    for gstr, igroup in zip(group_str, group_list):
        candidate = [ii for ii in config.index_phase_ref if ii in igroup]
        if len(candidate) != 1:
            index_phase_ref.append(None)
        else:
            index_phase_ref.append(candidate[0])

    logger.info("Phase reference: %s" % ', '.join(['%s = %s' % tpl for tpl in zip(group_str, index_phase_ref)]))

    ## Apply thermal correction to amplitude
    if config.amp_thermal.enabled:

        logger.info("Applying thermal correction.")

        # Load the temperatures
        tdata = TempData.from_acq_h5(config.amp_thermal.filename)

        index = tdata.search_sensors(config.amp_thermal.sensor)[0]

        temp = tdata.datasets[config.amp_thermal.field][index]
        temp_func = scipy.interpolate.interp1d(tdata.time, temp, **config.amp_thermal.interp)

        itemp = temp_func(timestamp)
        dtemp = itemp[0] - itemp[1]

        flag_func = scipy.interpolate.interp1d(tdata.time, tdata.datasets['flag'][index].astype(np.float32),
                                               **config.amp_thermal.interp)

        dtemp_flag = np.all(flag_func(timestamp) == 1.0, axis=0)

        flag &= dtemp_flag[np.newaxis, np.newaxis, :]

        for gstr, igroup in zip(group_str, group_list):
            pstr = gstr[0]
            thermal_coeff = np.polyval(config.amp_thermal.coeff[pstr], freq)
            gthermal = 1.0 + thermal_coeff[:, np.newaxis, np.newaxis] * dtemp[np.newaxis, np.newaxis, :]

            amp[:, igroup, :] *= tools.invert_no_zero(gthermal)


    ## Compute common mode
    if config.subtract_common_mode_before:
        logger.info("Calculating common mode amplitude and phase.")
        cmn_amp, flag_cmn_amp = compute_common_mode(amp, flag, group_list_noref, median=False)
        cmn_phi, flag_cmn_phi = compute_common_mode(phi, flag, group_list_noref, median=False)

        # Subtract common mode (from phase only)
        logger.info("Subtracting common mode phase.")
        group_flag = np.zeros((ngroup, ninput), dtype=np.bool)
        for gg, igroup in enumerate(group_list):
            group_flag[gg, igroup] = True
            phi[:, igroup, :] = phi[:, igroup, :] - cmn_phi[:, gg, np.newaxis, :]

            for iref in index_phase_ref:
                if (iref is not None) and (iref in igroup):
                    flag[:, iref, :] = flag_cmn_phi[:, gg, :]


    ## If requested, determine and subtract a delay template
    if config.fit_delay_before:
        logger.info("Fitting delay template.")
        omega = timing.FREQ_TO_OMEGA * freq

        tau, tau_flag = construct_delay_template(omega, phi, c_flag & flag,
                                                 min_num_freq_for_delay_fit=config.min_num_freq_for_delay_fit)

        # Compute residuals
        logger.info("Subtracting delay template.")
        phi = phi - tau[np.newaxis, :, :] * omega[:, np.newaxis, np.newaxis]


    ## Normalize by median over time
    logger.info("Calculating median amplitude and phase.")
    med_amp = np.zeros((nfreq, ninput, ndiff), dtype=amp.dtype)
    med_phi = np.zeros((nfreq, ninput, ndiff), dtype=phi.dtype)

    count_by_diff = np.zeros((nfreq, ninput, ndiff), dtype=np.int)
    stat_flag_by_diff = np.zeros((nfreq, ninput, ndiff), dtype=np.bool)

    def weighted_mean(yy, ww, axis=-1):
        return np.sum(ww * yy, axis=axis) * tools.invert_no_zero(np.sum(ww, axis=axis))

    for dd in range(ndiff):

        this_diff = np.flatnonzero(lbl_diff == dd)

        this_flag = flag[:, :, this_diff]

        this_amp = amp[:, :, this_diff]
        this_amp_err = this_amp * frac_err_cal[:, :, this_diff] * this_flag.astype(np.float32)

        this_phi = phi[:, :, this_diff]
        this_phi_err = frac_err_cal[:, :, this_diff] * this_flag.astype(np.float32)

        count_by_diff[:, :, dd] = np.sum(this_flag, axis=-1, dtype=np.int)
        stat_flag_by_diff[:, :, dd] = count_by_diff[:, :, dd] > config.min_num_transit

        if config.weighted_mean == 2:
            logger.info("Calculating inverse variance weighted mean.")
            med_amp[:, :, dd] = weighted_mean(this_amp, tools.invert_no_zero(this_amp_err**2), axis=-1)
            med_phi[:, :, dd] = weighted_mean(this_phi, tools.invert_no_zero(this_phi_err**2), axis=-1)

        elif config.weighted_mean == 1:
            logger.info("Calculating uniform weighted mean.")
            med_amp[:, :, dd] = weighted_mean(this_amp, this_flag.astype(np.float32), axis=-1)
            med_phi[:, :, dd] = weighted_mean(this_phi, this_flag.astype(np.float32), axis=-1)

        else:
            logger.info("Calculating median value.")
            for ff in range(nfreq):
                for ii in range(ninput):
                    if np.any(this_flag[ff, ii, :]):
                        med_amp[ff, ii, dd] = wq.median(this_amp[ff, ii, :], this_flag[ff, ii, :].astype(np.float32))
                        med_phi[ff, ii, dd] = wq.median(this_phi[ff, ii, :], this_flag[ff, ii, :].astype(np.float32))

    damp = np.zeros_like(amp)
    dphi = np.zeros_like(phi)
    for dd in range(ndiff):
        this_diff = np.flatnonzero(lbl_diff == dd)
        damp[:, :, this_diff] = amp[:, :, this_diff] * tools.invert_no_zero(med_amp[:, :, dd, np.newaxis]) - 1.0
        dphi[:, :, this_diff] = phi[:, :, this_diff] - med_phi[:, :, dd, np.newaxis]


    # Compute common mode
    if not config.subtract_common_mode_before:
        logger.info("Calculating common mode amplitude and phase.")
        cmn_amp, flag_cmn_amp = compute_common_mode(damp, flag, group_list_noref, median=True)
        cmn_phi, flag_cmn_phi = compute_common_mode(dphi, flag, group_list_noref, median=True)

        # Subtract common mode (from phase only)
        logger.info("Subtracting common mode phase.")
        group_flag = np.zeros((ngroup, ninput), dtype=np.bool)
        for gg, igroup in enumerate(group_list):
            group_flag[gg, igroup] = True
            dphi[:, igroup, :] = dphi[:, igroup, :] - cmn_phi[:, gg, np.newaxis, :]

            for iref in index_phase_ref:
                if (iref is not None) and (iref in igroup):
                    flag[:, iref, :] = flag_cmn_phi[:, gg, :]

    ## Compute RMS
    logger.info("Calculating RMS of amplitude and phase.")
    mad_amp = np.zeros((nfreq, ninput), dtype=amp.dtype)
    std_amp = np.zeros((nfreq, ninput), dtype=amp.dtype)

    mad_phi = np.zeros((nfreq, ninput), dtype=phi.dtype)
    std_phi = np.zeros((nfreq, ninput), dtype=phi.dtype)

    mad_amp_by_diff = np.zeros((nfreq, ninput, ndiff), dtype=amp.dtype)
    std_amp_by_diff = np.zeros((nfreq, ninput, ndiff), dtype=amp.dtype)

    mad_phi_by_diff = np.zeros((nfreq, ninput, ndiff), dtype=phi.dtype)
    std_phi_by_diff = np.zeros((nfreq, ninput, ndiff), dtype=phi.dtype)

    for ff in range(nfreq):
        for ii in range(ninput):
            this_flag = flag[ff, ii, :]
            if np.any(this_flag):
                std_amp[ff, ii] = np.std(damp[ff, ii, this_flag])
                std_phi[ff, ii] = np.std(dphi[ff, ii, this_flag])

                mad_amp[ff, ii] = 1.48625 * wq.median(np.abs(damp[ff, ii, :]), this_flag.astype(np.float32))
                mad_phi[ff, ii] = 1.48625 * wq.median(np.abs(dphi[ff, ii, :]), this_flag.astype(np.float32))

                for dd in range(ndiff):
                    this_diff = this_flag & (lbl_diff == dd)
                    if np.any(this_diff):

                        std_amp_by_diff[ff, ii, dd] = np.std(damp[ff, ii, this_diff])
                        std_phi_by_diff[ff, ii, dd] = np.std(dphi[ff, ii, this_diff])

                        mad_amp_by_diff[ff, ii, dd] = 1.48625 * wq.median(np.abs(damp[ff, ii, :]), this_diff.astype(np.float32))
                        mad_phi_by_diff[ff, ii, dd] = 1.48625 * wq.median(np.abs(dphi[ff, ii, :]), this_diff.astype(np.float32))


    ## Construct delay template
    if not config.fit_delay_before:
        logger.info("Fitting delay template.")
        omega = timing.FREQ_TO_OMEGA * freq

        tau, tau_flag = construct_delay_template(omega, dphi, c_flag & flag,
                                                 min_num_freq_for_delay_fit=config.min_num_freq_for_delay_fit)

        # Compute residuals
        logger.info("Subtracting delay template from phase.")
        resid = (dphi - tau[np.newaxis, :, :] * omega[:, np.newaxis, np.newaxis]) * flag.astype(np.float32)

    else:
        resid = dphi

    tau_count = np.sum(tau_flag, axis=-1, dtype=np.int)
    tau_stat_flag = tau_count > config.min_num_transit

    tau_count_by_diff = np.zeros((ninput, ndiff), dtype=np.int)
    tau_stat_flag_by_diff = np.zeros((ninput, ndiff), dtype=np.bool)
    for dd in range(ndiff):
        this_diff = np.flatnonzero(lbl_diff == dd)
        tau_count_by_diff[:, dd] = np.sum(tau_flag[:, this_diff], axis=-1, dtype=np.int)
        tau_stat_flag_by_diff[:, dd] = tau_count_by_diff[:, dd] > config.min_num_transit


    ## Calculate statistics of residuals
    std_resid = np.zeros((nfreq, ninput), dtype=phi.dtype)
    mad_resid = np.zeros((nfreq, ninput), dtype=phi.dtype)

    std_resid_by_diff = np.zeros((nfreq, ninput, ndiff), dtype=phi.dtype)
    mad_resid_by_diff = np.zeros((nfreq, ninput, ndiff), dtype=phi.dtype)

    for ff in range(nfreq):
        for ii in range(ninput):
            this_flag = flag[ff, ii, :]
            if np.any(this_flag):
                std_resid[ff, ii] = np.std(resid[ff, ii, this_flag])
                mad_resid[ff, ii] = 1.48625 * wq.median(np.abs(resid[ff, ii, :]), this_flag.astype(np.float32))

                for dd in range(ndiff):
                    this_diff = this_flag & (lbl_diff == dd)
                    if np.any(this_diff):
                        std_resid_by_diff[ff, ii, dd] = np.std(resid[ff, ii, this_diff])
                        mad_resid_by_diff[ff, ii, dd] = 1.48625 * wq.median(np.abs(resid[ff, ii, :]), this_diff.astype(np.float32))


    ## Calculate statistics of delay template
    mad_tau = np.zeros((ninput,), dtype=phi.dtype)
    std_tau = np.zeros((ninput,), dtype=phi.dtype)

    mad_tau_by_diff = np.zeros((ninput, ndiff), dtype=phi.dtype)
    std_tau_by_diff = np.zeros((ninput, ndiff), dtype=phi.dtype)

    for ii in range(ninput):
        this_flag = tau_flag[ii]
        if np.any(this_flag):
            std_tau[ii] = np.std(tau[ii, this_flag])
            mad_tau[ii] = 1.48625 * wq.median(np.abs(tau[ii]), this_flag.astype(np.float32))

            for dd in range(ndiff):
                this_diff = this_flag & (lbl_diff == dd)
                if np.any(this_diff):
                    std_tau_by_diff[ii, dd] = np.std(tau[ii, this_diff])
                    mad_tau_by_diff[ii, dd] = 1.48625 * wq.median(np.abs(tau[ii]), this_diff.astype(np.float32))


    ## Define output
    res =  {
            "timestamp": {
                     "data": timestamp,
                     "axis": ["div", "time"]
                    },

            "is_daytime": {
                     "data": is_daytime,
                     "axis": ["div", "time"]
                    },

            "csd": {
                     "data": kcsd,
                     "axis": ["div", "time"]
                    },

            "pair_map": {
                     "data": lbl_diff,
                     "axis": ["time"]
                    },

            "pair_count": {
                     "data": cnt_diff,
                     "axis": ["pair"]
                    },

            "gain": {
                     "data": gaincal,
                     "axis": ["freq", "input", "time"]
                    },

            "frac_err": {
                     "data": frac_err_cal,
                     "axis": ["freq", "input", "time"]
                    },

            "flags/gain": {
                     "data": flag,
                     "axis": ["freq", "input", "time"],
                     "flag": True
                    },

            "flags/gain_conservative": {
                     "data": c_flag,
                     "axis": ["freq", "input", "time"],
                     "flag": True
                    },

            "flags/count": {
                     "data": count,
                     "axis": ["freq", "input"],
                     "flag": True
                    },

            "flags/stat": {
                     "data": stat_flag,
                     "axis": ["freq", "input"],
                     "flag": True
                    },

            "flags/count_by_pair": {
                     "data": count_by_diff,
                     "axis": ["freq", "input", "pair"],
                     "flag": True
                    },

            "flags/stat_by_pair": {
                     "data": stat_flag_by_diff,
                     "axis": ["freq", "input", "pair"],
                     "flag": True
                    },

            "med_amp": {
                     "data": med_amp,
                     "axis": ["freq", "input", "pair"]
                    },

            "med_phi": {
                     "data": med_phi,
                     "axis": ["freq", "input", "pair"]
                    },

            "flags/group_flag": {
                     "data": group_flag,
                     "axis": ["group", "input"],
                     "flag": True
                    },

            "cmn_amp": {
                     "data": cmn_amp,
                     "axis": ["freq", "group", "time"]
                    },

            "cmn_phi": {
                     "data": cmn_phi,
                     "axis": ["freq", "group", "time"]
                    },

            "amp": {
                     "data": damp,
                     "axis": ["freq", "input", "time"]
                    },

            "phi": {
                     "data": dphi,
                     "axis": ["freq", "input", "time"]
                    },

            "std_amp": {
                     "data": std_amp,
                     "axis": ["freq", "input"]
                    },

            "std_amp_by_pair": {
                     "data": std_amp_by_diff,
                     "axis": ["freq", "input", "pair"]
                    },

            "mad_amp": {
                     "data": mad_amp,
                     "axis": ["freq", "input"]
                    },

            "mad_amp_by_pair": {
                     "data": mad_amp_by_diff,
                     "axis": ["freq", "input", "pair"]
                    },

            "std_phi": {
                     "data": std_phi,
                     "axis": ["freq", "input"]
                    },

            "std_phi_by_pair": {
                     "data": std_phi_by_diff,
                     "axis": ["freq", "input", "pair"]
                    },

            "mad_phi": {
                     "data": mad_phi,
                     "axis": ["freq", "input"]
                    },

            "mad_phi_by_pair": {
                     "data": mad_phi_by_diff,
                     "axis": ["freq", "input", "pair"]
                    },

            "tau": {
                     "data": tau,
                     "axis": ["input", "time"]
                    },

            "flags/tau": {
                     "data": tau_flag,
                     "axis": ["input", "time"],
                     "flag": True
                    },

            "flags/tau_count": {
                     "data": tau_count,
                     "axis": ["input"],
                     "flag": True
                    },

            "flags/tau_stat": {
                     "data": tau_stat_flag,
                     "axis": ["input"],
                     "flag": True
                    },

            "flags/tau_count_by_pair": {
                     "data": tau_count_by_diff,
                     "axis": ["input", "pair"],
                     "flag": True
                    },

            "flags/tau_stat_by_pair": {
                     "data": tau_stat_flag_by_diff,
                     "axis": ["input", "pair"],
                     "flag": True
                    },

            "std_tau": {
                     "data": std_tau,
                     "axis": ["input"]
                    },

            "std_tau_by_pair": {
                     "data": std_tau_by_diff,
                     "axis": ["input", "pair"]
                    },

            "mad_tau": {
                     "data": mad_tau,
                     "axis": ["input"]
                    },

            "mad_tau_by_pair": {
                     "data": mad_tau_by_diff,
                     "axis": ["input", "pair"]
                    },

            "resid_phi": {
                     "data": resid,
                     "axis": ["freq", "input", "time"]
                    },

            "std_resid_phi": {
                     "data": std_resid,
                     "axis": ["freq", "input"]
                    },

            "std_resid_phi_by_pair": {
                     "data": std_resid_by_diff,
                     "axis": ["freq", "input", "pair"]
                    },

            "mad_resid_phi": {
                     "data": mad_resid,
                     "axis": ["freq", "input"]
                    },

            "mad_resid_phi_by_pair": {
                     "data": mad_resid_by_diff,
                     "axis": ["freq", "input", "pair"]
                    },
            }

    ## Create the output container
    logger.info("Creating StabilityData container.")
    data = StabilityData()

    data.create_index_map("div", np.array(["numerator", "denominator"], dtype=np.string_))
    data.create_index_map("pair", np.array(uniq_diff, dtype=np.string_))
    data.create_index_map("group", np.array(group_str, dtype=np.string_))

    data.create_index_map("freq", freq)
    data.create_index_map("input", inputs)
    data.create_index_map("time", timestamp[0, :])

    logger.info("Writing datsets to container.")
    for name, dct in res.iteritems():
        is_flag = dct.get('flag', False)
        if is_flag:
            dset = data.create_flag(name.split('/')[-1], data=dct['data'])
        else:
            dset = data.create_dataset(name, data=dct['data'])

        dset.attrs['axis'] = np.array(dct['axis'], dtype=np.string_)

    data.attrs['phase_ref'] = np.array([iref for iref in index_phase_ref if iref is not None])

    # Determine the output filename and save results
    start_time, end_time = ephemeris.unix_to_datetime(np.percentile(timestamp, [0, 100]))
    tfmt = "%Y%m%d"
    night_str = 'night_' if not np.any(is_daytime) else ''
    output_file = os.path.join(config.output_dir,
                               "%s_%s_%sraw_stability_data.h5" %
                               (start_time.strftime(tfmt), end_time.strftime(tfmt), night_str))

    logger.info("Saving results to %s." % output_file)
    data.save(output_file)


class StabilityData(andata.BaseData):

    convert_attribute_strings = True
    convert_dataset_strings = True

    @property
    def time(self):
        return self.index_map['time']

    @property
    def freq(self):
        return self.index_map['freq']


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