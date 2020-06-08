import logging
import time
import heapq

import numpy as np
import weighted as wq
import scipy.sparse
import scipy.interpolate
from scipy.interpolate import interp1d
from scipy.constants import c as speed_of_light
from sklearn.linear_model import HuberRegressor

from ch_util import tools
from ch_util import timing
from ch_util import ephemeris
from ch_util import cal_utils

from complex_gain import kzfilt

class TempRegression(object):

    def __init__(self, x, data, flag=None, time_flag=None):

        self.log = logging.getLogger(str(self))

        if flag is None:
            fshp = tuple([1] * (data.ndim - 1) + [data.shape[-1]])
            flag = np.ones(fshp, dtype=np.bool)

        if time_flag is None:
            fshp = tuple([1] * (data.ndim - 1) + [data.shape[-1]])
            time_flag = np.ones(fshp, dtype=np.bool)
        elif time_flag.ndim != data.ndim:
            fshp = tuple([None] * (data.ndim - time_flag.ndim) + [slice(None)] * time_flag.ndim)
            time_flag = time_flag[fshp]

        self._flag = flag
        self._time_flag = time_flag

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

    def process(self, **kwargs):

        self.fit_temp(**kwargs)
        self.stats()

    def _resolve_index(self, ind, shp):
        ind = np.atleast_1d(ind)
        return tuple([ii % ss for ii, ss in zip(ind, shp[0:len(ind)])])

    def flag(self, ind):
        return (self._flag[self._resolve_index(ind, self._flag.shape)] &
                self._time_flag[self._resolve_index(ind, self._time_flag.shape)])

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

    def fit_temp(self, max_iter=100):

        self.intercept = np.zeros(self.N, dtype=np.float32)
        self.number = np.zeros(self.N, dtype=np.int)
        self.coeff = np.zeros(self.N + (self.nfeature,), dtype=np.float32)

        for counter, ii in enumerate(np.ndindex(*self.N)):

            if not (counter % 1000):
                self.log.info("Fitting index %d of %d." % (counter, np.prod(self.N)))

            nsample = self.count(ii)
            if nsample > (self.nfeature+1):

                try:
                    huber = HuberRegressor(fit_intercept=True, max_iter=max_iter).fit(self.fit_x(ii), self.fit_data(ii))

                except Exception as exc:
                    self.log.info("Fit failed for index %s:  %s" % (str(ii), exc))

                else:
                    self.intercept[ii] = huber.intercept_
                    self.coeff[ii] = huber.coef_[:]
                    self.number[ii] = nsample

        self.model = self.intercept[..., np.newaxis] + np.sum(self.coeff[..., np.newaxis, :] * self._x, axis=-1)
        self.resid = self.data - self.model

    def stats(self, reshape=None, time_flag=None):

        if time_flag is None:
            time_flag = np.ones(self._flag.shape[-1], dtype=np.bool)

        fshp = tuple([None] * (self._flag.ndim - time_flag.ndim) + [slice(None)] * time_flag.ndim)
        time_flag = time_flag[fshp]

        flag = self._flag & time_flag

        for key in ['data', 'model', 'resid']:

            ts = np.where(flag, getattr(self, key), np.nan)
            if reshape is not None:
                ts = ts.reshape(reshape)

            mad = 1.46825 * np.nanmedian(np.abs(ts - np.nanmedian(ts, axis=-1, keepdims=True)), axis=-1)
            bad = ~np.isfinite(mad)
            if np.any(bad):
                mad[bad] = 0.0

            sigma = np.nanstd(ts, axis=-1)
            bad = ~np.isfinite(sigma)
            if np.any(bad):
                sigma[bad] = 0.0

            setattr(self, 'mad_' + key, mad)
            setattr(self, 'std_' + key, sigma)


class TempRegressionGroups(TempRegression):

    def __init__(self, x, data, groups, flag=None, time_flag=None):

        self.groups = groups
        self.ngroups = len(groups)

        super(TempRegressionGroups, self).__init__(x, data, flag=flag, time_flag=time_flag)

    def _resolve_group_indices(self, group_index, shp=None):
        if shp is None:
            shp = self.N
        return tuple(zip(*[self._resolve_index(ind, shp) for ind in self.groups[group_index]]))

    def flag(self, group_index):
        return (self._flag[self._resolve_group_indices(group_index, self._flag.shape)] &
                self._time_flag[self._resolve_group_indices(group_index, self._time_flag.shape)])

    def x(self, group_index):
        return self._x[self._resolve_group_indices(group_index, self._x.shape)]

    def fit_data(self, group_index):
        return self.data[self._resolve_group_indices(group_index)][self.flag(group_index)]

    def fit_x(self, group_index):
        flag = self.flag(group_index)
        return self.x(group_index)[flag]

    def fit_temp(self):

        self.intercept = np.zeros(self.N, dtype=np.float32)
        self.number = np.zeros(self.N, dtype=np.int)
        self.coeff = np.zeros(self.N + (self.nfeature,), dtype=np.float32)

        for gg in range(self.ngroups):

            self.log.info("Fitting group %d of %d." % (gg, self.ngroups))

            gindex = self._resolve_group_indices(gg)

            x = self.fit_x(gg)
            y = self.fit_data(gg)

            nsample = y.size
            if nsample > (self.nfeature+1):

                try:
                    huber = HuberRegressor(fit_intercept=True).fit(x, y)

                except Exception as exc:
                    self.log.info("Fit failed for index %s:  %s" % (str(ii), exc))

                else:
                    self.intercept[gindex] = huber.intercept_
                    self.coeff[gindex] = huber.coef_[:]
                    self.number[gindex] = nsample

        self.model = self.intercept[..., np.newaxis] + np.sum(self.coeff[..., np.newaxis, :] * self._x, axis=-1)
        self.resid = self.data - self.model



class JointTempRegression(TempRegression):

    def __init__(self, x, data, groups, classification=None,
                 flag=None, fit_intercept=True, coeff_name=None):

        self.log = logging.getLogger(str(self))

        self.fit_intercept = fit_intercept

        # Parse input, ensure consistent shapes
        self._time_flag = None
        if flag is None:
            flag = np.ones(data.shape, dtype=np.bool)

        if x.ndim == 1:
            x = x[:, np.newaxis]

        if x.ndim != (data.ndim+1):
            expand = tuple([None] * (data.ndim+1 - x.ndim) + [slice(None)] * x.ndim)
            x = x[expand]

        ninput, ntime = data.shape
        nfeature = x.shape[-1]

        self._base_x = x

        if coeff_name is None:
            coeff_name = ['feature%d' % dd for dd in range(nfeature)]

        self.coeff_name = coeff_name

        # Determine correlated groups
        self._groups = groups
        uniq_groups = [np.unique(groups[:, ff]) for ff in range(nfeature)]
        ncoeff = [ug.size for ug in uniq_groups]

        self._coeff_bounds = np.concatenate(([0], np.cumsum(ncoeff)))
        self.ncoeff = sum(ncoeff)

        # Repackage x into sparse array
        flat_index = np.arange(ninput * ntime, dtype=np.int).reshape(ninput, ntime)
        counter = 0
        row, col, y = [], [], []
        for ff, ugs in enumerate(uniq_groups):
            for uu, ug in enumerate(ugs):
                this_group = np.flatnonzero(groups[:, ff] == ug)
                this_x = x[this_group % x.shape[0], :, ff].ravel()

                y += list(this_x)
                row += list(flat_index[this_group, :].ravel())
                col += [counter] * this_x.size

                self._groups[this_group, ff] = uu
                counter += 1

        # Add input dependent intercept
        self.nintercept = 0
        self.intercept_name = ['intercept']
        if self.fit_intercept:
            if classification is not None:
                uniqc, indc = np.unique(classification, return_inverse=True)
                self.nintercept = uniqc.size
                self.intercept_name = uniqc
                cgroups = [(cg.size, cg) for cg in [np.flatnonzero(indc == uc) for uc in range(self.nintercept)]]
            else:
                self.nintercept = 1
                cgroups = [(ntime, slice(None))]

            for ii in range(ninput):
                for ng, cg in cgroups:
                    y += [1.0] * ng
                    row += list(flat_index[ii, cg])
                    col += [counter] * ng
                    counter += 1

        # Convert to sparse matrix
        y = np.array(y)
        row = np.array(row)
        col = np.array(col)

        nparam = self.ncoeff + self.nintercept * ninput
        self.log.info("%d parameters in total.  (%d coefficients, %d x %d intercepts)" %
                      (nparam, self.ncoeff, self.nintercept, ninput))

        self._x = scipy.sparse.coo_matrix((y, (row, col)), shape=(ninput * ntime, nparam))

        # Save datasets to object
        self._flag = flag
        self.data = data

        self.N = self.data.shape[:-1]
        self.nsample = self.data.shape[-1]
        self.nfeature = nfeature

    def fit_temp(self, sparse=False, lsqr=False, fmt='csr', time_flag=None, **kwargs):

        # Prepare data to fit.  Unravels over input, time.
        if time_flag is None:
            time_flag = np.ones(self._flag.shape[-1], dtype=np.bool)

        fshp = tuple([None] * (self._flag.ndim - time_flag.ndim) + [slice(None)] * time_flag.ndim)
        time_flag = time_flag[fshp]

        to_fit = np.flatnonzero((self._flag & time_flag).ravel())

        y = self.data.ravel()[to_fit]

        # Repackage x into spare array
        t0 = time.time()
        if fmt == 'csr':
            x = self._x.tocsr()[to_fit, :]

        elif fmt == 'csc':
            x = self._x.tocsc()[to_fit, :]

        else:
            raise ValueError("Do not recognize format: %s" % fmt)

        self.log.info("Took %0.2f seconds to format %s feature array." %
                      (time.time() - t0, fmt))

        # Call the scipy sparse least squares solver
        t0 = time.time()
        if sparse:
            if lsqr:
                self.log.info("Using sparse matrix least squares.")
                results = scipy.sparse.linalg.lsqr(x, y, **kwargs)

                self.fit_results = results
                coeff = results[0]

                if results[1] == 1:
                    self.log.info("coefficients are an approximate solution.")
                elif results[1] == 2:
                    self.log.info("coefficients approximately solve the least-squares problem.")
                else:
                    self.log.info("Do not recognize status code: %d" % results[1])

                self.log.info("Performed %d interations in total." % results[2])

            else:
                self.log.info("Using sparse matrix solver.")
                xT = x.T
                coeff = scipy.sparse.linalg.spsolve(self.cov, xT.dot(y), **kwargs)

        else:
            self.log.info("Using dense matrix solver.")
            xT = x.T
            cov = xT * x
            self.cov_coeff = np.linalg.pinv(cov.toarray(), **kwargs)
            yproj = xT.dot(y)
            coeff = np.dot(self.cov_coeff, yproj)

        self.log.info("Took %0.2f seconds to perform fit." %
                      (time.time() - t0,))

        # Save coefficients
        self.coeff = np.zeros(self.N + (self.nfeature,), dtype=np.float32)
        for ff in range(self.nfeature):
            aa, bb = self._coeff_bounds[ff], self._coeff_bounds[ff+1]
            self.coeff[..., ff] = coeff[aa:bb][self._groups[:, ff]]

        # Save intercept
        if self.fit_intercept:
            self.intercept = coeff[self.ncoeff:].reshape(self.N + (self.nintercept,))
        else:
            self.intercept = np.zeros(self.N + (1,), dtype=np.float32)

        # Compute number of data points that were fit, as well as the model and residual
        self.number = np.sum(self._flag.astype(np.int), axis=-1)

        self.model = self._x.dot(coeff).reshape(self.data.shape)
        self.resid = self.data - self.model

    def refine_model(self, include):

        all_names = []
        for inc in include:

            name, index = self._get_feature_index(inc)

            if index.size > 0:

                model = np.sum(self.coeff[:, np.newaxis, index] * self._base_x[:, :, index], axis=-1)

                attr = '_'.join(['model', name])
                setattr(self, attr, model)
                all_names.append(attr)

                attr = '_'.join(['resid', name])
                setattr(self, attr, self.data - model)
                all_names.append(attr)

        return all_names

    def _get_feature_index(self, include):

        if not isinstance(include, (list, tuple)):
            include = [include]

        index = []
        for inc in include:
            index += [ff for ff, feat in enumerate(self.coeff_name) if feat.startswith(inc)]

        description = '_'.join(include)

        return description, np.unique(index)


def get_pol(sdata, inputmap):

    good_input = np.flatnonzero(np.any(sdata.flags['tau'][:], axis=-1))

    pol_x = np.array([ii for ii, inp in enumerate(inputmap) if tools.is_chime(inp) and
                                                               tools.is_array_x(inp) and
                                                               (ii in good_input)])
    pol_y = np.array([ii for ii, inp in enumerate(inputmap) if tools.is_chime(inp) and
                                                               tools.is_array_y(inp) and
                                                               (ii in good_input)])

    return [pol_x, pol_y]


def get_mezz_and_crate(sdata, inputmap):

    inputs_per_mezz = 8

    # Get location of inputs within correlator
    def cyl_to_corr(idd, inputs):
        sn = sorted(list(inputs['correlator_input']))
        return np.array([sn.index(inputs['correlator_input'][ii]) for ii in idd])

    mezz_index = cyl_to_corr(np.arange(len(inputmap), dtype=np.int), sdata.index_map['input']) // inputs_per_mezz
    crate_index = np.array([inp.crate for inp in inputmap])

    return mezz_index, crate_index


def solve_temp(timestamp, outside_temp, params):

    # Solve for the cable temperature by solving the ODE
    ntime = timestamp.size
    temp = np.zeros(ntime, dtype=outside_temp.dtype)
    temp[0] = outside_temp[0]

    if len(params) > 2:
        planets = ephemeris.skyfield_wrapper.load('de421.bsp')
        obs = ephemeris._get_chime()
        obs.date = timestamp
        alt, _ = obs.altaz(planets['sun'])
        sin_alt = np.sin(alt.radians) * (alt.radians > 0.0)

    for ii in range(ntime-1):
        dydt = params[0] * (temp[ii] - outside_temp[ii])

        if len(params) > 1:
            dydt += params[1] * (temp[ii]**4.0 - outside_temp[ii]**4.0)

        if len(params) > 2:
            dydt += params[2] * sin_alt[ii]

        temp[ii+1] = temp[ii] + dydt*(timestamp[ii+1] - timestamp[ii])

    return temp


def evaluate_derivative(timestamp, temp, flag, order=2):

    index = np.flatnonzero(flag)

    t, c, k = scipy.interpolate.splrep(timestamp[index], temp[index], s=0, k=4)
    spline = scipy.interpolate.BSpline(t, c, k, extrapolate=False)

    print("Evaluating derivative %d." % order)

    deriv = spline.derivative(order)(timestamp)

    return deriv


def ns_distance_dependence(sdata, tdata, inputmap, phase_ref=None, params=None, deriv=0, sep_cyl=False, include_offset=False,
                           sensor='weather_outTemp', temp_field='temp', is_cable_monitor=False, use_alpha=False,
                           **interp_kwargs):

    # Some hardcoded parameters
    unique_source = np.unique(sdata['source'][:])
    nfeature = 1 + int(include_offset) + unique_source.size

    scale = 1e12 * 1e-5 / speed_of_light

    # Interpolate the temperature to the times of measurement
    if is_cable_monitor:

        if use_alpha:
            print("Using cable monitor alpha for NS distance dependence.")
            tempy = np.mean(tdata.alpha[:], axis=0)
        else:
            print("Using cable monitor tau for NS distance dependence.")
            tempy = np.mean(tdata.tau[:], axis=0)

        tempx = tdata.time[:]
        tempf = np.all(tdata.num_freq[:] > 0.0, axis=0)

    else:

        tind = tdata.search_sensors(sensor)

        tempx = tdata.time[:]
        tempy = tdata.datasets[temp_field][tind[0]]
        tempf = tdata.datasets['flag'][tind[0]].astype(np.float32)

        if params is not None:
            trange = np.percentile(np.concatenate((sdata.time[:], sdata['calibrator_time'][:])), [0, 100])
            trange[0], trange[-1] = trange[0] - 3600.0, trange[-1] + 3600.0

            in_range = np.flatnonzero((tempx >= trange[0]) & (tempx <= trange[-1]) & tempf.astype(np.bool))

            tempf = tempf[in_range]
            tempx = tempx[in_range]
            tempy = tempy[in_range]

            tempy = solve_temp(tempx, tempy, np.atleast_1d(params))

        if deriv > 0:
            tempy = evaluate_derivative(tempx, tempy, tempf, order=deriv)

    # Create interpolators
    temp_func = interp1d(tempx, tempy, axis=-1, **interp_kwargs)
    flag_func = interp1d(tempx, tempf, axis=-1, **interp_kwargs)

    # Calculate dependence on source-coordinates/temperature
    def ecoord(dec, ha=0.0):
        lat = np.radians(ephemeris.CHIMELATITUDE)
        return np.cos(lat) * np.sin(np.radians(dec)) - np.sin(lat) * np.cos(np.radians(dec)) * np.cos(np.radians(ha))

    ix = np.zeros((sdata.ntime, nfeature), dtype=np.float32)

    ix[:, 0] = scale * (ecoord(sdata['dec'][:], ha=sdata['ha'][:]) * temp_func(sdata.time[:]) -
                        ecoord(sdata['calibrator_dec'][:], ha=0.0) * temp_func(sdata['calibrator_time'][:]))

    hdep = scale * (ecoord(sdata['dec'][:], ha=sdata['ha'][:]) - ecoord(sdata['dec'][:], ha=0.0))
    for ss, src in enumerate(unique_source):
        this_source = sdata['source'][:] == src
        ix[:, 1+ss] = np.where(this_source, hdep, 0.0)

    if include_offset:
        print("Fitting ns distance for nominal temperature.")
        ix[:, -1] = scale * (ecoord(sdata['dec'][:], ha=0.0) - ecoord(sdata['calibrator_dec'][:], ha=0.0))

    # Determine NS baseline distance and reference appropriately
    feedpos = tools.get_feed_positions(inputmap)
    y = feedpos[:, 1]

    if phase_ref is None:
        y = y - np.nanmean(y)

    else:
        pol = get_pol(sdata, inputmap)
        for ipol, iref in zip(pol, phase_ref):
            y[ipol] = y[ipol] - np.nanmean(y[iref])

    y = np.where(np.isfinite(y), y, 0.0)

    # Scale source-coordinate/temperature dependence by baseline dependence
    xdist = ix[np.newaxis, :, :] * y[:, np.newaxis, np.newaxis]

    # Determine an effective flag
    xdist_flag = (((flag_func(sdata.time[:]) == 1.0) & (flag_func(sdata['calibrator_time'][:]) == 1.0))[np.newaxis, :, np.newaxis] &
                  np.isfinite(feedpos[:, 1, np.newaxis, np.newaxis]) &
                  np.repeat(True, xdist.shape[-1])[np.newaxis, np.newaxis, :])

    # Specify a grouping (assumes all inputs are fit simultaneously)
    grouping = np.zeros((sdata.index_map['input'].size, nfeature), dtype=np.int)
    if sep_cyl:
        is_chime = np.array([tools.is_chime(inp) for inp in inputmap])
        ucyl, cylmap = np.unique([inp.cyl if is_chime[ii] else 100 for ii, inp in enumerate(inputmap)],
                                  return_inverse=True)
        grouping[is_chime, :] = cylmap[is_chime, np.newaxis]

    return xdist, xdist_flag, grouping


def get_timing_correction(sdata, files, set_reference=False, transit_window=2400.0,
                          ignore_amp=True, return_amp=False, **interp_kwargs):

    tcorr = [timing.TimingCorrection.from_acq_h5(tf) for tf in files]

    inputs = tcorr[0].noise_source
    nns = inputs.size
    shp = (nns, sdata.ntime)
    dtype = sdata['tau'].dtype

    ns = np.zeros(shp, dtype=dtype)
    ns_cal = np.zeros(shp, dtype=dtype)
    ns_flag = np.zeros(shp, dtype=np.bool)

    uniq_transits = set(zip(sdata['csd'][:], sdata['source'][:]))
    for cc, src in sorted(uniq_transits):

        this_transit = (sdata['csd'][:] == cc) & (sdata['source'][:] == src)

        timestamp = sdata.time[this_transit]
        timestamp_cal = sdata['calibrator_time'][this_transit]

        # Find the right timing correction
        for tc in tcorr:
            if timestamp[0] >= tc.time[0] and timestamp[-1] <= tc.time[-1]:
                break
        else:
            print(
                "Could not find timing correction file covering "
                "range of timestream data (%s to %s)" %
                tuple(ephemeris.unix_to_datetime([timestamp[0], timestamp[-1]]))
            )
            continue

        if set_reference:
            transit_time = ephemeris.transit_times(ephemeris.source_dictionary[src],
                                                   timestamp[0] - transit_window,
                                                   timestamp[-1] + transit_window)[0]
            tc.set_global_reference_time(transit_time, window=transit_window, **interp_kwargs)

        if return_amp:
            scale = 1.0 if tc.amp_to_delay is None else tc.amp_to_delay
            print("Scaling amplitude by %0.2f" % scale)
            ns[:, this_transit], _ = scale * tc.get_alpha(timestamp, **interp_kwargs)
            ns_cal[:, this_transit], _ = scale * tc.get_alpha(timestamp_cal, **interp_kwargs)

        else:
            ns[:, this_transit], _ = tc.get_tau(timestamp, ignore_amp=ignore_amp, **interp_kwargs)
            ns_cal[:, this_transit], _ = tc.get_tau(timestamp_cal, ignore_amp=ignore_amp, **interp_kwargs)

        ns_flag[:, this_transit] = True

    # Flag missing timing data
    for tc in tcorr:
        dt = np.diff(tc.time)
        dt = dt / np.median(np.abs(dt))
        step = np.flatnonzero(dt > 2.0)
        if step.size > 0:
            for ss in step:

                start_time, end_time = tc.time[ss], tc.time[ss+1]

                this_step = np.flatnonzero(((sdata.time >= start_time) & (sdata.time <= end_time)) |
                                           ((sdata['calibrator_time'][:] >= start_time) &
                                            (sdata['calibrator_time'][:] <= end_time)))

                if this_step.size > 0:
                    ns_flag[:, this_step] = False

    return ns, ns_cal, ns_flag, inputs


def timing_dependence(sdata, tfiles, inputmap, ns_ref=[7, 5], inter_cmn=False,
                      fit_amp=False, ref_amp=False, cmn_amp=True, afiles=None):

    # Some hardcoded parameters
    crates_per_hut = 4
    inputs_per_hut = 1024

    # Some necessary parameters
    pol = get_pol(sdata, inputmap)

    ninput, ntime = sdata['tau'].shape
    dtype = sdata['tau'].dtype

    # Load the timing correction
    ns_tau, ns_tau_cal, ns_flag, ns_inputs = get_timing_correction(sdata, tfiles, set_reference=False,
                                                                   ignore_amp=True, return_amp=False)

    # Load the amplitude correction
    if fit_amp:
        if afiles is None:
            afiles = tfiles

        ns_amp, ns_amp_cal, _, _ = get_timing_correction(sdata, afiles, set_reference=False,
                                                         ignore_amp=True, return_amp=True)

        if cmn_amp:
            cmn_groups = [np.flatnonzero(ns_inputs['chan_id'] < inputs_per_hut),
                          np.flatnonzero(ns_inputs['chan_id'] >= inputs_per_hut)]
            ngroup = len(cmn_groups)

            ns_cmn_amp = np.zeros((ngroup, ntime), dtype=ns_amp.dtype)
            ns_cmn_flag = np.zeros((ngroup, ntime), dtype=np.bool)

            for gg, grp in enumerate(cmn_groups):

                norm = np.sum(ns_flag[grp], axis=0, dtype=np.float32)
                ns_cmn_amp[gg, :] = np.sum(ns_flag[grp] * ns_amp[grp], axis=0) * tools.invert_no_zero(norm)
                ns_cmn_flag[gg, :] = norm > 0.0

    # Get location of inputs within correlator
    mezz_index, crate_index = get_mezz_and_crate(sdata, inputmap)

    mezz_ns = mezz_index[ns_inputs['chan_id']]
    crate_ns = crate_index[ns_inputs['chan_id']]

    # Determine output dimensions
    nns = ns_tau.shape[0]
    ntiming = nns
    if fit_amp:
        if cmn_amp:
            ntiming += ngroup
        else:
            ntiming += nns

    # Set up grouping based on mezzanines
    grouping = np.zeros((ninput, ntiming), dtype=np.int)
    grouping[:, :nns] = mezz_index[:, np.newaxis]
    if fit_amp and not cmn_amp:
        grouping[:, nns:] = mezz_index[:, np.newaxis]

    inter_cmn_count = np.max(mezz_index) + 1

    # Create arrays to hold timing dependence
    xtiming = np.zeros((ninput, ntime, ntiming), dtype=dtype)
    xtiming_flag = np.zeros((ninput, ntime, ntiming), dtype=np.bool)

    # Loop over polarisation groups
    for ipol, iref in zip(pol, ns_ref):

        xtiming[ipol, :, :nns] = np.transpose(ns_tau - ns_tau[iref, np.newaxis, :])[np.newaxis, :, :]
        xtiming_flag[ipol, :, :nns] = np.transpose(ns_flag & ns_flag[iref, np.newaxis, :])[np.newaxis, :, :]

        if fit_amp:
            if ref_amp:
                xtiming[ipol, :, nns:] = np.transpose(ns_amp - ns_amp[iref, np.newaxis, :])[np.newaxis, :, :]
                xtiming_flag[ipol, :, nns:] = np.transpose(ns_flag & ns_flag[iref, np.newaxis, :])[np.newaxis, :, :]
            elif cmn_amp:
                xtiming[ipol, :, nns:] = np.transpose(ns_cmn_amp)[np.newaxis, :, :]
                xtiming_flag[ipol, :, nns:] = np.transpose(ns_cmn_flag)[np.newaxis, :, :]
            else:
                xtiming[ipol, :, nns:] = np.transpose(ns_amp)[np.newaxis, :, :]
                xtiming_flag[ipol, :, nns:] = np.transpose(ns_flag)[np.newaxis, :, :]

        ns_diff_hut = np.flatnonzero((crate_ns // crates_per_hut) != (crate_ns[iref] // crates_per_hut))
        ns_same_hut = np.flatnonzero((crate_ns // crates_per_hut) == (crate_ns[iref] // crates_per_hut))

        is_intra = ipol[np.flatnonzero((crate_index[ipol] // crates_per_hut) == (crate_ns[iref] // crates_per_hut))]
        is_inter = ipol[np.flatnonzero((crate_index[ipol] // crates_per_hut) != (crate_ns[iref] // crates_per_hut))]

        for ndh in ns_diff_hut:
            xtiming[is_intra, :, ndh] = 0.0

            if fit_amp and not cmn_amp:
                xtiming[is_intra, :, nns+ndh] = 0.0

        if fit_amp and cmn_amp:
            xtiming[is_intra, :, nns:] = 0.0

        if inter_cmn:
            for ndh in ns_same_hut:
                grouping[is_inter, ndh] = inter_cmn_count

                if fit_amp and not cmn_amp:
                    grouping[is_inter, nns+ndh] = inter_cmn_count

            inter_cmn_count += 1

    return xtiming, xtiming_flag, grouping


def mean_lna_temp(tdata, field='temp'):

    index = tdata.search_sensors("28.*")

    lna_temp = tdata[field][index, :]
    flag = tdata['flag'][index, :].astype(np.float32)

    sigma = np.nanstd(np.where(flag, lna_temp, np.nan), axis=-1)
    med_sigma = np.median(sigma)
    mad_sigma = 1.48625 * np.median(np.abs(sigma - med_sigma))
    ibad = np.flatnonzero(np.abs(sigma - med_sigma) > (5.0 * mad_sigma))

    if ibad.size > 0:
        print("Flagging %d sensors as bad." % ibad.size)
        flag[ibad, :] = 0.0

    norm = np.sum(flag, axis=0)
    mu_lna_temp = np.sum(flag * lna_temp, axis=0) * tools.invert_no_zero(norm)

    return mu_lna_temp, norm > 0.0


def cable_monitor_dependence(sdata, tdata, include_diff=False, **interp_kwargs):

    # Determine dimensionality
    ninput, ntime = sdata['tau'].shape
    nsource = tdata.nsource

    # Use single flag for all noise source inputs
    flag = np.all(tdata.num_freq[:] > 0.0, axis=0)

    # Calculate the average delay over noise source inputs
    tau = np.mean(tdata.tau[:], axis=0)[:, np.newaxis]

    # If requested, include the difference in delay between adjacent inputs
    if include_diff:
        tau_diff = np.diff(tdata.tau[:], axis=0).T
        tau = np.concatenate((tau, tau_diff), axis=1)

    nseries = tau.shape[1]

    # Create interpolators
    func = interp1d(tdata.time, tau, axis=0, **interp_kwargs)
    flag_func = interp1d(tdata.time, flag.astype(np.float32), **interp_kwargs)

    # Interpolate to the provided times and reference with respect to calibrator transit
    tdep = (func(sdata.time[:]) -
            func(sdata['calibrator_time'][:]))[np.newaxis, :, :]

    tdep = np.repeat(tdep, ninput, axis=0)

    # Generate common flag for all inputs and series
    tflag = ((flag_func(sdata.time[:]) == 1.0) &
             (flag_func(sdata['calibrator_time'][:]) == 1.0))[np.newaxis, :, np.newaxis]

    tflag = np.repeat(tflag, ninput, axis=0)

    # Group by input
    tgroup = np.zeros((ninput, nseries), dtype=np.int)
    for tt in range(nseries):
        tgroup[:, tt] = np.arange(ninput)

    return tdep, tflag, tgroup


def temperature_dependence(sdata, tdata, sensors, field='temp', deriv=0, **interp_kwargs):

    nsensors = len(sensors)

    if np.isscalar(field):
        field = [field] * nsensors

    ninput, ntime = sdata['tau'].shape
    dtype = sdata['tau'].dtype

    # Use lna temperature
    tdep = np.zeros((ninput, ntime, nsensors), dtype=dtype)
    tflag = np.zeros((ninput, ntime, nsensors), dtype=np.bool)
    tgroup = np.zeros((ninput, nsensors), dtype=np.int)

    for tt, (name, grp) in enumerate(zip(sensors, field)):

        if name == 'lna':
            temp_series, temp_flag = mean_lna_temp(tdata, field=grp)

        else:
            tind = tdata.search_sensors(name)[0]
            temp_series = tdata[grp][tind, :]
            temp_flag = tdata['flag'][tind, :]

        if deriv > 0:
            temp_series = evaluate_derivative(tdata.time[:], temp_series, temp_flag, order=deriv)

        temp_func = interp1d(tdata.time, temp_series, axis=-1, **interp_kwargs)
        flag_func = interp1d(tdata.time, temp_flag.astype(np.float32), axis=-1, **interp_kwargs)

        tdep[:, :, tt] = (temp_func(sdata.time[:]) - temp_func(sdata['calibrator_time'][:]))[np.newaxis, :]
        tflag[:, :, tt] = ((flag_func(sdata.time[:]) == 1.0) & (flag_func(sdata['calibrator_time'][:]) == 1.0))[np.newaxis, :]
        tgroup[:, tt] = np.arange(ninput)

    return tdep, tflag, tgroup


def mean_subtract(axes, dataset, flag, use_calibrator=False):

    ninput, ntime = dataset.shape
    weight = flag.astype(dataset.dtype)

    usources = np.unique(axes['source'][:])
    ucsd = np.unique(axes['csd'][:])

    cal = axes.attrs.get('calibrator', 'CYG_A')

    nsources = usources.size
    ncsd = ucsd.size

    dataset_no_mu = dataset.copy()
    mu = np.zeros((ninput, ncsd, nsources), dtype=dataset.dtype)
    mu_flag = np.zeros((ninput, ncsd, nsources), dtype=np.bool)

    for cc, csd in enumerate(ucsd):

        for ss, src in enumerate(usources):

            mcalc = cal if use_calibrator else src
            this_calc = np.flatnonzero((axes['source'][:] == mcalc) & (axes['csd'][:] == csd))

            this_time = np.flatnonzero((axes['source'][:] == src) & (axes['csd'][:] == csd))

            if (this_calc.size > 0) and (this_time.size > 0):
                norm = np.sum(weight[:, this_calc], axis=-1)
                this_mu = np.sum(weight[:, this_calc] * dataset[:, this_calc], axis=-1) * tools.invert_no_zero(norm)

                dataset_no_mu[:, this_time] -= this_mu[:, np.newaxis]
                mu[:, cc, ss] = this_mu
                mu_flag[:, cc, ss] = norm > 0.0

    return dataset_no_mu, mu, mu_flag


def short_long_stat(axes, dataset, flag, stat='mad', ref_common=False, pol=None):

    ninput, ntime = dataset.shape
    weight = flag.astype(dataset.dtype)

    usources = list(set([pair for pair in zip(axes['calibrator'][:], axes['source'][:])]))
    ucsd = np.unique(axes['csd'][:])

    nsources = len(usources)
    ncsd = ucsd.size

    dataset_no_mu = dataset.copy()

    if ref_common and (pol is not None):
        npol = len(pol)
        for pp, ipol in enumerate(pol):
            cmn = np.nanmedian(np.where(flag[ipol], dataset[ipol], np.nan), axis=0, keepdims=True)
            dataset_no_mu[ipol, :] = dataset_no_mu[ipol, :] - cmn

    mu = np.zeros((ninput, ncsd, nsources), dtype=dataset.dtype)
    mu_flag = np.zeros((ninput, ncsd, nsources), dtype=np.bool)
    time_diff = np.zeros((ncsd, nsources), dtype=np.float32)

    for cc, csd in enumerate(ucsd):

        for ss, (cal, src) in enumerate(usources):

            this_time = np.flatnonzero((axes['source'][:] == src) & (axes['calibrator'][:] == cal) & (axes['csd'][:] == csd))

            if this_time.size > 0:
                norm = np.sum(weight[:, this_time], axis=-1)
                this_mu = (np.sum(weight[:, this_time] * dataset_no_mu[:, this_time], axis=-1) *
                           tools.invert_no_zero(norm))

                dataset_no_mu[:, this_time] -= this_mu[:, np.newaxis]
                mu[:, cc, ss] = this_mu
                mu_flag[:, cc, ss] = norm > 0.0

                csd_cal = np.mean(axes['calibrator_time'][this_time])
                time_diff[cc, ss] = np.mean(axes.time[this_time]) - np.mean(axes['calibrator_time'][this_time])

    nan_short = np.where(flag, dataset_no_mu, np.nan)
    nan_long = np.where(mu_flag, mu, np.nan)

    non_cal = np.array([ii for ii, pair in enumerate(usources) if pair[0] != pair[1]])
    nan_long_noncal = np.where(mu_flag[:, :, non_cal], mu[:, :, non_cal], np.nan)

    res = {}

    # Save the mean value for each CSD/source pair
    res['mu'] = mu
    res['mu_flag'] = mu_flag

    # Save the source pair axes and the time offset between them
    res['source_pair'] = np.array(["%s/%s" % pair for pair in usources])
    res['time_diff'] = time_diff

    # Save the other axes
    res['input'] = axes.index_map['input'][:]
    res['csd'] = ucsd

    # Save the number of data points used to calculate statistics
    res['num_short'] = np.sum(flag, axis=1)
    res['num_long'] = np.sum(mu_flag, axis=(1, 2))
    res['num_long_by_source'] = np.sum(mu_flag, axis=1)

    # Calculate the requested statistics
    if stat == 'mad':
        med = np.nanmedian(nan_short, axis=1)
        res['short'] = 1.48625 * np.nanmedian(np.abs(nan_short - med[:, np.newaxis]), axis=1)

        med = np.nanmedian(nan_long_noncal, axis=(1, 2))
        res['long'] = 1.48625 * np.nanmedian(np.abs(nan_long_noncal - med[:, np.newaxis, np.newaxis]), axis=(1, 2))

        med = np.nanmedian(nan_long, axis=1)
        res['long_by_source'] =  1.48625 * np.nanmedian(np.abs(nan_long - med[:, np.newaxis, :]), axis=1)

    else:
        res['short'] = np.nanstd(nan_short, axis=1)
        res['long'] = np.nanstd(nan_long_noncal, axis=(1, 2))
        res['long_by_source'] = np.nanstd(nan_long, axis=1)

    return res


def apply_kz_lpf(x, y, w=20.0, k=4):

    window = w * 60.0 / np.median(np.abs(np.diff(x)))
    window = int(window)
    window += (not (window %2))

    total = k * (window - 1) + 1
    hwidth = total // 2

    shp = y.shape[:-1]

    nan_matrix = np.full(shp + (hwidth,), np.nan, dtype=y.dtype)
    nan_y = np.concatenate((nan_matrix, y, nan_matrix), axis=-1)

    y_lpf = np.zeros_like(y)
    for ii in np.ndindex(*shp):
        y_lpf[ii] = kzfilt.kz_filter(nan_y[ii], window, k)

    return y_lpf



class StreamMedian:
    def __init__(self):
        self.minHeap, self.maxHeap = [], []
        self.N=0

    def insert(self, num):
        if self.N%2==0:
            heapq.heappush(self.maxHeap, -1*num)
            self.N+=1
            if len(self.minHeap)==0:
                return
            if -1*self.maxHeap[0]>self.minHeap[0]:
                toMin=-1*heapq.heappop(self.maxHeap)
                toMax=heapq.heappop(self.minHeap)
                heapq.heappush(self.maxHeap, -1*toMax)
                heapq.heappush(self.minHeap, toMin)
        else:
            toMin=-1*heapq.heappushpop(self.maxHeap, -1*num)
            heapq.heappush(self.minHeap, toMin)
            self.N+=1

    def getMedian(self):
        if self.N%2==0:
            return (-1*self.maxHeap[0]+self.minHeap[0])/2.0
        else:
            return -1*self.maxHeap[0]


def cummedian_stream(arr, flag=None, group_index=None,
                     rng=[0.001, 0.101], bins=100, no_weight=True,
                     nsigma=5.0, nstart=2, nscale=6):

    nfreq, ninput, ntime = arr.shape

    if flag is None:
        flag = np.ones((nfreq, ninput, ntime), dtype=np.bool)

    if group_index is None:
        group_index = [np.arange(ninput)]

    ngroup = len(group_index)

    if nscale is None:
        nscale = ntime

    # Create containers
    any_scale = np.zeros((ninput,), dtype=np.bool)
    any_flag = np.zeros((nfreq, ninput), dtype=np.bool)

    time_flag = np.ones((ninput, ntime), dtype=np.bool)

    delta_arr = np.zeros((ninput, ntime), dtype=np.float32)
    mu_delta = np.zeros((ngroup, ntime), dtype=np.float32)
    sig_delta = np.zeros((ngroup, ntime), dtype=np.float32)

    stream = [[StreamMedian()] * ninput] * nfreq

    scale_stream = [StreamMedian()] * ninput

    # Loop over times
    for tt in range(0, ntime):

        # If this isn't the first time, then calculate the median of past good values
        # for each frequency and input
        if tt >= nstart:

            median = np.array([[si.getMedian() for si in sf] for sf in stream])

            flag_tt = flag[..., tt] & any_flag
            darr = np.abs(arr[..., tt] - median)

            for ii in range(ninput):
                this_flag = flag_tt[:, ii]
                if np.any(this_flag):
                    delta_arr[ii, tt] = 1.48625 * wq.median(darr[:, ii], this_flag.astype(np.float32))

            delta = delta_arr[:, tt]

            for gg, gindex in enumerate(group_index):

                delta_group = delta[gindex].copy()

                if tt > nscale:
                    for gi, ii in enumerate(gindex):
                        if any_scale[ii] and (delta_group[gi] > 0.0):
                            delta_group[gi] -= scale_stream[ii].getMedian()

                try:
                    res = cal_utils.fit_histogram(delta_group[delta_group > 0.0],
                                                  bins=bins, rng=rng, no_weight=no_weight,
                                                  test_normal=False, return_histogram=False)

                except Exception as exc:

                    time_flag[gindex, tt] = False

                else:

                    mu_delta[gg, tt] = res['par'][1]
                    sig_delta[gg, tt] = res['par'][2]

                    time_flag[gindex, tt] = ((delta_group > 0.0) &
                                             ((delta_group - res['par'][1]) < (nsigma * res['par'][2])))


                    for gi, ii in enumerate(gindex):
                        if time_flag[ii, tt]:
                            scale_stream[ii].insert(delta[ii] - res['par'][1])
                            any_scale[ii] = True

        # Loop over frequencies and inputs for this time.  If this time showed
        # low scatter and was not flagged previously, then add it to the
        # appropriate heap for calculating the median
        for (ff, ii), val in np.ndenumerate(arr[..., tt]):

            if flag[ff, ii, tt] and time_flag[ii, tt]:

                stream[ff][ii].insert(val)

                any_flag[ff, ii] = True

    # Get the final median
    median = np.array([[si.getMedian() for si in sf] for sf in stream])

    # Return median over time, mad over frequency, and time flag
    return median, time_flag, delta_arr, mu_delta, sig_delta
