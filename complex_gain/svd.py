import numpy as np
import pytz

from ch_util import ephemeris
from ch_util.fluxcat import FluxCatalog

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

plt.rc('font', family='sans-serif', weight='normal', size=20)

COLORMAP = matplotlib.cm.__dict__["viridis"]
COLORMAP.set_under("black")
COLORMAP.set_bad("lightgray")


def perform_svd(data, source=None, input_threshold=0.80, transit_threshold=0.85,
                      niter=10, rank=16, nsigma=5.0, bad_csd=None):

    if source is not None:
        this_source = np.flatnonzero(data['source'][:] == source)

        X = data['tau'][:, this_source].T
        flag = data.flags['tau'][:, this_source].T

    else:

        X = data['tau'][:].T
        flag = data.flags['tau'][:].T

    if bad_csd is not None:
        good_csd = np.array([xx not in bad_csd for xx in data['csd']])
    else:
        good_csd = np.ones(X.shape[0], dtype=np.bool)

    mad_X = 1.48625 * np.median(np.abs(X[flag]))
    flag = flag & (np.abs(X) < (nsigma * mad_X))

    good_input = np.flatnonzero((np.sum(flag, axis=0) / float(flag.shape[0])) > input_threshold)
    good_transit = np.flatnonzero(good_csd &
                                  ((np.sum(flag[:, good_input], axis=1) / float(good_input.size)) > transit_threshold))

    results = {}
    results['good_input'] = good_input
    results['good_transit'] = good_transit

    X = X[good_transit][:, good_input]
    flag = flag[good_transit][:, good_input]

    weight = flag.astype(np.float32)
    mu_X = np.sum(weight * X, axis=0) * tools.invert_no_zero(np.sum(weight, axis=0))
    X = X - mu_X[np.newaxis, :]

    mask = ~flag
    X[mask] = 0.0

    itt = 0
    while itt < niter:

        print("Iteration %d" % itt)

        U, S, VH = np.linalg.svd(X.copy(), full_matrices=False)

        S_tr = np.zeros_like(S)
        S_tr[:rank] = S[:rank]

        X[mask] = np.dot(S_tr * U, VH)[mask]

        mu_X = np.sum(X, axis=0) / float(X.shape[0])
        X = X - mu_X[np.newaxis, :]

        itt += 1

    results['X'] = X
    results['U'] = U
    results['S'] = S
    results['VH'] = VH

    return results


def svd_panels(data, results, rank, cmap=COLORMAP, vrng=None, nticks=20, cyl_size=256,
                                    input_range=None, input_map=None, corr_order=False,
                                    zoom_date=None, timezone='Canada/Pacific', phys_unit=True):

    inputs = data.index_map['input'][:]
    ninput = inputs.size

    timestamp = data.time[:]
    ntime = timestamp.size

    good_input = results['good_input']
    good_transit = results['good_transit']

    if (input_map is not None) and corr_order:
        cord = np.array([inp.corr_order for inp in input_map])
        cord = cord[good_input]
        isort = np.argsort(cord)
        cttl = "Correlator Order"
    else:
        cord = good_input
        isort = np.argsort(cord)
        cttl = "Feed Number"

    if input_range is None:
        input_range = [0, ninput]

    tfmt = "%b-%d"
    tz = pytz.timezone(timezone)

    skip = ntime // nticks
    xtck = np.arange(0, ntime, skip, dtype=np.int)
    xtcklbl= [tt.astimezone(tz).strftime(tfmt) for tt in ephemeris.unix_to_datetime(timestamp[::skip])]

    ytck = np.arange(ninput // cyl_size + 1) * cyl_size

    if zoom_date is None:
        zoom_date = []
    else:
        zoom_date = [list(ephemeris.datetime_to_unix(zd)) for zd in zoom_date]

    zoom_date = [[timestamp[0], timestamp[-1]]] + zoom_date
    nzoom = len(zoom_date)

    gs = gridspec.GridSpec(2 + nzoom // 2, 2, height_ratios=[3, 2] + [2] * (nzoom // 2), hspace=0.30)

    this_rank = np.outer(results['S'][rank] * results['U'][:, rank], results['VH'][rank, :])

    var = results['S']**2
    exp_var = 100.0 * var / np.sum(var)
    exp_var = exp_var[rank]

    if vrng is None:
        vrng = np.nanpercentile(this_rank, [2, 98])

    y = np.full((ninput, ntime), np.nan, dtype=np.float32)
    for ii, gi in enumerate(cord):
        y[gi, good_transit] = this_rank[:, ii]

    plt.subplot(gs[0, :])

    img = plt.imshow(y, aspect='auto', origin='lower', interpolation='nearest',
                     extent=(0, ntime, 0, ninput), vmin=vrng[0], vmax=vrng[1], cmap=cmap)

    cbar = plt.colorbar(img)
    cbar.ax.set_ylabel(r'$\tau$ [picosec]')

    plt.ylabel(cttl)
    plt.title("k = %d | Explains %0.1f%% of Variance" % (rank, exp_var))

    plt.xticks(xtck, xtcklbl, rotation=70.0)
    plt.yticks(ytck)
    plt.ylim(input_range)

    if phys_unit:
        scale = 1.48625 * np.median(np.abs(results['S'][rank] * results['U'][:, rank]))
        lbl = r"v$_{%d}$  x MED$(|\sigma_{%d} u_{%d}|)$ [picosec]" % (rank, rank, rank)
    else:
        scale = 1.0
        lbl = r"v$_{%d}$" % rank

    plt.subplot(gs[1, 0])
    plt.plot(cord, scale * results['VH'][rank, :], color='k', marker='.', linestyle='None')
    plt.grid()
    plt.xticks(ytck, rotation=70)
    plt.xlim(input_range)
    plt.xlabel(cttl)
    plt.ylabel(lbl)

    scolors = ['b', 'r', 'forestgreen', 'magenta', 'orange']
    list_good = list(results['good_transit'])

    if phys_unit:
        scale  = 1.48625 * np.median(np.abs(results['VH'][rank, :]))
        lbl = r"$\sigma_{%d}$ u$_{%d}$  x MED$(|v_{%d}|)$ [picosec]" % (rank, rank, rank)
    else:
        scale = 1.0
        lbl = r"$\sigma_{%d}$ u$_{%d}$" % (rank, rank)

    classification = np.char.add(np.char.add(data['calibrator'][:], '/'), data['source'][:])
    usource, ind_source = np.unique(classification, return_inverse=True)
    nsource = usource.size

    for zz, (start_time, end_time) in enumerate(zoom_date):

        row = (zz + 1) // 2 + 1
        col = (zz + 1) % 2

        plt.subplot(gs[row, col])

        this_zoom = np.flatnonzero((data.time >= start_time) & (data.time <= end_time))
        nsamples = this_zoom.size

        skip = nsamples // (nticks // 2)

        xtck = np.arange(0, nsamples, skip, dtype=np.int)
        ztfmt = "%b-%d %H:%M" if zz > 0 else tfmt
        xtcklbl = np.array([tt.astimezone(tz).strftime(ztfmt)
                            for tt in ephemeris.unix_to_datetime(data.time[this_zoom[xtck]])])

        for ss, src in enumerate(usource):

            this_source = np.flatnonzero(ind_source[this_zoom] == ss)
            if this_source.size == 0:
                continue

            this_source = np.array([ts for ts in this_source if this_zoom[ts] in list_good])
            this_source_good = np.array([list_good.index(ind) for ind in this_zoom[this_source]])

            plt.plot(this_source, scale * results['S'][rank] * results['U'][this_source_good, rank],
                     color=scolors[ss], marker='.', markersize=4, linestyle='None', label=src)

        plt.xticks(xtck, xtcklbl, rotation=70.0)
        plt.xlim(0, nsamples)

        plt.grid()
        plt.ylabel(lbl)
        plt.legend(prop={'size': 10})