import numpy as np

def sliding_window(arr, window):

    # Advanced numpy tricks
    shape = arr.shape[:-1] + (arr.shape[-1]-window+1, window)
    strides = arr.strides + (arr.strides[-1],)
    return np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides)


def _kz_coeffs(m, k):

    # Coefficients at degree one
    coef = np.ones(m)

    # Iterate k-1 times over coefficients
    for i in range(1, k):

        t = np.zeros((m, m+i*(m-1)))
        for km in range(m):
            t[km, km:km+coef.size] = coef

        coef = np.sum(t, axis=0)

    assert coef.size == k*(m-1)+1

    return coef/m**k


def _kz_prod(data, coef, m, k, t=None):

    n = data.size
    data = sliding_window(data, k*(m-1)+1)
    assert data.shape == (n-k*(m-1), len(coef))

    # Restrict KZ product calculation to provided indices
    if t is not None:
        data = data[t]
        assert data.shape == (len(t), len(coef))

    return data*coef


def _kz_sum(data, coef):

    knan = np.isnan(data)

    # Handle missing values if any
    if np.any(knan):

        coef = np.ma.MaskedArray(np.broadcast_to(coef[np.newaxis, :], data.shape), mask=knan)
        coef = np.sum(coef, axis=-1)

        data = np.nansum(data, axis=-1)

        # Restore nan were data are missing
        data[coef.mask] = np.nan

        # Divide by coefficients sum, which may not be 1
        k = np.logical_not(coef.mask)
        data[k] = data[k]/coef[k]

        return data

    else:
        return np.sum(data, axis=-1)


def kz_filter(data, m, k):
    """Kolmogorov-Zurbenko fitler
    :param numpy.ndarray data: A 1-dimensional numpy array of size `N`. Any missing value should be set to ``np.nan``.
    :param int m: Filter window width.
    :param int k: Filter degree.
    :return: A :class:`numpy.ndarray` of size `N-k*(m-1)`
    Given a time series :math:`X_t, t \\in \\{0, 1, \\cdots, N-1\\}`, the Kolmogorov-Zurbenko fitler is defined for
    :math:`t \\in \\{\\frac{k(m-1)}{2}, \\cdots, N-1-\\frac{k(m-1)}{2}\\}` by
    .. math::
        KZ_{m,k}[X_t] = \\sum_{s=-k(m-1)/2}^{k(m-1)/2} \\frac{a_s^{m,k}}{m^k} \\cdot X_{t+s}
    Definition of coefficients :math:`a_s^{m,k}` is given in :func:`_kz_coeffs`.
    """

    coef = _kz_coeffs(m, k)
    data = _kz_prod(data, coef, m, k)

    return _kz_sum(data, coef)


def kzft(data, nu, m, k, t=None, dt=1.):
    """Kolmogorov-Zurbenko Fourier transform filter
    :param numpy.ndarray data: A 1-dimensional numpy array of size `N`. Any missing value should be set to ``np.nan``.
    :param list-like nu: Frequencies, length `Nnu`.
    :param int m: Filter window width.
    :param int k: Filter degree.
    :param list-like t: Calculation indices, of length `Nt`. If provided, KZFT filter will be calculated only for values
      ``data[t]``. Note that the KZFT filter can only be calculated for indices in the range [k(m-1)/2, (N-1)-k(m-1)/2].
      Trying to calculate the KZFT out of this range will raise an `IndexError`. `None`, calculation will happen over
      the whole calculable range.
    :param float dt: Time step, if not 1.
    :return: A :class:`numpy.ndarray` of shape `(Nnu, Nt)` or `(Nnu, N-k(m-1))` if `t` is `None`.
    :raise IndexError: If `t` contains one or more indices out of the calculation range. See documentation of keyword
      argument `t`.
    Given a time series :math:`X_t, t \\in \\{0, 1, \\cdots, N-1\\}`, the Kolmogorov-Zurbenko Fourier transform filter
    is defined for :math:`t \\in \\{\\frac{k(m-1)}{2}, \\cdots, N-1-\\frac{k(m-1)}{2}\\}` by

    .. math::
        KZFT_{m,k,\\nu}[X_t] = \\sum_{s=-k(m-1)/2}^{k(m-1)/2} \\frac{a_s^{m,k}}{m^k} \\cdot X_{t+s} \\cdot
        e^{-2\\pi i\\nu s}
    """

    if not dt == 1.:
        nu = np.asarray(nu)*dt
        m = int((m-1)/dt+1)
        if not m%2:
            m += 1

    if t is not None:
        w = int(k*(m-1)/2)
        t = np.asarray(t)-w
        if np.any(t < 0) or np.any(t > (data.size-1-2*w)):
            raise IndexError('Inpunt calculation indices are out of range. Calculation indices should be in the range '
                             '[k*(m-1)/2, (N-1)-k*(m-1)/2], hence [{}, {}] in the present case.'
                             .format(w, data.size-1-w))

    coef = _kz_coeffs(m, k)
    data = _kz_prod(data, coef, m, k, t=t)

    nu = np.asarray(nu)
    s = k*(m-1)/2
    s = np.arange(-s, s+1)
    s = np.exp(-1j*2*np.pi*nu[:, np.newaxis]*s)

    data = data[np.newaxis]*s[:, np.newaxis]

    return _kz_sum(data, coef)


def kzp(data, nu, m, k, dt=1.):
    """Kolmogorov-Zurbenko periodogram
    :param numpy.ndarray data: A 1-dimensional numpy array of size `N`. Any missing value should be set to ``np.nan``.
    :param list-like nu: Frequencies, length `Nnu`.
    :param int m: Filter window width.
    :param int k: Filter degree.
    :param float dt: Time step, if not 1.
    :return: A :class:`numpy.ndarray` os size `Nnu`.
    Given a time series :math:`X_t, t \\in \\{0, 1, \\cdots, N-1\\}`, the Kolmogorov-Zurbenko periodogram is defined by
    .. math::
        KZP_{m,k}(\\nu) = \\sqrt{\\sum_{h=0}^{T-1} \\lvert 2 \\cdot  KZFT_{m,k,\\nu}[X_{hL+k(m-1)/2}] \\rvert ^2}
    where :math:`L=(N-w)/(T-1)` is the distance between the beginnings of two successive intervals, :math:`w` being the
    calculation window width of the :func:`kzft` and :math:`T` the number of intervals.
    The assumption was made that :math:`L \\ll w \\ll N`, implying that the intervals overlap.
    """

    if not dt == 1.:
        nu = nu*dt
        m = int((m-1)/dt+1)
        if not m%2:
            m += 1

    # w is the width of the KZFT. As m is odd, k*(m-1) is always even, so w is always odd.
    w = k*(m-1)+1

    # Distance between two successve intervals
    l = int(m/10)
    nt = int((data.size-w)/l+1)

    # Calculation indices
    l = np.arange(nt-1)*l+k*(m-1)/2
    l = np.floor(l).astype(int)

    return np.sqrt(np.nanmean(np.square(2*np.abs(kzft(data, nu, m, k, t=l))), axis=-1))


def kznotch(data, weight, nu, dt=1.0, m=5, k=3):

    # Make sure we have a separate parameter for each frequency
    nu = _ensure_list(nu)
    m = _ensure_list(m)
    k = _ensure_list(k)

    nfreq = nu.size
    ndata = data.size

    dtype = data.dtype
    wtype = np.float64

    if m.size not in [1, nfreq]:
        ValueError("Keyword m has size %d.  Must have size 1 or %d." % (m.size, nfreq))
    elif m.size == 1:
        m = np.repeat(m, nfreq)

    if k.size not in [1, nfreq]:
        ValueError("Keyword k has size %d.  Must have size 1 or %d." % (k.size, nfreq))
    elif k.size == 1:
        k = np.repeat(k, nfreq)

    # Make sure we have odd number of elements in window
    for ii in range(nfreq):
        m[ii] += not (m[ii] % 2)

    # Find unique windows
    param = zip(m, k)
    uparam = list(set(param))

    # Create array to fill in
    yft = np.zeros((nfreq, ndata), dtype=dtype)

    # Loop over frequencies
    for par in uparam:

        ifreq = np.array([nn for nn in range(nfreq) if param[nn] == par])

        width = par[1] * (par[0] - 1) + 1
        hwidth = width / 2

        y = np.concatenate((np.zeros(hwidth, dtype=dtype),
                            data,
                            np.zeros(hwidth, dtype=dtype)))

        w = np.concatenate((np.zeros(hwidth, dtype=wtype),
                           (weight > 0.0).astype(wtype),
                            np.zeros(hwidth, dtype=wtype)))

        y = sliding_window(y, width)
        w = sliding_window(w, width)

        coeff = _kz_coeffs(*par)

        wcoeff = w * coeff[np.newaxis, :]
        wcoeff *= tools.invert_no_zero(np.sum(wcoeff, axis=-1))[:, np.newaxis]

        s = np.arange(-hwidth, hwidth + 1)
        s = np.exp(-2.0J * np.pi * nu[ifreq, np.newaxis] * dt * s[np.newaxis, :])

        # Save to array
        yft[ifreq, :] = np.sum((wcoeff * y)[np.newaxis] * s[:, np.newaxis, :], axis=-1)

    # Return filtered values
    return yft


def _ensure_list(x):

    if not hasattr(x, '__iter__'):
        x = [x]

    x = np.array(x)

    return x
