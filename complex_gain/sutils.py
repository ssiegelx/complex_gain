import heapq

import numpy as np
import weighted as wq

from ch_util import cal_utils

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

            print tt,

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


