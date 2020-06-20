import re

import numpy as np

from ch_util import andata

class TempData(andata.BaseData):
    
    convert_attribute_strings = False
    convert_dataset_strings = False

    @property
    def sensor(self):
        return self.index_map['sensor']

    @property
    def nsensor(self):
        return self.index_map['sensor'].size

    @property
    def temp(self):
        return self['temp']

    @property
    def flag(self):
        return self['flag']

    @property
    def flag_acq(self):
        return self['flag'] & self['data_flag'][np.newaxis, :]

    def search_sensors(self, pattern, is_regex=False):
        """Find the index into the `sensor` axis corresponding to a desired pattern.

        Parameters
        ----------
        pattern : str
            The desired `sensor` or a glob pattern to search.
        is_regex : bool
            Set to True if `pattern` is a regular expression.

        Returns
        -------
        index : `np.ndarray` of `dtype = int`
            Index into the `sensor` axis that will yield all
            sensors matching the requested pattern.
        """
        import fnmatch

        ptn = pattern if is_regex else fnmatch.translate(pattern)
        regex = re.compile(ptn)
        index = np.array([ii for ii, ss in enumerate(self.sensor[:].astype(str)) if regex.match(ss)], dtype=np.int)
        return index

    def group_name_allowed(self, name):
        return True

    def dataset_name_allowed(self, name):
        return True