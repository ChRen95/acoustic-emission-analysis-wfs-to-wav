import numpy as np
import random
import warnings
from math import floor, ceil, log, sqrt
from collections import namedtuple, OrderedDict
from matplotlib.ticker import ScalarFormatter
from scipy.signal import welch
import sys
import time
import wave
import types
from itertools import zip_longest, repeat
from collections import OrderedDict
from tqdm import tqdm
import locale




class TimeFormatter(ScalarFormatter):
    def format_data(self, value):
        'return a formatted string representation of a number'
        if self._useLocale:
            s = locale.format_string(self.format, (value,))
        else:
            s = self.format % value
        s = self._formatSciNotation(s)
        return self.fix_minus(s)

    def format_data_short(self, value):
        more = 1
        s = '%1.*f' % (int(self.format[3:-1]) + more, value)
        return s


Event = namedtuple("Event", "start, stop, data")
Event.duration = property(lambda e: e.data.size)
Event.energy = property(lambda e: (e.data ** 2).sum())
Event.max = property(lambda e: e.data.max())
Event.rise_time = property(lambda e: np.argmax(e.data))
Event.count = lambda e, thresh: count(e.data, thresh)
Event.psd = lambda e, **kwargs: welch(e.data, **kwargs)









