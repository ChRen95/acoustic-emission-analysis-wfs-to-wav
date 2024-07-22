import numpy as np


class Events(np.ndarray):
    def __new__(cls, data, **kwargs):
        obj = np.asarray(data, dtype=object).view(cls)
        return obj

    def __init__(self, data, source=None, thresh=None, pre=None, hdt=None, dead=None):
        self.source = source
        self.thresh = thresh
        self.pre = pre
        self.hdt = hdt
        self.dead = dead

    @property
    def starts(self):
        return np.array([e.start for e in self]) * self.source.timescale

    @property
    def durations(self):
        return np.array([e.duration for e in self]) * self.source.timescale

    @property
    def energies(self):
        return np.array([e.energy for e in self]) * self.source.timescale

    @property
    def maxima(self):
        return np.array([e.max for e in self])

    @property
    def rise_times(self):
        return np.array([e.rise_time for e in self]) * self.source.timescale

    @property
    def counts(self):
        return np.array([e.count(self.thresh) for e in self])

    def psd(self, **kwargs):
        Pxxs = np.array([e.psd(**kwargs)[1] for e in self])
        f = self[0].psd(**kwargs)[0]
        return f, Pxxs
