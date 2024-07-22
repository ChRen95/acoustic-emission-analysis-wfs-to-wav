import time
import warnings
import numpy as np
from abc import ABC, abstractmethod
from tqdm import tqdm
from math import floor, ceil

from ..events import Events
from ..event_detector import process_block


class BaseData(ABC):
    def __init__(self, fname, block_dtype, datascale, timescale):
        self.fname = fname
        self.block_dtype = block_dtype
        self.datascale = datascale
        self.timescale = timescale
        self.dtype = None
        self.shape = None
        self.size = None
        self.channels = None
        self._min_max_cache = {}

    def __repr__(self):
        return f"{self.__class__.__name__}({self.fname!r})"

    @abstractmethod
    def raw_iter_blocks(self, start, stop):
        pass

    @abstractmethod
    def check_block(self, pos, raw):
        pass

    @abstractmethod
    def get_block_data(self, raw):
        pass

    @abstractmethod
    def progress(self, percent, elapsed_time):
        pass

    def iterate_blocks(self, start=0, stop=float('inf'), channel=slice(None)):
        start_time = time.time()
        for pos, raw in self.raw_iter_blocks(start, stop):
            self.check_block(pos, raw)
            yield pos, self.get_block_data(raw)[..., channel]
            self.progress(100.0 * pos / self.size, time.time() - start_time)
        self.progress(100, time.time() - start_time)

    def calculate_sizes(self, file_size):
        n_blocks = file_size // self.block_dtype.itemsize
        rest = file_size % self.block_dtype.itemsize

        if rest:
            warnings.warn(f"{rest} bytes at the end of the file won't fit into blocks")

        tmp = self.get_block_data(np.empty(0, self.block_dtype))
        self.dtype = tmp.dtype
        self.shape = (n_blocks,) + tmp.shape[1:-1]
        self.size = np.prod(self.shape)
        self.channels = tmp.shape[-1]

        assert self.channels == len(self.datascale)

    def get_min_max(self, channel=0):
        if channel in self._min_max_cache:
            return self._min_max_cache[channel]

        cachefn = f"{self.fname}.envelope.cache.{channel}.npz"
        try:
            d = np.load(cachefn)
            mins, maxs = d['mins'], d['maxs']
            self._min_max_cache[channel] = (mins, maxs)
            print("# read min/max from cache")
        except (OSError, KeyError):
            print("# calculating min/max envelope")

            mins, maxs = np.inf, -np.inf
            for pos, raw in tqdm(self.iterate_blocks(channel=channel)):
                data = self.get_block_data(raw)
                mins = np.minimum(mins, np.nanmin(data, axis=0))
                maxs = np.maximum(maxs, np.nanmax(data, axis=0))

            self._min_max_cache[channel] = (mins, maxs)
            np.savez_compressed(cachefn, mins=mins, maxs=maxs)

        return mins, maxs

    def resample(self, range, channel=0, num=768):
        def clip(x, a, b):
            return min(max(x, a), b)

        a, b = range
        a = int(floor(a / self.timescale))
        b = int(ceil(b / self.timescale)) + 1
        s = max((b - a) // num, 1)

        a = clip(a, 0, self.size)
        b = clip(b, 0, self.size)

        r = self.shape[-1]
        if s > r:
            s //= r
            a //= r
            b //= r
            mins, maxs = self.get_min_max(channel)
            mins = mins[a // s * s:b // s * s]
            mins.shape = (b // s - a // s, s)
            maxs = maxs[a // s * s:b // s * s]
            maxs.shape = (b // s - a // s, s)
            s *= r
            a *= r
            b *= r
        else:
            blocks = []
            for pos, d in self.iterate_blocks(start=a // s * s, stop=b // s * s, channel=channel):
                aa = clip(a // s * s - pos, 0, d.size)
                bb = clip(b // s * s - pos, 0, d.size)
                blocks.append(d.flat[aa:bb])
            d = np.concatenate(blocks)
            d.shape = (d.size // s, s)
            mins = maxs = d

        x = np.empty(2 * mins.shape[0])
        y = np.empty(2 * mins.shape[0])
        x[::2] = x[1::2] = np.arange(a // s * s, b // s * s, s) * self.timescale

        mins.min(axis=-1, out=y[::2])
        maxs.max(axis=-1, out=y[1::2])
        y *= self.datascale[channel]
        return x, y

    def get_events(self, thresh, hdt=0.001, dead=0.001, pretrig=0.001, channel=0, limit=0):
        raw_thresh = int(thresh / self.datascale[channel])
        raw_hdt = int(hdt / self.timescale)
        raw_pre = int(pretrig / self.timescale)
        raw_dead = int(dead / self.timescale)
        raw_limit = int(limit / self.timescale)

        def _get_event(start, stop, pos, prev_data, data):
            a = start - raw_pre - pos
            b = stop + raw_hdt - pos
            datascale = self.datascale[channel]

            assert a < b, (a, b)

            if a < 0:
                assert a >= -prev_data.size, (a, prev_data.size)
                if b < 0:
                    ev_data = prev_data[a:b] * datascale
                else:
                    assert b <= data.size, (b, data.size)
                    ev_data = np.concatenate((prev_data[a:], data.flat[:b])) * datascale
            else:
                if b < data.size:
                    ev_data = data.flat[a:b] * datascale
                else:
                    assert a <= data.size, (a, data.size)
                    ev_data = np.concatenate((data.flat[a:], np.zeros(b - data.size, dtype=data.dtype))) * datascale

            assert ev_data.size == raw_pre + stop - start + raw_hdt
            return Events(start, stop, ev_data)

        def _add_event(*args):
            try:
                events.append(_get_event(*args))
            except Exception:
                import traceback
                traceback.print_exc()

        last = None
        events = []
        prev_data = np.zeros(raw_pre, dtype=self.dtype)

        for pos, data in self.iterate_blocks(channel=channel):
            ev, last = process_block(data.astype("i2"), raw_thresh, hdt=raw_hdt, dead=raw_dead, event=last, pos=pos, limit=raw_limit)
            for start, stop in ev:
                _add_event(start, stop, pos, prev_data, data)
            start = last[0] - pos if last else 0
            prev_data = data.flat[start - raw_pre:]
        if last:
            _add_event(last[0], last[1], pos, None, data)

        return Events(source=self, thresh=thresh, pre=raw_pre, hdt=raw_hdt, dead=raw_dead, data=events)
