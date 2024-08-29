import os
import time
import math
import warnings
import traceback

import numpy as np

from scipy.signal import welch
from collections import namedtuple
from collections import OrderedDict
from struct import unpack_from, calcsize


def count(data, thresh):
    return np.logical_and(data[:-1] < thresh, data[1:] >= thresh).sum()


Event = namedtuple("Event", "start, stop, data")
Event.duration = property(lambda e: e.data.size)
Event.energy = property(lambda e: (e.data ** 2).sum())
Event.max = property(lambda e: e.data.max())
Event.rise_time = property(lambda e: np.argmax(e.data))
Event.count = lambda e, thresh: count(e.data, thresh)

Event.psd = lambda e, **kwargs: welch(e.data, **kwargs)


class Events(np.ndarray):
    extra_attributes = ["source", "thresh", "pre", "hdt", "dead"]

    def __new__(cls, data, **kwargs):
        self = np.ndarray.__new__(cls, len(data), dtype=object)
        self[:] = data
        for a in self.extra_attributes:  # initialize attributes
            setattr(self, a, kwargs.get(a))
        return self

    def __array_finalize__(self, obj):
        if obj is not None:
            for a in self.extra_attributes:  # copy attributes
                setattr(self, a, getattr(obj, a, None))

    starts = property(lambda self: np.array([e.start for e in self]) * self.source.timescale)
    durations = property(lambda self: np.array([e.duration for e in self]) * self.source.timescale)
    energies = property(lambda self: np.array([e.energy for e in self]) * self.source.timescale)
    maxima = property(lambda self: np.array([e.max for e in self]))
    rise_times = property(lambda self: np.array([e.rise_time for e in self]) * self.source.timescale)

    counts = property(lambda self: np.array([e.count(self.thresh) for e in self]))


class Data:
    def __repr__(self):
        return "{}({!r})".format(self.__class__.__name__, self.fname)

    def iter_blocks(self, start=0, stop=float('inf'), channel=slice(None)):

        start_time = time.time()
        for pos, raw in self.raw_iter_blocks(start, stop):
            self.check_block(pos, raw)
            yield pos, self.get_block_data(raw)[..., channel]

    def calc_sizes(self, file_size):
        n_blocks = file_size // self.block_dtype.itemsize
        rest = file_size % self.block_dtype.itemsize
        if rest:
            warnings.warn("{} bytes at the end of file won't fit into blocks".format(rest))

        tmp = self.get_block_data(np.empty(0, self.block_dtype))
        self.dtype = tmp.dtype
        self.shape = (n_blocks,) + tmp.shape[1:-1]
        self.size = np.prod(self.shape)

        self.channels = tmp.shape[-1]
        assert self.channels == len(self.datascale)

    def get_min_max(self, channel=0):
        try:
            return self._min_max_cache[channel]
        except AttributeError:
            self._min_max_cache = {}
        except KeyError:
            pass

        cachefn = self.fname + ".envelope.cache.{}.npz".format(channel)
        try:
            d = np.load(cachefn)
            mins, maxs = d['mins'], d['maxs']
            self._min_max_cache[channel] = (mins, maxs)
            print
            "# envelope in cache"
            return (mins, maxs)
        except:
            pass

        mins = np.empty(self.shape[:-1], dtype=self.dtype)
        maxs = np.empty(self.shape[:-1], dtype=self.dtype)
        block = 0
        for _, d in self.iter_blocks(channel=channel):
            d.min(axis=-1, out=mins[block:block + d.shape[0]])
            d.max(axis=-1, out=maxs[block:block + d.shape[0]])
            block += d.shape[0]
        mins.shape = (mins.size,)
        maxs.shape = (maxs.size,)

        self._min_max_cache[channel] = (mins, maxs)

        np.savez(cachefn, mins=mins, maxs=maxs)

        return mins, maxs

    def resample(self, range, channel=0, num=768):

        def clip(x, a, b):
            return min(max(x, a), b)

        a, b = range
        a = int(math.floor(a / self.timescale))
        b = int(math.ceil(b / self.timescale)) + 1
        s = max((b - a) // num, 1)
        # print "resample", s, b-a

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
            for pos, d in self.iter_blocks(start=a // s * s, stop=b // s * s, channel=channel):
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

            assert a < b, (a, b)  # sanity

            if a < 0:
                assert a >= -prev_data.size, (a, prev_data.size)
                if b < 0:
                    # whole event in prev_data, we waited only for dead time 
                    ev_data = prev_data[a:b] * datascale
                else:
                    # part in prev_data part in data
                    assert b <= data.size, (b, data.size)
                    ev_data = np.concatenate((
                        prev_data[a:],
                        data.flat[:b]
                    )) * datascale
            else:
                if b < data.size:
                    # all in data
                    ev_data = data.flat[a:b] * datascale
                else:
                    # pad with zeros
                    assert a <= data.size, (a, data.size)
                    ev_data = np.concatenate((
                        data.flat[a:],
                        np.zeros(b - data.size, dtype=data.dtype)
                    )) * datascale

            assert ev_data.size == raw_pre + stop - start + raw_hdt
            return Event(start, stop, ev_data)

        def _add_event(*args):
            try:
                events.append(_get_event(*args))
            except:

                traceback.print_exc()

        last = None
        events = []
        prev_data = np.zeros(raw_pre, dtype=self.dtype)
        from .event_detector import process_block
        for pos, data in self.iter_blocks(channel=channel):
            ev, last = process_block(data.astype("i2"), raw_thresh, hdt=raw_hdt, dead=raw_dead, event=last, pos=pos,
                                     limit=raw_limit)
            for start, stop in ev:
                _add_event(start, stop, pos, prev_data, data)
            start = last[0] - pos if last else 0
            prev_data = data.flat[start - raw_pre:]
        if last:
            _add_event(last[0], last[1], pos, None, data)

        return Events(source=self, thresh=thresh, pre=raw_pre, hdt=raw_hdt, dead=raw_dead, data=events)

    def save_wav(self, fname, channel=0, range=(0, float("inf")), rate=None):
        """
        save data to fname as wav file format 
        channel, data range (in samples) to save can be specfied
        if rate is not None it overides the sample rate saved into wav
        """

        import wave

        w = wave.open(fname, "w")
        w.setnchannels(1)
        w.setsampwidth(self.dtype.itemsize)
        if rate is None:
            w.setframerate(1. / self.timescale)
        else:
            w.setframerate(rate)

        start, end = range
        for pos, data in self.iter_blocks(start=start, stop=end, channel=channel):
            if pos < start or pos + data.size > end:
                a = max(0, start - pos)
                b = min(data.size, end - pos)
                w.writeframes(data.flat[a:b].astype("<i2").tostring())
            else:
                w.writeframes(data.astype("<i2").tostring())
        w.close()


class PrettyOrderedDict(OrderedDict):
    def __str__(d, prefix=""):
        indent = "    "
        s = ["OrderedDict("]
        for k, v in d.items():
            if isinstance(v, PrettyOrderedDict):
                s.append(prefix + indent + "({!r}, {}),".format(k, v.__str__(prefix + indent)))
            else:
                s.append(prefix + indent + "({!r}, {!r}),".format(k, v))
        s.append(prefix + ")")
        return "\n".join(s)


class ContigousBase(Data):
    get_block_data = staticmethod(lambda d: d)

    def check_block(self, pos, raw):
        pass

    def raw_iter_blocks(self, start=0, stop=float('inf')):

        buffer = np.empty(math.ceil(8 * 1024 * 1024 / self.block_dtype.itemsize), self.block_dtype)
        block_size = self.get_block_data(buffer)[0, ..., 0].size

        pos = start // block_size * block_size
        seek = start // block_size * buffer.itemsize
        with open(self.fname, "rb", buffering=0) as fh:
            fh.seek(self._offset + seek)
            while True:
                read = fh.readinto(buffer)
                if read < buffer.size * buffer.itemsize:
                    break
                else:
                    yield pos, buffer
                    pos += buffer.size * block_size
                    if pos > stop:
                        return

        remains = read // buffer.itemsize
        rest = read % buffer.itemsize
        if remains:
            yield pos, buffer[:remains]
        if rest:
            self.check_rest(buffer.view('B')[read - rest:read])

    def check_rest(self, data):
        warnings.warn("{} bytes left in the buffer".format(data.size))


class WAV(ContigousBase):

    def __init__(self, fname, **kwargs):
        self.fname = fname
        self.checks = kwargs.get("checks", False)

        with open(self.fname, "rb") as fh:
            self._offset, size = self.parse_meta(fh.read(1024))

        assert self.meta['fmt'] == 1
        type_code = {8: "<i1",
                     16: "<i2",
                     32: "<i4"}[self.meta['bps']]
        self.block_dtype = np.dtype((type_code, (self.meta['nchan'],)))

        self.datascale = [1.] * self.meta['nchan']
        self.timescale = 1. / self.meta['rate']
        self.timeunit = "s"
        self.dataunit = "?"

        self.calc_sizes(size)

    def parse_meta(self, data):

        self.meta = PrettyOrderedDict()
        offset = 0
        while offset < len(data):
            chunk, size = unpack_from("<4sL", data, offset)
            if chunk == "RIFF":
                waveid, = unpack_from("<4s", data, offset + 8)
                assert waveid == "WAVE"
                offset += 12  # "dive" into this chunk, don't use size to skip over

            elif chunk == "fmt ":
                assert size == 16
                fmt, nchan, rate, _, _, bps = unpack_from("<HHLLHH", data, offset + 8)
                self.meta['fmt'] = fmt
                self.meta['nchan'] = nchan
                self.meta['rate'] = rate
                self.meta['bps'] = bps

                offset += 8 + size

            elif chunk == "data":
                return offset + 8, size

            else:

                warnings.warn("unknown chunk {!r}".format(chunk))
                offset += 8 + size

        raise ValueError("Data block not found")


class WFS(ContigousBase):

    def __init__(self, fname, **kwargs):
        self.fname = fname
        self.checks = kwargs.get("checks", False)

        with open(self.fname, 'rb') as fh:
            file_size = os.fstat(fh.fileno()).st_size
            self._offset = self.parse_meta(fh.read(1024), unknown_meta=kwargs.get("unknown_meta", False))

        # determine number of channels
        buf = np.empty(10, self.ch_block_dtype)
        with open(self.fname, "rb", buffering=0) as fh:
            fh.seek(self._offset)
            fh.readinto(buf)
        for ch in range(1, 10):
            if buf['chan'][ch] == ch + 1:
                continue
            elif buf['chan'][ch] == 1:
                break
            else:
                raise ValueError("Invalid channels: %s".format(buf['chan']))

        self.meta['start_time'] = buf['y'][0] * 0.002 / self.meta['hwsetup']['rate']

        self.block_dtype = np.dtype([('ch_data', self.ch_block_dtype, (ch,))])

        self.datascale = [self.meta['hwsetup']['max.volt'] / 32768.] * ch
        self.timescale = 0.001 / self.meta['hwsetup']['rate']
        self.timeunit = "s"
        self.dataunit = "V"

        self.calc_sizes(file_size - self._offset)

    def parse_meta(self, data, unknown_meta=False):

        self.meta = PrettyOrderedDict()
        offset = 0
        while offset < len(data):
            size, id1, id2 = unpack_from("<HBB", data, offset)
            if (size, id1, id2) == (2076, 174, 1):
                return offset
            offset += 2
            if id1 in (173, 174):
                # these have two ids
                offset += 2
                size -= 2
                if (id1, id2) == (174, 42):
                    fmt = [("ver", "H"),
                           ("AD", "B"),
                           ("num", "H"),
                           ("size", "H"),

                           ("id", "B"),
                           ("unk1", "H"),
                           ("rate", "H"),
                           ("trig.mode", "H"),
                           ("trig.src", "H"),
                           ("trig.delay", "h"),
                           ("unk2", "H"),
                           ("max.volt", "H"),
                           ("trig.thresh", "H"),
                           ]
                    sfmt = "<" + "".join(code for name, code in fmt)
                    assert calcsize(sfmt) == size
                    self.meta['hwsetup'] = PrettyOrderedDict(zip(
                        [name for name, code in fmt],
                        unpack_from(sfmt, data, offset)))
                    if self.meta['hwsetup']['AD'] == 2:
                        self.meta['hwsetup']['AD'] = "16-bit signed"
                elif unknown_meta:
                    self.meta[(id1, id2)] = data[offset:offset + size]
            else:
                # only one id
                offset += 1
                size -= 1
                if id1 == 99:
                    self.meta['date'] = data[offset:offset + size].rstrip(b"\0\n")
                elif id1 == 41:
                    self.meta['product'] = PrettyOrderedDict([
                        ("ver", unpack_from("<xH", data, offset)[0]),
                        ("text", data[offset + 3:offset + size].rstrip(b"\r\n\0\x1a"))])
                elif unknown_meta:
                    self.meta[id1] = data[offset:offset + size]
            offset += size
        raise ValueError("Data block not found")

    ch_block_dtype = np.dtype([
        ("size", "u2"),
        ("id1", "u1"),
        ("id2", "u1"),
        ("unknown1", "S6"),
        ("chan", "u1"),
        ("zeros", "S7"),
        ("x", "u4"),
        ("unknown2", "S4"),
        ("y", "u4"),
        ("data", "i2", (1024))])

    get_block_data = staticmethod(lambda d: d['ch_data']['data'].swapaxes(-1, -2))

    def check_block(self, pos, raw):
        if self.checks:
            assert np.all(raw['size'] == 2076)
            assert np.all(raw['id1'] == 174)
            assert np.all(raw['id2'] == 1)

    def check_rest(self, data):
        if not np.all(data == (7, 0, 15, 255, 255, 255, 255, 255, 127)):
            warnings.warn("{} bytes left in the buffer".format(data.size))
