

import numpy as np


class WFS(ContigousBase):

    def __init__(self, fname, **kwargs):
        self.fname = fname
        self.checks = kwargs.get("checks", False)

        import os
        with open(self.fname, "rb") as fh:
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
        from struct import unpack_from, calcsize
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
                    self.meta['date'] = data[offset:offset + size].rstrip("\0\n")
                elif id1 == 41:
                    self.meta['product'] = PrettyOrderedDict([
                        ("ver", unpack_from("<xH", data, offset)[0]),
                        ("text", data[offset + 3:offset + size].rstrip("\r\n\0\x1a"))])
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
            assert np.alltrue(raw['size'] == 2076)
            assert np.alltrue(raw['id1'] == 174)
            assert np.alltrue(raw['id2'] == 1)

    def check_rest(self, data):
        if not np.alltrue(data == (7, 0, 15, 255, 255, 255, 255, 255, 127)):
            import warnings
            warnings.warn("{} bytes left in the buffer".format(data.size))