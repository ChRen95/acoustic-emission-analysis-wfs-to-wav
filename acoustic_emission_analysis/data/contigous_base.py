import warnings
import numpy as np

from .base_data import BaseData


class ContigousBase(BaseData):
    get_block_data = staticmethod(lambda d: d)

    def check_block(self, pos, raw):
        pass

    def raw_iter_blocks(self, start=0, stop=float('inf')):
        buffer = np.empty(8 * 1024 * 1024 / self.block_dtype.itemsize, self.block_dtype)
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

    @staticmethod
    def check_rest(data):
        warnings.warn("{} bytes left in the buffer".format(data.size))
