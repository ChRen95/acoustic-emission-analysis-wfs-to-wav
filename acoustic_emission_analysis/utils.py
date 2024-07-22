import random
import numpy as np


def loghist(data, bins=50, range=None, density=None):
    """
    Creates logarithmically spaced bins and calls :func:`numpy.histogram`.

    :param ndarray data:
    :param int bins: number of bins
    :param tuple range: histogram range, by default determined from minimum and maximum in data
    :return: (hist, bins) - histogram counts and bin boundaries
    """
    data = np.asarray(data)
    if range is None:
        a, b = data.min(), data.max()
        if a == 0:
            a = b / 1000.
    else:
        a, b = range
    bins = np.exp(np.linspace(np.log(a), np.log(b), bins))
    hist, bins = np.histogram(data, bins, density=density)
    return hist, bins


def random_power(xmin, a, size=1):
    return xmin * (1 - random.uniform(size=size)) ** (1. / (a + 1))


def bin_centers(bins):
    return (bins[1:] + bins[:-1]) / 2.


def join_bins(bins, counts, mincount=10):
    newbins, newcounts = [bins[0]], []
    s = 0
    for a, b in zip(counts, bins[1:]):
        s += a
        if s < mincount:
            continue
        newcounts.append(s)
        newbins.append(b)
        s = 0
    if s > 0:
        newcounts.append(s)
        newbins.append(b)
    return np.asarray(newbins), np.asarray(newcounts)


def cdf(data):
    return np.sort(data), np.arange(data.size, 0, -1)


def mle(xmin, data):
    d = data[data >= xmin]
    a = 1 - data.size / sum(np.log(xmin / d))
    return -a, (a - 1) / np.sqrt(d.size)


def hist(data, bins=50, range=None, ax=None, density=None):
    if ax is None:
        from matplotlib.pyplot import gca
        ax = gca()

    hist, bins = loghist(data, bins=bins, range=range, density=density)
    l, = ax.plot((bins[1:] + bins[:-1]) / 2, hist, "o")
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True)
    return hist, bins, l


def count(data, thresh):
    return np.logical_and(data[:-1] < thresh, data[1:] >= thresh).sum()
