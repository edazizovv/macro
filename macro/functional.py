import numpy as np
import numpy
from scipy.stats._stats import _kendall_dis
from sklearn.metrics import r2_score
from scipy import stats

def r2_metric(x, y):
    return r2_score(y_pred=x, y_true=y)

def pv_metric(x, y):
    if numpy.unique(x).shape[0] == 1:
        return 0
    else:
        return stats.pearsonr(x, y)[0]

def pa_metric(x, y):
    return numpy.abs(pv_metric(x=x, y=y))

def ps_metric(x, y):
    return stats.pearsonr(x, y)[1]

def sd_metric(x, y):
    """
    There is a known issue in the scipy's notation: scipy's x should be the dependent variable and y should be the independent one
    """

    return stats.somersd(y=x, x=y).statistic


def _false_SomersD(x, y):
    """
    ATTENTION:

    this is an incorrect implementation, do not use it!

    """

    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()

    if x.size != y.size:
        raise ValueError("All inputs must be of the same size, "
                         "found x-size %s and y-size %s" % (x.size, y.size))

    def count_rank_tie(ranks):
        cnt = np.bincount(ranks).astype('int64', copy=False)
        cnt = cnt[cnt > 1]
        return ((cnt * (cnt - 1) // 2).sum(),
            (cnt * (cnt - 1.) * (cnt - 2)).sum(),
            (cnt * (cnt - 1.) * (2*cnt + 5)).sum())

    size = x.size
    perm = np.argsort(y)  # sort on y and convert y to dense ranks
    x, y = x[perm], y[perm]
    y = np.r_[True, y[1:] != y[:-1]].cumsum(dtype=np.intp)

    # stable sort on x and convert x to dense ranks
    perm = np.argsort(x, kind='mergesort')
    x, y = x[perm], y[perm]
    x = np.r_[True, x[1:] != x[:-1]].cumsum(dtype=np.intp)

    dis = _kendall_dis(x, y)  # discordant pairs

    obs = np.r_[True, (x[1:] != x[:-1]) | (y[1:] != y[:-1]), True]
    cnt = np.diff(np.where(obs)[0]).astype('int64', copy=False)

    ntie = (cnt * (cnt - 1) // 2).sum()  # joint ties
    xtie, x0, x1 = count_rank_tie(x)     # ties in x, stats
    ytie, y0, y1 = count_rank_tie(y)     # ties in y, stats

    tot = (size * (size - 1)) // 2

    # Note that tot = con + dis + (xtie - ntie) + (ytie - ntie) + ntie
    #               = con + dis + xtie + ytie - ntie
    #con_minus_dis = tot - xtie - ytie + ntie - 2 * dis
    SD = (tot - xtie - ytie + ntie - 2 * dis) / (tot - ntie)
    return SD   # (SD, dis)
