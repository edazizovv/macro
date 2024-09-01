#


#
import numpy
import scipy
import pandas
from sklearn.linear_model import enet_path, LinearRegression
from matplotlib import pyplot
from newborn import FrameOld
from sklearn.metrics import r2_score
from sklearn.exceptions import ConvergenceWarning
# ConvergenceWarning('ignore')
import warnings
import time
from functools import partial
from scipy.stats import kendalltau   # , somersd
from macro.macro.functional import SomersD as somersd

from statsmodels.tsa.seasonal import MSTL

# rs = 999
# numpy.random.seed(rs)


run_time = time.time()
#
# keep in mind LARS
class Model:
    def __init__(self, l1_ratio=0.9, eps=1e-12, n_alphas=1_000, standardize=False):
        self.l1_ratio = l1_ratio
        self.eps = eps
        self.n_alphas = n_alphas
        self.alphas_enet = None
        self.coefs_enet = None
        self.neg_log_alphas_lasso = None
        self.standardize = standardize
        self.std_mean = None
        self.std_std = None
    def fit(self, x, y):
        xx = x.copy()
        if self.standardize:
            self.std_mean = []
            self.std_std = []
            for i in range(x.shape[1]):
                std_mean = x[:, i].mean()
                std_std = x[:, i].std()
                xx[:, i] = (x[:, i] - std_mean) / std_std
                self.std_mean.append(std_mean)
                self.std_std.append(std_std)
                if pandas.isna(xx[:, i]).all():
                    xx[:, i] = 0
                else:
                    xx[pandas.isna(xx[:, i]), i] = xx[~pandas.isna(xx[:, i]), i].mean()
                xx[xx[:, i] == numpy.inf, i] = xx[xx[:, i] != numpy.inf, i].max()
                xx[xx[:, i] == -numpy.inf, i] = xx[xx[:, i] != -numpy.inf, i].min()
        self.alphas_enet, self.coefs_enet, _ = enet_path(X=xx, y=y, eps=self.eps, l1_ratio=self.l1_ratio, n_alphas=self.n_alphas)
        self.neg_log_alphas_lasso = -numpy.log10(self.alphas_enet)
    def plot(self, neg=True):
        if neg:
            x_axis = self.neg_log_alphas_lasso
        else:
            x_axis = self.alphas_enet
        for coef_e in self.coefs_enet:
            pyplot.plot(x_axis, coef_e)


f = FrameOld()
target = 'TLT_MEAN'
target_e = f[target]
time_axis = target_e.frame['DATE'].copy()
_ = f.cast(time_axis=time_axis, gb_funcs='last', fill_values='ffill')

factors = []
win_type = [[]]
win_kwgs = [[]]
_ = f.represent(factors=factors, win_type_s=win_type, win_kwgs_s=win_kwgs)

result = f.tighten_dates(cutoff_date='2007-01-01')
# why duplicated???
# result = result.loc[:, ~result.columns.duplicated()].copy()

x_factors = result.columns.values


def linreg_relative(y):
    yy = y.values
    xx = numpy.array(numpy.arange(yy.shape[0]))
    rs = scipy.stats.linregress(x=xx, y=yy)
    tt = xx * rs.slope + rs.intercept
    r = y[-1] / tt[-1] - 1
    return r


def mean_relative(y):
    ym = y.mean()
    r = y.values[-1] / ym - 1
    return r


def median_relative(y):
    ym = y.median()
    r = y.values[-1] / ym - 1
    return r


def relative_q(y):
    r = scipy.stats.percentileofscore(y.values, y.values[-1]) / 100 - 0.5
    return r


def ewm_1shock_relative_3(y):
    window = y.shape[0]
    alpha = 2 / (3 + 1)
    weights = list(reversed([(1-alpha) ** n for n in range(window)]))
    ewma = partial(numpy.average, weights=weights)
    ra = y[-1] / ewma(y) - 1
    return ra


def ewm_1shock_relative_6(y):
    window = y.shape[0]
    alpha = 2 / (6 + 1)
    weights = list(reversed([(1-alpha) ** n for n in range(window)]))
    ewma = partial(numpy.average, weights=weights)
    ra = y[-1] / ewma(y) - 1
    return ra


def ewm_1shock_relative_12(y):
    window = y.shape[0]
    alpha = 2 / (12 + 1)
    weights = list(reversed([(1-alpha) ** n for n in range(window)]))
    ewma = partial(numpy.average, weights=weights)
    ra = y[-1] / ewma(y) - 1
    return ra


def ewm_3shock_relative_6(y):
    window = y.shape[0]
    alpha = 2 / (3 + 1)
    weights = list(reversed([(1-alpha) ** n for n in range(window)]))
    ewma_3 = partial(numpy.average, weights=weights)
    alpha = 2 / (6 + 1)
    weights = list(reversed([(1-alpha) ** n for n in range(window)]))
    ewma_6 = partial(numpy.average, weights=weights)
    ra = ewma_3(y) / ewma_6(y) - 1
    return ra


def ewm_3shock_relative_12(y):
    window = y.shape[0]
    alpha = 2 / (3 + 1)
    weights = list(reversed([(1-alpha) ** n for n in range(window)]))
    ewma_3 = partial(numpy.average, weights=weights)
    alpha = 2 / (12 + 1)
    weights = list(reversed([(1-alpha) ** n for n in range(window)]))
    ewma_12 = partial(numpy.average, weights=weights)
    ra = ewma_3(y) / ewma_12(y) - 1
    return ra


def ewm_6shock_relative_12(y):
    window = y.shape[0]
    alpha = 2 / (6 + 1)
    weights = list(reversed([(1-alpha) ** n for n in range(window)]))
    ewma_6 = partial(numpy.average, weights=weights)
    alpha = 2 / (12 + 1)
    weights = list(reversed([(1-alpha) ** n for n in range(window)]))
    ewma_12 = partial(numpy.average, weights=weights)
    ra = ewma_6(y) / ewma_12(y) - 1
    return ra


def switchmean(y):
    yy = y.values[1:]
    rs = yy.mean() / y.values[0] - 1
    return rs


def switchline(y):
    yy = y.values
    xx = numpy.array(numpy.arange(yy.shape[0]))
    rs = scipy.stats.linregress(x=xx, y=yy)
    return rs.slope


def seasonal_remove_mstl_get_1q(dy):
    ret = []
    for c in dy.columns:
        y = dy[c].values
        mstl = MSTL(y, periods=[3, 6, 12, 36])    # 3 = quarter, 6 = semiyear, 12 = year, 36 = 3 years
        res = mstl.fit()
        e = res.seasonal[:, 0]
        # res.trend
        # res.resid
        ret.append(e.reshape(-1, 1))
    extracted = numpy.concatenate(ret, axis=1)
    return extracted


def seasonal_remove_mstl_get_6m(dy):
    ret = []
    for c in dy.columns:
        y = dy[c].values
        mstl = MSTL(y, periods=[3, 6, 12, 36])    # 3 = quarter, 6 = semiyear, 12 = year, 36 = 3 years
        res = mstl.fit()
        e = res.seasonal[:, 1]
        # res.trend
        # res.resid
        ret.append(e.reshape(-1, 1))
    extracted = numpy.concatenate(ret, axis=1)
    return extracted


def seasonal_remove_mstl_get_1y(dy):
    ret = []
    for c in dy.columns:
        y = dy[c].values
        mstl = MSTL(y, periods=[3, 6, 12, 36])    # 3 = quarter, 6 = semiyear, 12 = year, 36 = 3 years
        res = mstl.fit()
        e = res.seasonal[:, 2]
        # res.trend
        # res.resid
        ret.append(e.reshape(-1, 1))
    extracted = numpy.concatenate(ret, axis=1)
    return extracted


def seasonal_remove_mstl_get_3y(dy):
    ret = []
    for c in dy.columns:
        y = dy[c].values
        mstl = MSTL(y, periods=[3, 6, 12, 36])    # 3 = quarter, 6 = semiyear, 12 = year, 36 = 3 years
        res = mstl.fit()
        e = res.seasonal[:, 3]
        # res.trend
        # res.resid
        ret.append(e)
    extracted = numpy.concatenate(ret, axis=1)
    return extracted


def seasonal_remove_mstl_get_seasonal_trend(dy):
    ret = []
    for c in dy.columns:
        y = dy[c].values
        mstl = MSTL(y, periods=[3, 6, 12, 36])    # 3 = quarter, 6 = semiyear, 12 = year, 36 = 3 years
        res = mstl.fit()
        e = res.trend
        # res.trend
        # res.resid
        ret.append(e.reshape(-1, 1))
    extracted = numpy.concatenate(ret, axis=1)
    return extracted


def seasonal_remove_mstl_get_seasonal_resid(dy):
    ret = []
    for c in dy.columns:
        y = dy[c].values
        mstl = MSTL(y, periods=[3, 6, 12, 36])    # 3 = quarter, 6 = semiyear, 12 = year, 36 = 3 years
        res = mstl.fit()
        e = res.resid
        # res.trend
        # res.resid
        ret.append(e.reshape(-1, 1))
    extracted = numpy.concatenate(ret, axis=1)
    return extracted

from sklearn.preprocessing import KBinsDiscretizer
def binners_20_perc(y):
    for c in y.columns:
        y.loc[pandas.isna(y[c]), c] = y.loc[~pandas.isna(y[c]), c].mean()
        y.loc[y.loc[:, c] == numpy.inf, c] = y.loc[y.loc[:, c] != numpy.inf, c].max()
        y.loc[y.loc[:, c] == -numpy.inf, c] = y.loc[y.loc[:, c] != -numpy.inf, c].min()
    y = y.values
    bb = KBinsDiscretizer(n_bins=20, encode='ordinal', strategy='quantile')
    z = bb.fit_transform(X=y) / 20
    return z

def binners_20_smile(y):
    for c in y.columns:
        y.loc[pandas.isna(y[c]), c] = y.loc[~pandas.isna(y[c]), c].mean()
        y.loc[y.loc[:, c] == numpy.inf, c] = y.loc[y.loc[:, c] != numpy.inf, c].max()
        y.loc[y.loc[:, c] == -numpy.inf, c] = y.loc[y.loc[:, c] != -numpy.inf, c].min()
    y = y.values
    bb = KBinsDiscretizer(n_bins=20, encode='ordinal', strategy='quantile')
    z = bb.fit_transform(X=y)
    zz = z.copy()
    for c in range(y.shape[1]):
        for jj in range(20):
            zz[z[:, c] == jj, c] = zz[z[:, c] == jj, c].mean()
        low, up = numpy.quantile(y[:, c], 1 / 20), numpy.quantile(y[:, c], 19 / 20)
        zz[z[:, c] == 0, c] = low
        zz[z[:, c] == 20 - 1, c] = up
    zz = (zz - zz.min(axis=0)) / (zz.max(axis=0) - zz.min(axis=0))
    return zz


def linear_slope(y):
    yy = y.values
    xx = numpy.array(numpy.arange(yy.shape[0]))
    rs = scipy.stats.linregress(x=xx, y=yy)
    r = rs.slope
    return r


def linear_r2err(y):
    yy = y.values
    xx = numpy.array(numpy.arange(yy.shape[0]))
    rs = scipy.stats.linregress(x=xx, y=yy)
    tt = xx * rs.slope + rs.intercept
    r = r2_score(y_true=yy, y_pred=tt)
    return r


def n_rate_conseq(y):
    yy = y.pct_change().values[1:]
    stop_jj = False
    jj = yy.shape[0] - 1
    r = 0
    while not stop_jj:
        if jj > 1:
            if r == 0:
                if yy[jj] > yy[jj - 1]:
                    r = +1
                elif yy[jj] < yy[jj - 1]:
                    r = -1
                else:
                    pass
            elif r > 0:
                if yy[jj] > yy[jj - 1]:
                    r += 1
                elif yy[jj] < yy[jj - 1]:
                    stop_jj = True
                else:
                    pass
            elif r < 0:
                if yy[jj] > yy[jj - 1]:
                    stop_jj = True
                elif yy[jj] < yy[jj - 1]:
                    r -= 1
                else:
                    pass
        else:
            stop_jj = True
        jj -= 1
    r = r / yy.shape[0]
    return r


def n_rate_full(y):
    yy = y.pct_change().values[1:]
    r = ((yy > 0).sum() - (yy < 0).sum()) / yy.shape[0]
    return r


def rel_to_max(y):
    r = y.values[-1] / y.max()
    return r


def rel_to_min(y):
    r = y.values[-1] / y.min()
    return r


def pct_std(y):
    yy = y.pct_change().values[1:]
    r = yy.std()
    return r


def pct_std_blow(y):
    yy = y.pct_change().values[1:]
    yy_std = yy.std()
    r = ((yy > yy_std).sum() + (yy < yy_std).sum()) / yy.shape[0]
    return r


def relative_mean(y):
    yy = y.values
    r = (yy > yy.mean()).sum() / yy.shape[0]
    return r


def relative_positive_pct(y):
    yy = y.pct_change().values[1:]
    r = (yy > 0).sum() / yy.shape[0]
    return r










calculate = False

if not calculate:
    result = pandas.read_csv('./result_model_medieval.csv')
    result = result.set_index('jx')
else:
    factors = [x_factors.tolist()] * 49 + [x_factors.tolist()] * 54 + [x_factors.tolist()] * 2 + [x_factors.tolist()] * 2
    win_type = [['full_apply', 'rolling']] * 49 + [['rolling']] * 54 + [['pct', 'full_apply']] * 2 + [['full_apply']] * 2
    win_kwgs = [[{'func': seasonal_remove_mstl_get_seasonal_resid}, {'win_func': None, 'agg_func': linreg_relative, 'window': 3}],
                [{'func': seasonal_remove_mstl_get_seasonal_resid}, {'win_func': None, 'agg_func': mean_relative, 'window': 3}],
                [{'func': seasonal_remove_mstl_get_seasonal_resid}, {'win_func': None, 'agg_func': median_relative, 'window': 3}],
                [{'func': seasonal_remove_mstl_get_seasonal_resid}, {'win_func': None, 'agg_func': relative_q, 'window': 3}],
                [{'func': seasonal_remove_mstl_get_seasonal_resid}, {'win_func': None, 'agg_func': ewm_1shock_relative_3, 'window': 3}],
                [{'func': seasonal_remove_mstl_get_seasonal_resid}, {'win_func': None, 'agg_func': ewm_1shock_relative_6, 'window': 3}],
                [{'func': seasonal_remove_mstl_get_seasonal_resid}, {'win_func': None, 'agg_func': ewm_1shock_relative_12, 'window': 3}],

                [{'func': seasonal_remove_mstl_get_seasonal_resid},
                 {'win_func': None, 'agg_func': linear_slope, 'window': 3}],
                [{'func': seasonal_remove_mstl_get_seasonal_resid},
                 {'win_func': None, 'agg_func': linear_r2err, 'window': 3}],
                [{'func': seasonal_remove_mstl_get_seasonal_resid},
                 {'win_func': None, 'agg_func': rel_to_max, 'window': 3}],
                [{'func': seasonal_remove_mstl_get_seasonal_resid},
                 {'win_func': None, 'agg_func': rel_to_min, 'window': 3}],
                [{'func': seasonal_remove_mstl_get_seasonal_resid},
                 {'win_func': None, 'agg_func': relative_mean, 'window': 3}],

                [{'func': seasonal_remove_mstl_get_seasonal_resid}, {'win_func': None, 'agg_func': linreg_relative, 'window': 6}],
                [{'func': seasonal_remove_mstl_get_seasonal_resid}, {'win_func': None, 'agg_func': mean_relative, 'window': 6}],
                [{'func': seasonal_remove_mstl_get_seasonal_resid}, {'win_func': None, 'agg_func': median_relative, 'window': 6}],
                [{'func': seasonal_remove_mstl_get_seasonal_resid}, {'win_func': None, 'agg_func': relative_q, 'window': 6}],
                [{'func': seasonal_remove_mstl_get_seasonal_resid}, {'win_func': None, 'agg_func': ewm_1shock_relative_3, 'window': 6}],
                [{'func': seasonal_remove_mstl_get_seasonal_resid}, {'win_func': None, 'agg_func': ewm_1shock_relative_6, 'window': 6}],
                [{'func': seasonal_remove_mstl_get_seasonal_resid}, {'win_func': None, 'agg_func': ewm_1shock_relative_12, 'window': 6}],

                [{'func': seasonal_remove_mstl_get_seasonal_resid},
                 {'win_func': None, 'agg_func': linear_slope, 'window': 6}],
                [{'func': seasonal_remove_mstl_get_seasonal_resid},
                 {'win_func': None, 'agg_func': linear_r2err, 'window': 6}],
                [{'func': seasonal_remove_mstl_get_seasonal_resid},
                 {'win_func': None, 'agg_func': n_rate_conseq, 'window': 6}],
                [{'func': seasonal_remove_mstl_get_seasonal_resid},
                 {'win_func': None, 'agg_func': n_rate_full, 'window': 6}],
                [{'func': seasonal_remove_mstl_get_seasonal_resid},
                 {'win_func': None, 'agg_func': rel_to_max, 'window': 6}],
                [{'func': seasonal_remove_mstl_get_seasonal_resid},
                 {'win_func': None, 'agg_func': rel_to_min, 'window': 6}],
                [{'func': seasonal_remove_mstl_get_seasonal_resid},
                 {'win_func': None, 'agg_func': pct_std, 'window': 6}],
                [{'func': seasonal_remove_mstl_get_seasonal_resid},
                 {'win_func': None, 'agg_func': pct_std_blow, 'window': 6}],
                [{'func': seasonal_remove_mstl_get_seasonal_resid},
                 {'win_func': None, 'agg_func': relative_mean, 'window': 6}],
                [{'func': seasonal_remove_mstl_get_seasonal_resid},
                 {'win_func': None, 'agg_func': relative_positive_pct, 'window': 6}],

                [{'func': seasonal_remove_mstl_get_seasonal_resid}, {'win_func': None, 'agg_func': linreg_relative, 'window': 12}],
                [{'func': seasonal_remove_mstl_get_seasonal_resid}, {'win_func': None, 'agg_func': mean_relative, 'window': 12}],
                [{'func': seasonal_remove_mstl_get_seasonal_resid}, {'win_func': None, 'agg_func': median_relative, 'window': 12}],
                [{'func': seasonal_remove_mstl_get_seasonal_resid}, {'win_func': None, 'agg_func': relative_q, 'window': 12}],
                [{'func': seasonal_remove_mstl_get_seasonal_resid}, {'win_func': None, 'agg_func': ewm_1shock_relative_3, 'window': 12}],
                [{'func': seasonal_remove_mstl_get_seasonal_resid}, {'win_func': None, 'agg_func': ewm_1shock_relative_6, 'window': 12}],
                [{'func': seasonal_remove_mstl_get_seasonal_resid}, {'win_func': None, 'agg_func': ewm_1shock_relative_12, 'window': 12}],
                [{'func': seasonal_remove_mstl_get_seasonal_resid}, {'win_func': None, 'agg_func': ewm_3shock_relative_6, 'window': 12}],
                [{'func': seasonal_remove_mstl_get_seasonal_resid}, {'win_func': None, 'agg_func': ewm_3shock_relative_12, 'window': 12}],
                [{'func': seasonal_remove_mstl_get_seasonal_resid}, {'win_func': None, 'agg_func': ewm_6shock_relative_12, 'window': 12}],

                [{'func': seasonal_remove_mstl_get_seasonal_resid},
                 {'win_func': None, 'agg_func': linear_slope, 'window': 12}],
                [{'func': seasonal_remove_mstl_get_seasonal_resid},
                 {'win_func': None, 'agg_func': linear_r2err, 'window': 12}],
                [{'func': seasonal_remove_mstl_get_seasonal_resid},
                 {'win_func': None, 'agg_func': n_rate_conseq, 'window': 12}],
                [{'func': seasonal_remove_mstl_get_seasonal_resid},
                 {'win_func': None, 'agg_func': n_rate_full, 'window': 12}],
                [{'func': seasonal_remove_mstl_get_seasonal_resid},
                 {'win_func': None, 'agg_func': rel_to_max, 'window': 12}],
                [{'func': seasonal_remove_mstl_get_seasonal_resid},
                 {'win_func': None, 'agg_func': rel_to_min, 'window': 12}],
                [{'func': seasonal_remove_mstl_get_seasonal_resid},
                 {'win_func': None, 'agg_func': pct_std, 'window': 12}],
                [{'func': seasonal_remove_mstl_get_seasonal_resid},
                 {'win_func': None, 'agg_func': pct_std_blow, 'window': 12}],
                [{'func': seasonal_remove_mstl_get_seasonal_resid},
                 {'win_func': None, 'agg_func': relative_mean, 'window': 12}],
                [{'func': seasonal_remove_mstl_get_seasonal_resid},
                 {'win_func': None, 'agg_func': relative_positive_pct, 'window': 12}],

                [{'win_func': None, 'agg_func': linreg_relative, 'window': 3}],
                [{'win_func': None, 'agg_func': mean_relative, 'window': 3}],
                [{'win_func': None, 'agg_func': median_relative, 'window': 3}],
                [{'win_func': None, 'agg_func': relative_q, 'window': 3}],
                [{'win_func': None, 'agg_func': ewm_1shock_relative_3, 'window': 3}],
                [{'win_func': None, 'agg_func': ewm_1shock_relative_6, 'window': 3}],
                [{'win_func': None, 'agg_func': ewm_1shock_relative_12, 'window': 3}],
                [{'win_func': None, 'agg_func': linreg_relative, 'window': 6}],
                [{'win_func': None, 'agg_func': mean_relative, 'window': 6}],
                [{'win_func': None, 'agg_func': median_relative, 'window': 6}],
                [{'win_func': None, 'agg_func': relative_q, 'window': 6}],
                [{'win_func': None, 'agg_func': ewm_1shock_relative_3, 'window': 6}],
                [{'win_func': None, 'agg_func': ewm_1shock_relative_6, 'window': 6}],
                [{'win_func': None, 'agg_func': ewm_1shock_relative_12, 'window': 6}],
                [{'win_func': None, 'agg_func': linreg_relative, 'window': 12}],
                [{'win_func': None, 'agg_func': mean_relative, 'window': 12}],
                [{'win_func': None, 'agg_func': median_relative, 'window': 12}],
                [{'win_func': None, 'agg_func': relative_q, 'window': 12}],
                [{'win_func': None, 'agg_func': ewm_1shock_relative_3, 'window': 12}],
                [{'win_func': None, 'agg_func': ewm_1shock_relative_6, 'window': 12}],
                [{'win_func': None, 'agg_func': ewm_1shock_relative_12, 'window': 12}],
                [{'win_func': None, 'agg_func': ewm_3shock_relative_6, 'window': 12}],
                [{'win_func': None, 'agg_func': ewm_3shock_relative_12, 'window': 12}],
                [{'win_func': None, 'agg_func': ewm_6shock_relative_12, 'window': 12}],

                [{'win_func': None, 'agg_func': linear_slope, 'window': 3}],
                [{'win_func': None, 'agg_func': linear_r2err, 'window': 3}],
                [{'win_func': None, 'agg_func': n_rate_conseq, 'window': 3}],
                [{'win_func': None, 'agg_func': n_rate_full, 'window': 3}],
                [{'win_func': None, 'agg_func': rel_to_max, 'window': 3}],
                [{'win_func': None, 'agg_func': rel_to_min, 'window': 3}],
                [{'win_func': None, 'agg_func': pct_std, 'window': 3}],
                [{'win_func': None, 'agg_func': pct_std_blow, 'window': 3}],
                [{'win_func': None, 'agg_func': relative_mean, 'window': 3}],
                [{'win_func': None, 'agg_func': relative_positive_pct, 'window': 3}],

                [{'win_func': None, 'agg_func': linear_slope, 'window': 6}],
                [{'win_func': None, 'agg_func': linear_r2err, 'window': 6}],
                [{'win_func': None, 'agg_func': n_rate_conseq, 'window': 6}],
                [{'win_func': None, 'agg_func': n_rate_full, 'window': 6}],
                [{'win_func': None, 'agg_func': rel_to_max, 'window': 6}],
                [{'win_func': None, 'agg_func': rel_to_min, 'window': 6}],
                [{'win_func': None, 'agg_func': pct_std, 'window': 6}],
                [{'win_func': None, 'agg_func': pct_std_blow, 'window': 6}],
                [{'win_func': None, 'agg_func': relative_mean, 'window': 6}],
                [{'win_func': None, 'agg_func': relative_positive_pct, 'window': 6}],

                [{'win_func': None, 'agg_func': linear_slope, 'window': 12}],
                [{'win_func': None, 'agg_func': linear_r2err, 'window': 12}],
                [{'win_func': None, 'agg_func': n_rate_conseq, 'window': 12}],
                [{'win_func': None, 'agg_func': n_rate_full, 'window': 12}],
                [{'win_func': None, 'agg_func': rel_to_max, 'window': 12}],
                [{'win_func': None, 'agg_func': rel_to_min, 'window': 12}],
                [{'win_func': None, 'agg_func': pct_std, 'window': 12}],
                [{'win_func': None, 'agg_func': pct_std_blow, 'window': 12}],
                [{'win_func': None, 'agg_func': relative_mean, 'window': 12}],
                [{'win_func': None, 'agg_func': relative_positive_pct, 'window': 12}],
    #             [{'win_func': None, 'agg_func': switchmean, 'window': 3+1}]]
                [{}, {'func': binners_20_perc}],
                [{}, {'func': binners_20_smile}],
                [{'func': binners_20_perc}],
                [{'func': binners_20_smile}]]
    _ = f.represent(factors=factors, win_type_s=win_type, win_kwgs_s=win_kwgs)

    result = f.tighten_dates(cutoff_date='2007-01-01')
    result.to_csv('./result_model_medieval.csv')

target = 'TLT_MEAN__pct'
# target = 'IVV__switchblade_4__None__switchmean'
# x_factors = result.columns.values
x_factors = [x for x in result.columns.values if x != target]


print('N of factors considered: {0}'.format(len(x_factors)))
time_axis = result.index

standardize = True
time_sub_rate = 0.75
time_sub_replace = False
nt = int(time_axis.shape[0] * time_sub_rate)
rere = []
x_bases = []
# up_to = 0.2
respo = []
sd_thresh = 0.1

sd_res = []
for t in range(100):
    ixes = numpy.random.choice(list(range(time_axis.shape[0] - 1)), size=(nt,), replace=time_sub_replace)
    x_ix_train = time_axis.values[:-1][ixes]
    y_ix_train = time_axis.values[1:][ixes]
    left_ixes = [x for x in list(range(time_axis.shape[0] - 1)) if x not in ixes]
    x_ix_test = time_axis.values[:-1][left_ixes]
    y_ix_test = time_axis.values[1:][left_ixes]

    x_train = result.loc[x_ix_train, :]
    y_train = result.loc[y_ix_train, target]
    x_test = result.loc[x_ix_test, :]
    y_test = result.loc[y_ix_test, target]

    sds = []
    for xb in x_factors:
        sd = somersd(y_train.values, x_train[xb].values)
        sds.append(sd)
    sdr = pandas.DataFrame(data={'drivers': x_factors,
                                 'sd': sds,
                                 't': t})
    sd_res.append(sdr)
sd_res = pandas.concat(sd_res, axis=0, ignore_index=True)
sd_res_avg = sd_res.groupby(by='drivers')['sd'].median()
x_factors_selection = sd_res_avg[sd_res_avg.abs() >= sd_thresh].index.values

x_sub_rate = 0.25
x_sub_replace = False
nx = int(len(x_factors_selection) * x_sub_rate)

print('Univariate thresh selected {0} out of {1} with {2:.4f} threshold'.format(len(x_factors_selection), len(x_factors), sd_thresh))

print('Takes {0} out of {1} factors ({2:.4f} rate); {3} out of {4} tx ({5:.4f} rate)'.format(nx, len(x_factors_selection), x_sub_rate, nt, time_axis.shape[0], time_sub_rate))
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    for k in range(500):
        print(k)

        nxx = numpy.random.choice(range(nx)) + 2
        x_base = numpy.random.choice(x_factors_selection, size=(nxx,), replace=x_sub_replace)
        x_bases.append(x_base)

        ixes = numpy.random.choice(list(range(time_axis.shape[0] - 1)), size=(nt,), replace=time_sub_replace)
        x_ix_train = time_axis.values[:-1][ixes]
        y_ix_train = time_axis.values[1:][ixes]
        left_ixes = [x for x in list(range(time_axis.shape[0] - 1)) if x not in ixes]
        x_ix_test = time_axis.values[:-1][left_ixes]
        y_ix_test = time_axis.values[1:][left_ixes]

        x_train = result.loc[x_ix_train, x_base].values
        y_train = result.loc[y_ix_train, target].values
        x_test = result.loc[x_ix_test, x_base].values
        y_test = result.loc[y_ix_test, target].values

        xx_train = x_train.copy()
        xx_test = x_test.copy()
        if standardize:
            std_means = []
            std_stds = []
            for i in range(xx_train.shape[1]):
                std_mean = xx_train[:, i].mean()
                std_std = xx_train[:, i].std()
                xx_train[:, i] = (xx_train[:, i] - std_mean) / std_std
                xx_test[:, i] = (xx_test[:, i] - std_mean) / std_std
                std_means.append(std_mean)
                std_stds.append(std_std)
                if numpy.ma.masked_invalid(xx_train[:, i]).mask.all():
                    xx_train[:, i] = 0
                else:
                    xx_train[pandas.isna(xx_train[:, i]), i] = xx_train[
                        ~pandas.isna(xx_train[:, i]), i].mean()
                if numpy.ma.masked_invalid(xx_test[:, i]).mask.all():
                    xx_test[:, i] = 0
                else:
                    xx_test[pandas.isna(xx_test[:, i]), i] = xx_test[
                        ~pandas.isna(xx_test[:, i]), i].mean()
                xx_train[xx_train[:, i] == numpy.inf, i] = xx_train[
                    xx_train[:, i] != numpy.inf, i].max()
                xx_train[xx_train[:, i] == -numpy.inf, i] = xx_train[
                    xx_train[:, i] != -numpy.inf, i].min()
                xx_test[xx_test[:, i] == numpy.inf, i] = xx_test[
                    xx_test[:, i] != numpy.inf, i].max()
                xx_test[xx_test[:, i] == -numpy.inf, i] = xx_test[
                    xx_test[:, i] != -numpy.inf, i].min()

        for j in range(len(x_factors_selection)):

            if x_factors_selection[j] in x_base:

                xx_train_w = xx_train.copy()
                xx_test_w = xx_test.copy()

            else:

                xx_train_w = result.loc[x_ix_train, x_base.tolist() + [x_factors_selection[j]]].values
                xx_test_w = result.loc[x_ix_test, x_base.tolist() + [x_factors_selection[j]]].values

                if standardize:
                    std_means = []
                    std_stds = []
                    for i in range(xx_train_w.shape[1]):
                        std_mean = xx_train_w[:, i].mean()
                        std_std = xx_train_w[:, i].std()
                        xx_train_w[:, i] = (xx_train_w[:, i] - std_mean) / std_std
                        xx_test_w[:, i] = (xx_test_w[:, i] - std_mean) / std_std
                        std_means.append(std_mean)
                        std_stds.append(std_std)
                        if numpy.ma.masked_invalid(xx_train_w[:, i]).mask.all():
                            xx_train_w[:, i] = 0
                        else:
                            xx_train_w[pandas.isna(xx_train_w[:, i]), i] = xx_train_w[
                                ~pandas.isna(xx_train_w[:, i]), i].mean()
                        if numpy.ma.masked_invalid(xx_test_w[:, i]).mask.all():
                            xx_test_w[:, i] = 0
                        else:
                            xx_test_w[pandas.isna(xx_test_w[:, i]), i] = xx_test_w[
                                ~pandas.isna(xx_test_w[:, i]), i].mean()
                        xx_train_w[xx_train_w[:, i] == numpy.inf, i] = xx_train_w[
                            xx_train_w[:, i] != numpy.inf, i].max()
                        xx_train_w[xx_train_w[:, i] == -numpy.inf, i] = xx_train_w[
                            xx_train_w[:, i] != -numpy.inf, i].min()
                        xx_test_w[xx_test_w[:, i] == numpy.inf, i] = xx_test_w[
                            xx_test_w[:, i] != numpy.inf, i].max()
                        xx_test_w[xx_test_w[:, i] == -numpy.inf, i] = xx_test_w[
                            xx_test_w[:, i] != -numpy.inf, i].min()
            lm = LinearRegression(n_jobs=-1)
            try:
                lm.fit(X=xx_train_w, y=y_train)
                yy_hat_test = lm.predict(X=xx_test_w)
                r2_tt2st_w = r2_score(y_true=y_test, y_pred=yy_hat_test)
                sd_tt2st_w = somersd(x=y_test, y=yy_hat_test)  # .statistic
            except numpy.linalg.LinAlgError as e:
                r2_tt2st_w = numpy.nan
                sd_tt2st_w = numpy.nan

            if r2_tt2st_w > 1:
                r2_tt2st_w = numpy.nan
            elif r2_tt2st_w < -1:
                r2_tt2st_w = numpy.nan
            if sd_tt2st_w > 1:
                sd_tt2st_w = numpy.nan
            elif sd_tt2st_w < -1:
                sd_tt2st_w = numpy.nan

            if x_factors_selection[j] in x_base:

                xx_train_wo = xx_train[:, [i for i in range(len(x_base)) if i != j]]
                xx_test_wo = xx_test[:, [i for i in range(len(x_base)) if i != j]]

            else:

                xx_train_wo = xx_train.copy()
                xx_test_wo = xx_test.copy()

            lm = LinearRegression(n_jobs=-1)
            try:
                lm.fit(X=xx_train_wo, y=y_train)
                yy_hat_test = lm.predict(X=xx_test_wo)
                r2_tt2st_wo = r2_score(y_true=y_test, y_pred=yy_hat_test)
                sd_tt2st_wo = somersd(x=y_test, y=yy_hat_test)  # .statistic
            except numpy.linalg.LinAlgError as e:
                r2_tt2st_wo = numpy.nan
                sd_tt2st_wo = numpy.nan

            if r2_tt2st_wo > 1:
                r2_tt2st_wo = numpy.nan
            elif r2_tt2st_wo < -1:
                r2_tt2st_wo = numpy.nan
            if sd_tt2st_wo > 1:
                sd_tt2st_wo = numpy.nan
            elif sd_tt2st_wo < -1:
                sd_tt2st_wo = numpy.nan

            res = pandas.DataFrame(data={'drivers': [x_factors_selection[j]],
                                         'perf_r2_w': [r2_tt2st_w],
                                         'perf_sd_w': [sd_tt2st_w],
                                         'perf_r2_wo': [r2_tt2st_wo],
                                         'perf_sd_wo': [sd_tt2st_wo],
                                         'k': [k]})
            respo.append(res)


respo = pandas.concat(respo, axis=0, ignore_index=True)
respo['diff_r2'] = respo['perf_r2_w'] - respo['perf_r2_wo']
respo['diff_sd'] = respo['perf_sd_w'] - respo['perf_sd_wo']
respo_mean = respo.groupby(by=['drivers'])[['perf_r2_w', 'diff_r2', 'perf_sd_w', 'diff_sd', 'perf_sd_wo']].mean()
respo_median = respo.groupby(by=['drivers'])[['perf_r2_w', 'diff_r2', 'perf_sd_w', 'diff_sd', 'perf_sd_wo']].median()
respo_min = respo.groupby(by=['drivers'])[['perf_r2_w', 'diff_r2', 'perf_sd_w', 'diff_sd', 'perf_sd_wo']].min()

"""
zz_tr_m0 = xx_micro_train.mean(axis=0)
zz_tr_m1 = xx_micro_train.mean(axis=1)
zz_tr_s0 = xx_micro_train.std(axis=0)
zz_tr_s1 = xx_micro_train.std(axis=1)
zz_ts_m1 = xx_micro_test.mean(axis=1)
zz_ts_m0 = xx_micro_test.mean(axis=0)
zz_ts_s1 = xx_micro_test.std(axis=1)
zz_ts_s0 = xx_micro_test.std(axis=0)

zz_tr_0 = pandas.DataFrame(data={'mean': zz_tr_m0, 'std': zz_tr_s0})
zz_tr_1 = pandas.DataFrame(data={'mean': zz_tr_m1, 'std': zz_tr_s1})
zz_ts_0 = pandas.DataFrame(data={'mean': zz_ts_m0, 'std': zz_ts_s0})
zz_ts_1 = pandas.DataFrame(data={'mean': zz_ts_m1, 'std': zz_ts_s1})

yzt = pandas.DataFrame(data={'y_true': y_train, 'y_hat': y_micro_hat_train})
yzs = pandas.DataFrame(data={'y_true': y_test, 'y_hat': y_micro_hat_test})
"""

# x_base = reported_mean.index.values[reported_mean['enter_step'].values > 0][:10]
# x_base = respo_mean.index.values[respo_mean['perf_sd_w'].values > 0.05]
# x_base = respo_mean['perf_sd_w'].sort_values().iloc[-30:].index.values.tolist()
x_base = respo_mean['diff_sd'].sort_values().dropna().iloc[-30:].index.values.tolist()

if len(x_base) == 0:
    raise Exception("No predictors found")

sds_t = []
for t in range(100):
    ixes = numpy.random.choice(list(range(time_axis.shape[0] - 1)), size=(nt,), replace=time_sub_replace)
    x_ix_train = time_axis.values[:-1][ixes]
    y_ix_train = time_axis.values[1:][ixes]
    left_ixes = [x for x in list(range(time_axis.shape[0] - 1)) if x not in ixes]
    x_ix_test = time_axis.values[:-1][left_ixes]
    y_ix_test = time_axis.values[1:][left_ixes]

    x_train = result.loc[x_ix_train, x_base].values
    y_train = result.loc[y_ix_train, target].values
    x_test = result.loc[x_ix_test, x_base].values
    y_test = result.loc[y_ix_test, target].values

    xx_train = x_train.copy()
    xx_test = x_test.copy()
    if standardize:
        std_means = []
        std_stds = []
        for i in range(xx_train.shape[1]):
            std_mean = xx_train[:, i].mean()
            std_std = xx_train[:, i].std()
            xx_train[:, i] = (xx_train[:, i] - std_mean) / std_std
            xx_test[:, i] = (xx_test[:, i] - std_mean) / std_std
            std_means.append(std_mean)
            std_stds.append(std_std)
            if numpy.ma.masked_invalid(xx_train[:, i]).mask.all():
                xx_train[:, i] = 0
            else:
                xx_train[pandas.isna(xx_train[:, i]), i] = xx_train[
                    ~pandas.isna(xx_train[:, i]), i].mean()
            if numpy.ma.masked_invalid(xx_test[:, i]).mask.all():
                xx_test[:, i] = 0
            else:
                xx_test[pandas.isna(xx_test[:, i]), i] = xx_test[
                    ~pandas.isna(xx_test[:, i]), i].mean()
            xx_train[xx_train[:, i] == numpy.inf, i] = xx_train[
                xx_train[:, i] != numpy.inf, i].max()
            xx_train[xx_train[:, i] == -numpy.inf, i] = xx_train[
                xx_train[:, i] != -numpy.inf, i].min()
            xx_test[xx_test[:, i] == numpy.inf, i] = xx_test[
                xx_test[:, i] != numpy.inf, i].max()
            xx_test[xx_test[:, i] == -numpy.inf, i] = xx_test[
                xx_test[:, i] != -numpy.inf, i].min()

    lm = LinearRegression(n_jobs=-1)
    try:
        lm.fit(X=xx_train, y=y_train)
        yy_hat_test = lm.predict(X=xx_test)
        r2_tt2st_w = r2_score(y_true=y_test, y_pred=yy_hat_test)
        sd_tt2st_w = somersd(x=y_test, y=yy_hat_test)  # .statistic
    except numpy.linalg.LinAlgError as e:
        r2_tt2st_w = numpy.nan
        sd_tt2st_w = numpy.nan

    if r2_tt2st_w > 1:
        r2_tt2st_w = numpy.nan
    elif r2_tt2st_w < -1:
        r2_tt2st_w = numpy.nan
    if sd_tt2st_w > 1:
        sd_tt2st_w = numpy.nan
    elif sd_tt2st_w < -1:
        sd_tt2st_w = numpy.nan

    sds_t.append(sd_tt2st_w)


previous_perf = pandas.Series(sds_t).median()
stop = False
while not stop:
    print('ITERATION')

    respox = []
    for k in range(100):
        print(k)

        ixes = numpy.random.choice(list(range(time_axis.shape[0] - 1)), size=(nt,), replace=time_sub_replace)
        x_ix_train = time_axis.values[:-1][ixes]
        y_ix_train = time_axis.values[1:][ixes]
        left_ixes = [x for x in list(range(time_axis.shape[0] - 1)) if x not in ixes]
        x_ix_test = time_axis.values[:-1][left_ixes]
        y_ix_test = time_axis.values[1:][left_ixes]

        x_left = [x for x in x_factors_selection if x not in x_base]

        for j in range(len(x_left)):

            #

            x_train = result.loc[x_ix_train, x_base].values
            y_train = result.loc[y_ix_train, target].values
            x_test = result.loc[x_ix_test, x_base].values
            y_test = result.loc[y_ix_test, target].values

            xx_train = x_train.copy()
            xx_test = x_test.copy()
            if standardize:
                std_means = []
                std_stds = []
                for i in range(xx_train.shape[1]):
                    std_mean = xx_train[:, i].mean()
                    std_std = xx_train[:, i].std()
                    xx_train[:, i] = (xx_train[:, i] - std_mean) / std_std
                    xx_test[:, i] = (xx_test[:, i] - std_mean) / std_std
                    std_means.append(std_mean)
                    std_stds.append(std_std)
                    if numpy.ma.masked_invalid(xx_train[:, i]).mask.all():
                        xx_train[:, i] = 0
                    else:
                        xx_train[pandas.isna(xx_train[:, i]), i] = xx_train[
                            ~pandas.isna(xx_train[:, i]), i].mean()
                    if numpy.ma.masked_invalid(xx_test[:, i]).mask.all():
                        xx_test[:, i] = 0
                    else:
                        xx_test[pandas.isna(xx_test[:, i]), i] = xx_test[
                            ~pandas.isna(xx_test[:, i]), i].mean()
                    xx_train[xx_train[:, i] == numpy.inf, i] = xx_train[
                        xx_train[:, i] != numpy.inf, i].max()
                    xx_train[xx_train[:, i] == -numpy.inf, i] = xx_train[
                        xx_train[:, i] != -numpy.inf, i].min()
                    xx_test[xx_test[:, i] == numpy.inf, i] = xx_test[
                        xx_test[:, i] != numpy.inf, i].max()
                    xx_test[xx_test[:, i] == -numpy.inf, i] = xx_test[
                        xx_test[:, i] != -numpy.inf, i].min()

            lm = LinearRegression(n_jobs=-1)
            try:
                lm.fit(X=xx_train, y=y_train)
                yy_hat_test = lm.predict(X=xx_test)
                r2_tt2st_w = r2_score(y_true=y_test, y_pred=yy_hat_test)
                sd_tt2st_w = somersd(x=y_test, y=yy_hat_test)  # .statistic
            except numpy.linalg.LinAlgError as e:
                r2_tt2st_w = numpy.nan
                sd_tt2st_w = numpy.nan

            if r2_tt2st_w > 1:
                r2_tt2st_w = numpy.nan
            elif r2_tt2st_w < -1:
                r2_tt2st_w = numpy.nan
            if sd_tt2st_w > 1:
                sd_tt2st_w = numpy.nan
            elif sd_tt2st_w < -1:
                sd_tt2st_w = numpy.nan

            #

            x_train = result.loc[x_ix_train, x_base + [x_left[j]]].values
            y_train = result.loc[y_ix_train, target].values
            x_test = result.loc[x_ix_test, x_base + [x_left[j]]].values
            y_test = result.loc[y_ix_test, target].values

            xx_train = x_train.copy()
            xx_test = x_test.copy()
            if standardize:
                std_means = []
                std_stds = []
                for i in range(xx_train.shape[1]):
                    std_mean = xx_train[:, i].mean()
                    std_std = xx_train[:, i].std()
                    xx_train[:, i] = (xx_train[:, i] - std_mean) / std_std
                    xx_test[:, i] = (xx_test[:, i] - std_mean) / std_std
                    std_means.append(std_mean)
                    std_stds.append(std_std)
                    if numpy.ma.masked_invalid(xx_train[:, i]).mask.all():
                        xx_train[:, i] = 0
                    else:
                        xx_train[pandas.isna(xx_train[:, i]), i] = xx_train[
                            ~pandas.isna(xx_train[:, i]), i].mean()
                    if numpy.ma.masked_invalid(xx_test[:, i]).mask.all():
                        xx_test[:, i] = 0
                    else:
                        xx_test[pandas.isna(xx_test[:, i]), i] = xx_test[
                            ~pandas.isna(xx_test[:, i]), i].mean()
                    xx_train[xx_train[:, i] == numpy.inf, i] = xx_train[
                        xx_train[:, i] != numpy.inf, i].max()
                    xx_train[xx_train[:, i] == -numpy.inf, i] = xx_train[
                        xx_train[:, i] != -numpy.inf, i].min()
                    xx_test[xx_test[:, i] == numpy.inf, i] = xx_test[
                        xx_test[:, i] != numpy.inf, i].max()
                    xx_test[xx_test[:, i] == -numpy.inf, i] = xx_test[
                        xx_test[:, i] != -numpy.inf, i].min()

            lm = LinearRegression(n_jobs=-1)
            try:
                lm.fit(X=xx_train, y=y_train)
                yy_hat_test = lm.predict(X=xx_test)
                r2_tt2st_wo = r2_score(y_true=y_test, y_pred=yy_hat_test)
                sd_tt2st_wo = somersd(x=y_test, y=yy_hat_test)  # .statistic
            except numpy.linalg.LinAlgError as e:
                r2_tt2st_wo = numpy.nan
                sd_tt2st_wo = numpy.nan


            if r2_tt2st_wo > 1:
                r2_tt2st_wo = numpy.nan
            elif r2_tt2st_wo < -1:
                r2_tt2st_wo = numpy.nan
            if sd_tt2st_wo > 1:
                sd_tt2st_wo = numpy.nan
            elif sd_tt2st_wo < -1:
                sd_tt2st_wo = numpy.nan

            res = pandas.DataFrame(data={'drivers': [x_left[j]],
                                         'perf_r2_w': [r2_tt2st_w],
                                         'perf_sd_w': [sd_tt2st_w],
                                         'perf_r2_wo': [r2_tt2st_wo],
                                         'perf_sd_wo': [sd_tt2st_wo],
                                         'k': [k]})
            respox.append(res)
    respox = pandas.concat(respox, axis=0, ignore_index=True)
    respox['diff_r2'] = respox['perf_r2_w'] - respox['perf_r2_wo']
    respox['diff_sd'] = respox['perf_sd_w'] - respox['perf_sd_wo']
    respox_mean = respox.groupby(by=['drivers'])[['perf_r2_w', 'diff_r2', 'perf_sd_w', 'diff_sd', 'perf_sd_wo']].mean()
    respox_median = respox.groupby(by=['drivers'])[['perf_r2_w', 'diff_r2', 'perf_sd_w', 'diff_sd', 'perf_sd_wo']].median()
    respox_min = respox.groupby(by=['drivers'])[['perf_r2_w', 'diff_r2', 'perf_sd_w', 'diff_sd', 'perf_sd_wo']].min()

    best_candidate = respox_median[respox_median['perf_sd_wo'] == respox_median['perf_sd_wo'].max()].index[0]
    best_candidate_perf = respox_median.loc[best_candidate, 'perf_sd_wo']

    if best_candidate_perf > previous_perf:
        stop = False
        x_base += [best_candidate]
        previous_perf = best_candidate_perf
    else:
        stop = True

    if len(x_base) == len(x_factors_selection):
        stop = True

run_time = time.time() - run_time
print('Run time: {0:.2f} minutes'.format(run_time / 60))
