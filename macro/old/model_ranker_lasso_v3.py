#


#
import numpy
import scipy
import pandas
from sklearn.linear_model import enet_path, LinearRegression
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

target = 'IVV_MEAN__pct'
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

        mm = Model(standardize=standardize)
        mm.fit(x=x_train, y=y_train)


        def fu(x):
            if x.any():
                return numpy.where(x)[0][0]
            else:
                return numpy.nan


        enter_step = pandas.DataFrame((numpy.abs(mm.coefs_enet) > 0)).apply(fu, axis=1).sort_values()


        def scored_step(x):
            if x == -1:
                return 0.01
            else:
                return 1 - (numpy.log(x) / (1 + numpy.log(x)))


        scored = enter_step.apply(func=scored_step)

        improvements = []
        s_impr = []
        sds_train = []
        sds_test = []
        r2_total = []
        r2_trains = []
        enters = []
        # upp_to = int(up_to * enter_step.shape[0])
        for j in range(enter_step.shape[0]):
            step = enter_step.values[j]
            if not pandas.isna(step):
                x_ixs = enter_step.index.values[:(j + 1)]
                x_micro_train = x_train[:, x_ixs]
                x_micro_test = x_test[:, x_ixs]
                xx_micro_train = x_micro_train.copy()
                xx_micro_test = x_micro_test.copy()
                if standardize:
                    std_means = []
                    std_stds = []
                    for i in range(xx_micro_train.shape[1]):
                        std_mean = xx_micro_train[:, i].mean()
                        std_std = xx_micro_train[:, i].std()
                        xx_micro_train[:, i] = (xx_micro_train[:, i] - std_mean) / std_std
                        xx_micro_test[:, i] = (xx_micro_test[:, i] - std_mean) / std_std
                        std_means.append(std_mean)
                        std_stds.append(std_std)
                        if numpy.ma.masked_invalid(xx_micro_train[:, i]).mask.all():
                            xx_micro_train[:, i] = 0
                        else:
                            xx_micro_train[pandas.isna(xx_micro_train[:, i]), i] = xx_micro_train[~pandas.isna(xx_micro_train[:, i]), i].mean()
                        if numpy.ma.masked_invalid(xx_micro_test[:, i]).mask.all():
                            xx_micro_test[:, i] = 0
                        else:
                            xx_micro_test[pandas.isna(xx_micro_test[:, i]), i] = xx_micro_test[~pandas.isna(xx_micro_test[:, i]), i].mean()
                        xx_micro_train[xx_micro_train[:, i] == numpy.inf, i] = xx_micro_train[xx_micro_train[:, i] != numpy.inf, i].max()
                        xx_micro_train[xx_micro_train[:, i] == -numpy.inf, i] = xx_micro_train[xx_micro_train[:, i] != -numpy.inf, i].min()
                        xx_micro_test[xx_micro_test[:, i] == numpy.inf, i] = xx_micro_test[xx_micro_test[:, i] != numpy.inf, i].max()
                        xx_micro_test[xx_micro_test[:, i] == -numpy.inf, i] = xx_micro_test[xx_micro_test[:, i] != -numpy.inf, i].min()
                micro_model = LinearRegression()
                micro_model.fit(X=xx_micro_train, y=y_train)
                y_micro_hat_train = micro_model.predict(X=xx_micro_train)
                y_micro_hat_test = micro_model.predict(X=xx_micro_test)
                r2_train = r2_score(y_true=y_train, y_pred=y_micro_hat_train)
                r2_test = r2_score(y_true=y_test, y_pred=y_micro_hat_test)
                t_sd = somersd(x=y_train, y=y_micro_hat_train)  # .statistic
                s_sd = somersd(x=y_test, y=y_micro_hat_test)  # .statistic
                if r2_test > 1:
                    r2_test = numpy.nan
                elif r2_test < -1:
                    r2_test = numpy.nan
                if t_sd > 1:
                    t_sd = numpy.nan
                elif t_sd < -1:
                    t_sd = numpy.nan
                if len(improvements) == 0:
                    improvement = r2_test
                    s_imp = s_sd
                else:
                    improvement = r2_test - r2_total[-1]
                    s_imp = s_sd - sds_test[-1]
                enter = 1
            else:
                improvement = numpy.nan
                r2_test = numpy.nan
                r2_train = numpy.nan
                s_imp = numpy.nan
                t_sd = numpy.nan
                s_sd = numpy.nan
                enter = 0
            improvements.append(improvement)
            s_impr.append(s_imp)
            sds_train.append(t_sd)
            sds_test.append(s_sd)
            r2_total.append(r2_test)
            r2_trains.append(r2_train)
            enters.append(enter)

        reported = pandas.DataFrame(data={'driver': [x_base[j] for j in enter_step.index],
                                          'enter_step': enter_step.values,
                                          'step_score': scored.values,
                                          'r2_improvement': improvements,
                                          'r2_total': r2_total,
                                          'r2_train': r2_trains,
                                          'sd_improvement': s_impr,
                                          'sd_train': sds_train,
                                          'sd_test': sds_test,
                                          'enter': enters
                                          })
        reported['k'] = k
        rere.append(reported)

reported = pandas.concat(rere, axis=0, ignore_index=True)
reported_mean = reported.groupby(by=['driver'])[['enter_step', 'r2_improvement', 'r2_total', 'r2_train', 'sd_improvement', 'sd_train', 'sd_test']].mean()
reported_median = reported.groupby(by=['driver'])[['enter_step', 'r2_improvement', 'r2_total', 'r2_train', 'sd_improvement', 'sd_train', 'sd_test']].median()
reported_min = reported.groupby(by=['driver'])[['enter_step', 'r2_improvement', 'r2_total', 'r2_train', 'sd_improvement', 'sd_train', 'sd_test']].min()

reported_mean = reported_mean.sort_values(by='enter_step', ascending=True)
reported_median = reported_median.sort_values(by='enter_step', ascending=True)
reported_min = reported_min.sort_values(by='enter_step', ascending=True)

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
x_base = reported_mean['enter_step'].sort_values().dropna().index.values.tolist()
x_base.reverse()

if len(x_base) == 0:
    raise Exception("No predictors found")

time_sub_rate = 0.50
time_sub_replace = False
nt = int(time_axis.shape[0] * time_sub_rate)
n = 100
r2_trains_z_up, r2_tests_z_up = [], []
r2_trains_z_mean, r2_tests_z_mean = [], []
r2_trains_z_down, r2_tests_z_down = [], []
kt_trains_z_up, kt_tests_z_up = [], []
kt_trains_z_mean, kt_tests_z_mean = [], []
kt_trains_z_down, kt_tests_z_down = [], []
per_p_kst = []
for i in range(len(x_base) - 1):
    current_base = x_base[:i + 1]
    r2_trains, r2_tests = [], []
    kt_trains, kt_tests = [], []
    kt_trains_bin, kt_tests_bin = [], []
    for j in range(n):
        import numpy

        # ix_time = numpy.random.choice(time_axis.values, size=(nt,), replace=time_sub_replace)
        ixes = numpy.random.choice(list(range(time_axis.shape[0] - 1)), size=(nt,), replace=time_sub_replace)
        x_ix_train = time_axis.values[:-1][ixes]
        y_ix_train = time_axis.values[1:][ixes]
        left_ixes = [x for x in list(range(time_axis.shape[0] - 1)) if x not in ixes]
        x_ix_test = time_axis.values[:-1][left_ixes]
        y_ix_test = time_axis.values[1:][left_ixes]

        x_train = result.loc[x_ix_train, current_base].values
        y_train = result.loc[y_ix_train, target].values
        x_test = result.loc[x_ix_test, current_base].values
        y_test = result.loc[y_ix_test, target].values

        xxx_train, xxx_test, yyy_train, yyy_test = x_train, x_test, y_train, y_test
        from sklearn.preprocessing import StandardScaler

        sk = StandardScaler()
        xxx_train_st = sk.fit_transform(X=xxx_train)
        xxx_test_st = sk.transform(X=xxx_test)
        from sklearn.linear_model import LinearRegression

        m = LinearRegression()
        m.fit(X=xxx_train_st, y=yyy_train)
        y_hat_train = m.predict(X=xxx_train_st)
        y_hat_test = m.predict(X=xxx_test_st)
        from sklearn.metrics import r2_score

        r2_train = r2_score(y_true=yyy_train, y_pred=y_hat_train)
        r2_test = r2_score(y_true=yyy_test, y_pred=y_hat_test)
        from scipy.stats import kendalltau

        kt_train = kendalltau(x=yyy_train, y=y_hat_train).statistic
        kt_test = kendalltau(x=yyy_test, y=y_hat_test).statistic
        from matplotlib import pyplot

        # fig, ax = pyplot.subplots(1, 2)
        # pandas.DataFrame(data={'error': yyy_train - y_hat_train}).hist(bins=20, ax=ax[0])
        # pandas.DataFrame(data={'error': yyy_test - y_hat_test}).hist(bins=20, ax=ax[1])
        r2_trains.append(r2_train)
        r2_tests.append(r2_test)
        kt_trains.append(kt_train)
        kt_tests.append(kt_test)
    r2_train_up = numpy.array(r2_trains).max()
    r2_train_mean = numpy.array(r2_trains).mean()
    r2_train_down = numpy.array(r2_trains).min()
    r2_test_up = numpy.array(r2_tests).max()
    r2_test_mean = numpy.array(r2_tests).mean()
    r2_test_down = numpy.array(r2_tests).min()
    kt_train_up = numpy.array(kt_trains).max()
    kt_train_mean = numpy.array(kt_trains).mean()
    kt_train_down = numpy.array(kt_trains).min()
    kt_test_up = numpy.array(kt_tests).max()
    kt_test_mean = numpy.array(kt_tests).mean()
    kt_test_down = numpy.array(kt_tests).min()
    r2_trains_z_up.append(r2_train_up)
    r2_trains_z_mean.append(r2_train_mean)
    r2_trains_z_down.append(r2_train_down)
    r2_tests_z_up.append(r2_test_up)
    r2_tests_z_mean.append(r2_test_mean)
    r2_tests_z_down.append(r2_test_down)
    kt_trains_z_up.append(kt_train_up)
    kt_trains_z_mean.append(kt_train_mean)
    kt_trains_z_down.append(kt_train_down)
    kt_tests_z_up.append(kt_test_up)
    kt_tests_z_mean.append(kt_test_mean)
    kt_tests_z_down.append(kt_test_down)
    print(i)

from scipy import stats
max_mean = max(kt_tests_z_mean)
max_mean_ix = kt_tests_z_mean.index(max_mean)
s = numpy.std(kt_tests_z_mean, ddof=1)
alpha = 0.05
max_mean_lower_ci = max_mean + stats.t.ppf(q=(alpha / 2), df=(n - 1)) * s / (n ** 0.5)
taken_ix = numpy.where(kt_tests_z_mean >= max_mean_lower_ci)[0][0]


current_base = x_base[:taken_ix+1]
# pyplot.plot(list(range(len(x_base)-1)), kt_tests_z_mean)
# pyplot.fill_between(list(range(len(x_base)-1)), kt_tests_z_down, kt_tests_z_up, alpha=.3)

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


run_time = time.time() - run_time
print('Run time: {0:.2f} minutes'.format(run_time / 60))
