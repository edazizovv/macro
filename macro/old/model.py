#


#
import numpy
import scipy
import pandas
from sklearn.linear_model import enet_path, LinearRegression
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from newborn import FrameOld
from sklearn.metrics import r2_score
from sklearn.exceptions import ConvergenceWarning
# ConvergenceWarning('ignore')
import warnings
import time

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
target = 'TLT'
target_e = f[target]
time_axis = target_e.frame['DATE'].copy()
_ = f.cast(time_axis=time_axis, gb_funcs='last', fill_values='ffill')

factors = []
win_type = []
win_kwgs = []
_ = f.represent(factors=factors, win_type=win_type, win_kwgs=win_kwgs)

result = f.tighten_dates(cutoff_date='2007-01-01')
# why duplicated???
# result = result.loc[:, ~result.columns.duplicated()].copy()

x_factors = result.columns.values

factors = [x_factors,
           x_factors,
           x_factors,
           x_factors,
           x_factors,
           x_factors,
           x_factors]
win_type = ['rolling', 'rolling', 'rolling', 'ewm', 'ewm', 'ewm', 'pct']
win_kwgs = [{'win_func': None, 'agg_func': 'mean', 'window': 4},
            {'win_func': None, 'agg_func': 'mean', 'window': 8},
            {'win_func': None, 'agg_func': 'mean', 'window': 16},
            {'alpha': 2 / (4 + 1), 'agg_func': 'mean'},
            {'alpha': 2 / (8 + 1), 'agg_func': 'mean'},
            {'alpha': 2 / (16 + 1), 'agg_func': 'mean'},
            {}]
_ = f.represent(factors=factors, win_type=win_type, win_kwgs=win_kwgs)

result = f.tighten_dates(cutoff_date='2007-01-01')
x_factors = result.columns.values
print('N of factors considered: {0}'.format(len(x_factors)))
time_axis = result.index
target = 'TLT__pct'

standardize = True
x_sub_rate = 0.25
x_sub_replace = False
nx = int(len(x_factors) * x_sub_rate)
time_sub_rate = 0.5
time_sub_replace = True
nt = int(time_axis.shape[0] * time_sub_rate)
rere = []
x_bases = []
# up_to = 0.2
print('Takes {0} out of {1} factors ({2:.4f} rate); {3} out of {4} tx ({5:.4f} rate)'.format(nx, len(x_factors), x_sub_rate, nt, time_axis.shape[0], time_sub_rate))
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    for k in range(40):
        print(k)

        ix_time = numpy.random.choice(time_axis.values, size=(nt,), replace=time_sub_replace)
        x_base = numpy.random.choice(x_factors, size=(nx,), replace=x_sub_replace)
        left_time = [x for x in result.index if x not in ix_time]
        x_bases.append(x_base)
        x_train = result.loc[ix_time, x_base].values[:-1]
        y_train = result.loc[ix_time, target].values[1:]
        x_test = result.loc[left_time, x_base].values[:-1]
        y_test = result.loc[left_time, target].values[1:]

        mm = Model(standardize=standardize)
        mm.fit(x=x_train, y=y_train)


        def fu(x):
            if x.any():
                return numpy.where(x)[0][0]
            else:
                return -1


        enter_step = pandas.DataFrame((numpy.abs(mm.coefs_enet) > 0)).apply(fu, axis=1).sort_values()


        def scored_step(x):
            if x == -1:
                return 0.01
            else:
                return 1 - (numpy.log(x) / (1 + numpy.log(x)))


        scored = enter_step.apply(func=scored_step)

        improvements = []
        r2_total = []
        r2_trains = []
        # upp_to = int(up_to * enter_step.shape[0])
        for j in range(enter_step.shape[0]):
            step = enter_step.values[j]
            if step > -1:
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
                if r2_test > 1:
                    r2_test = 1
                elif r2_test < -1:
                    r2_test = -1
                if len(improvements) == 0:
                    improvement = r2_test
                else:
                    if r2_total[-1] <= 0:
                        improvement = 0
                    else:
                        improvement = r2_test - r2_total[-1]
            else:
                improvement = 0
                r2_test = 0
                r2_train = 0
            improvements.append(improvement)
            r2_total.append(r2_test)
            r2_trains.append(r2_train)

        reported = pandas.DataFrame(data={'driver': [x_factors[j] for j in enter_step.index],
                                          'enter_step': enter_step.values,
                                          'step_score': scored.values,
                                          'r2_improvement': improvements,
                                          'r2_total': r2_total,
                                          'r2_train': r2_trains})
        reported['k'] = k
        rere.append(reported)
    reported = pandas.concat(rere, axis=0, ignore_index=True)
    reported_mean = reported.groupby(by=['driver'])[['enter_step', 'r2_improvement', 'r2_total', 'r2_train']].mean()
    reported_min = reported.groupby(by=['driver'])[['enter_step', 'r2_improvement', 'r2_total', 'r2_train']].min()

    reported_mean = reported_mean.sort_values(by='enter_step', ascending=True)
    reported_min = reported_min.sort_values(by='enter_step', ascending=True)
run_time = time.time() - run_time
print('Run time: {0:.2f} minutes'.format(run_time / 60))

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


def linreg_relative(y):
    yy = y.values
    xx = numpy.array(numpy.arange(yy.shape[0]))
    rs = scipy.stats.linregress(x=xx, y=yy)
    tt = xx * rs.slope + rs.intercept
    r = y[-1] / tt[-1] - 1
    return r

from functools import partial


def ewm_relative_3(y):
    window = y.shape[0]
    alpha = 2 / (3 + 1)
    weights = list(reversed([(1-alpha) ** n for n in range(window)]))
    ewma = partial(numpy.average, weights=weights)
    ra = ewma(y) / y[-1] - 1
    return ra
