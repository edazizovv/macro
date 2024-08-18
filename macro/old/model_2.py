
#


#
import numpy
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

x_factors = [x for x in result.columns if x != target]

factors = [x_factors,
           x_factors,
           x_factors,
           x_factors,
           x_factors,
           x_factors]
win_type = ['rolling', 'rolling', 'rolling', 'ewm', 'ewm', 'ewm']
win_kwgs = [{'win_func': None, 'agg_func': 'mean', 'window': 4},
            {'win_func': None, 'agg_func': 'mean', 'window': 8},
            {'win_func': None, 'agg_func': 'mean', 'window': 16},
            {'alpha': 2 / (4 + 1), 'agg_func': 'mean'},
            {'alpha': 2 / (8 + 1), 'agg_func': 'mean'},
            {'alpha': 2 / (16 + 1), 'agg_func': 'mean'}]
_ = f.represent(factors=factors, win_type=win_type, win_kwgs=win_kwgs)

result = f.tighten_dates(cutoff_date='2007-01-01')
x_factors = [x for x in result.columns if x != target]
print('N of factors considered: {0}'.format(len(x_factors)))
time_axis = result.index


top_10 = ['FRGSHPUSM649NCIS',
 'PCEDG__rolling_4__None__mean',
 'DNRGRC1M027SBEA__rolling_16__None__mean',
 'JTU5300QUL__rolling_4__None__mean',
 'DMOTRC1Q027SBEA__rolling_8__None__mean',
 'USTRADE__rolling_4__None__mean',
 'BOPGSTB',
 'GS1M__rolling_16__None__mean',
 'PCU4841214841212',
 'A091RC1Q027SBEA__rolling_4__None__mean']

random_10 = ['SPASTT01EZM657N__rolling_8__None__mean',
 'PCEC96',
 'QUSR628BIS__rolling_8__None__mean',
 'SPASTT01GBM657N__rolling_4__None__mean',
 'DHUTRC1Q027SBEA__rolling_4__None__mean',
 'BAA__rolling_8__None__mean',
 'SPASTT01EZM657N__rolling_8__None__mean',
 'EMRATIO',
 'IIPUSASSQ__rolling_4__None__mean',
 'SPASTT01KRM657N__rolling_8__None__mean']

total_random_10 = ['SPASTT01KRM657N__rolling_4__None__mean',
 'DFXARC1M027SBEA__rolling_8__None__mean',
 'CE16OV__rolling_8__None__mean',
 'TERMCBCCALLNS__rolling_4__None__mean',
 'PCEDG',
 'KCFSI__rolling_8__None__mean',
 'T20YIEM__rolling_4__None__mean',
 'GS3__ewm_0.4__mean',
 'SPASTT01KRM657N__rolling_4__None__mean',
 'PMSAVE__ewm_0.11764705882352941__mean']

standardize = False
time_sub_rate = 0.3
time_sub_replace = True
nt = int(time_axis.shape[0] * time_sub_rate)
r2_trains = []
r2_tests = []
ks = []
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    for k in range(40):
        print(k)

        ix_time = numpy.random.choice(time_axis.values, size=(nt,), replace=time_sub_replace)
        x_base = total_random_10
        left_time = [x for x in result.index if x not in ix_time]
        x_train = result.loc[ix_time, x_base].values[:-1]
        y_train = result.loc[ix_time, target].values[1:]
        x_test = result.loc[left_time, x_base].values[:-1]
        y_test = result.loc[left_time, target].values[1:]

        xx_micro_train = x_train.copy()
        xx_micro_test = x_test.copy()
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

        micro_model = LinearRegression()
        micro_model.fit(X=xx_micro_train, y=y_train)
        y_micro_hat_train = micro_model.predict(X=xx_micro_train)
        y_micro_hat_test = micro_model.predict(X=xx_micro_test)
        r2_train = r2_score(y_true=y_train, y_pred=y_micro_hat_train)
        r2_test = r2_score(y_true=y_test, y_pred=y_micro_hat_test)

        r2_trains.append(r2_train)
        r2_tests.append(r2_test)
        ks.append(k)

    reported = pandas.DataFrame(data={'r2_train': r2_trains,
                                      'r2_test': r2_tests,
                                      'k': ks})
run_time = time.time() - run_time
print('Run time: {0:.2f} minutes'.format(run_time / 60))

