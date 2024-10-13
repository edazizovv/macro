#


#
import numpy
import pandas
from matplotlib import pyplot
from sklearn.linear_model import enet_path, LinearRegression
from sklearn.metrics import r2_score
from scipy.stats import kendalltau
from macro.functional import sd_metric


#


#
def r2_metric(x, y):
    metered = r2_score(y_pred=x, y_true=y)
    return metered


def kendalltau_metric(x, y):
    metered = kendalltau(x=x, y=y).statistic
    return metered


def somersd_metric(x, y):
    metered = sd_metric(x=x, y=y)
    return metered


class BasicLinearModel:
    def __init__(self, score, metrics, **kwargs):
        self.kwargs = {**kwargs}
        self._model = LinearRegression
        self.model = None
        self._score = score
        self._metrics = metrics
    def fit(self, x, y):
        self.model = self._model(**self.kwargs)
        self.model.fit(X=x, y=y)
    def predict(self, x):
        y_hat = self.model.predict(X=x)
        return y_hat
    def score(self, x, y):
        y_hat = self.predict(x=x)
        scored = self._score(x=y_hat, y=y)
        return scored
    def metrics(self, x, y):
        metered = {key: self._metrics[key](x=self.predict(x=x), y=y) for key in self._metrics}
        return metered


class BasicLassoSelectorModel:
    def __init__(self, l1_ratio=0.9, eps=1e-12, n_alphas=1_000):
        self.l1_ratio = l1_ratio
        self.eps = eps
        self.n_alphas = n_alphas
        self.alphas_enet = None
        self.coefs_enet = None
        self.neg_log_alphas_lasso = None
        self.std_mean = None
        self.std_std = None
        self.enter_step = None
        self.features = None
    def fit(self, x, y):
        self.features = x.columns.values
        xx = x.values.copy()
        self.alphas_enet, self.coefs_enet, _ = enet_path(X=xx, y=y, eps=self.eps, l1_ratio=self.l1_ratio, n_alphas=self.n_alphas)
        self.neg_log_alphas_lasso = -numpy.log10(self.alphas_enet)
        self.estimate_ranker_frame()
    def plot(self, neg=True):
        if neg:
            x_axis = self.neg_log_alphas_lasso
        else:
            x_axis = self.alphas_enet
        for coef_e in self.coefs_enet:
            pyplot.plot(x_axis, coef_e)
    def estimate_ranker_frame(self):
        def fu(x):
            if x.any():
                return numpy.where(x)[0][0]
            else:
                return numpy.nan

        self.enter_step = pandas.DataFrame(data={'feature': self.features})
        self.enter_step['rank'] = pandas.DataFrame((numpy.abs(self.coefs_enet) > 0)).apply(fu, axis=1)
        self.enter_step = self.enter_step.sort_values(by='rank')
        self.enter_step['ranking'] = list(range(self.enter_step.shape[0]))
    @property
    def ranking(self):
        return self.enter_step['ranking'].values.tolist()
    @property
    def ranks(self):
        return self.enter_step['rank'].values.tolist()
    @property
    def ranking_features(self):
        return self.enter_step['feature'].values.tolist()
