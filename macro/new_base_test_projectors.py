#
import json
import time
import copy
import datetime
import warnings

#
import numpy
import pandas
import pmdarima
from scipy import stats
from sklearn.preprocessing import KBinsDiscretizer
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import kpss
# from statsmodels.tools.sm_exceptions import InterpolationWarning, ConvergenceWarning


#
from macro.new_base import UnholyVice
from macro._new_constants import ValueTypes
from macro.new_base_utils import my_hex, my_dict_hex, my_func_hex


#
# warnings.simplefilter('ignore', InterpolationWarning)
# warnings.simplefilter('ignore', ConvergenceWarning)

class Stayer:
    def __init__(self, method):
        self.method = method
        self.value_type = ValueTypes.CONTINUOUS
        self.impute_max = numpy.nan
        self.impute_min = numpy.nan
        self.wards = None
    @property
    def parametrization(self):
        dictated = tuple([my_hex('STAYER'),
                          my_hex(self.value_type)])
        return dictated
    @property
    def parametrization_hash(self):
        hashed = my_hex(self.parametrization)
        return hashed
    def project_first(self, series_dict):
        assert len(list(series_dict.keys())) == 1
        result = series_dict[list(series_dict.keys())[0]].copy()
        result = result.pct_change()
        result = (result + 1).apply(func=numpy.log)
        mask_positive_inf = result == numpy.inf
        mask_negative_inf = result == -numpy.inf
        self.impute_max = result[~mask_positive_inf].max()
        self.impute_min = result[~mask_negative_inf].min()
        result[mask_positive_inf] = self.impute_max
        result[mask_negative_inf] = self.impute_min
        self.wards = result.dropna().copy()
        if self.method == 'min_max':
            result = (result - self.wards.min()) / (self.wards.max() - self.wards.min())
        elif self.method == 'percentile':
            result = result.apply(func=lambda x: stats.percentileofscore(self.wards.values, x) / 100)
        else:
            raise Exception("wrong method provided for Binner class projector")
        return result
    def project_second(self, series_dict):
        assert len(list(series_dict.keys())) == 1
        result = series_dict[list(series_dict.keys())[0]].copy()
        result = result.pct_change()
        result = (result + 1).apply(func=numpy.log)
        mask_positive_inf = result == numpy.inf
        mask_negative_inf = result == -numpy.inf
        result[mask_positive_inf] = self.impute_max
        result[mask_negative_inf] = self.impute_min
        if self.method == 'min_max':
            result = (result - self.wards.min()) / (self.wards.max() - self.wards.min())
        elif self.method == 'percentile':
            result = result.apply(func=lambda x: stats.percentileofscore(self.wards.values, x) / 100)
        else:
            raise Exception("wrong method provided for Binner class projector")
        return result
    @property
    def lag(self):
        return 0


class Binner:
    def __init__(self, n_bins, method):
        self.n_bins = n_bins
        self.method = method
        self.value_type = ValueTypes.CONTINUOUS
        self.impute_max = numpy.nan
        self.impute_min = numpy.nan
        self._binner = KBinsDiscretizer
        self._binner_kwargs = {'encode': 'ordinal', 'strategy': 'quantile'}
        self.binner = None
        self.wards = None
    @property
    def parametrization(self):
        dictated = tuple([my_hex('Binner'),
                          my_hex(self.n_bins),
                          my_hex(self.method),
                          my_hex(self.value_type),
                          my_func_hex(self._binner),
                          my_hex(tuple([self._binner_kwargs['encode'],
                                        self._binner_kwargs['strategy']]))])
        return dictated
    @property
    def parametrization_hash(self):
        hashed = my_hex(self.parametrization)
        return hashed
    def project_first(self, series_dict):
        assert len(list(series_dict.keys())) == 1
        result = series_dict[list(series_dict.keys())[0]].copy()
        self.binner = self._binner(n_bins=self.n_bins, **self._binner_kwargs)
        result = result.pct_change()
        result = (result + 1).apply(func=numpy.log)
        mask_positive_inf = result == numpy.inf
        mask_negative_inf = result == -numpy.inf
        self.impute_max = result[~mask_positive_inf].max()
        self.impute_min = result[~mask_negative_inf].min()
        result[mask_positive_inf] = self.impute_max
        result[mask_negative_inf] = self.impute_min
        nan_mask = pandas.isna(result)
        self.binner.fit(X=result.dropna().values.reshape(-1, 1))
        if self.method == 'min_max':
            impute = pandas.DataFrame(data={'values': result.dropna().values})
            impute['bins'] = self.binner.transform(X=result.dropna().values.reshape(-1, 1))
            impute_agg = impute.groupby(by='bins')['values'].mean()
            self.wards = impute_agg.copy()
            self.wards = pandas.Series(index=numpy.arange(start=self.wards.index.min(), stop=self.wards.index.max() + 1, step=1), data=self.wards)
            self.wards = self.wards.interpolate()
            output = pandas.Series(index=result.index)
            output[~nan_mask] = pandas.Series(self.binner.transform(X=result.dropna().values.reshape(-1, 1))[:, 0]).apply(func=lambda x: (self.wards[x] - self.wards.min()) / (self.wards.max() - self.wards.min())).values
            output[nan_mask] = numpy.nan
        elif self.method == 'percentile':
            output = pandas.Series(index=result.index)
            output[~nan_mask] = self.binner.transform(X=result.dropna().values.reshape(-1, 1))[:, 0] / self.n_bins
            output[nan_mask] = numpy.nan
        else:
            raise Exception("wrong method provided for Binner class projector")
        return output
    def project_second(self, series_dict):
        assert len(list(series_dict.keys())) == 1
        result = series_dict[list(series_dict.keys())[0]].copy()
        result = result.pct_change()
        result = (result + 1).apply(func=numpy.log)
        mask_positive_inf = result == numpy.inf
        mask_negative_inf = result == -numpy.inf
        result[mask_positive_inf] = self.impute_max
        result[mask_negative_inf] = self.impute_min
        nan_mask = pandas.isna(result)
        if self.method == 'min_max':
            def fix_bin(x):
                if x > self.wards.index.max():
                    return self.wards.index.max()
                elif x < self.wards.index.min():
                    return self.wards.index.min()
                else:
                    return x
            output = pandas.Series(index=result.index)
            output[~nan_mask] = pandas.Series(self.binner.transform(X=result.dropna().values.reshape(-1, 1))[:, 0]).apply(func=fix_bin).apply(func=lambda x: (self.wards[x] - self.wards.min()) / (self.wards.max() - self.wards.min())).values
            output[nan_mask] = numpy.nan
        elif self.method == 'percentile':
            output = pandas.Series(index=result.index)
            output[~nan_mask] = self.binner.transform(X=result.dropna().values.reshape(-1, 1))[:, 0] / self.n_bins
            output[nan_mask] = numpy.nan
        else:
            raise Exception("wrong method provided for Binner class projector")
        return result
    @property
    def lag(self):
        return 0


class RangedPct:
    def __init__(self, shift, log=True):
        self.shift = shift
        self.value_type = ValueTypes.CONTINUOUS
        self.impute_max = numpy.nan
        self.impute_min = numpy.nan
        self.log = log
    @property
    def parametrization(self):
        dictated = tuple([my_hex('RangedPct'),
                          my_hex(self.shift),
                          my_hex(self.value_type),
                          my_hex(self.log)])
        return dictated
    @property
    def parametrization_hash(self):
        hashed = my_hex(self.parametrization)
        return hashed
    def project_first(self, series_dict):
        assert len(list(series_dict.keys())) == 1
        result = series_dict[list(series_dict.keys())[0]].copy()
        result = result.pct_change(periods=self.shift)
        if self.log:
            result = (result + 1).apply(func=numpy.log)
        mask_positive_inf = result == numpy.inf
        mask_negative_inf = result == -numpy.inf
        self.impute_max = result[~mask_positive_inf].max()
        self.impute_min = result[~mask_negative_inf].min()
        result[mask_positive_inf] = self.impute_max
        result[mask_negative_inf] = self.impute_min
        return result
    def project_second(self, series_dict):
        assert len(list(series_dict.keys())) == 1
        result = series_dict[list(series_dict.keys())[0]].copy()
        result = result.pct_change(periods=self.shift)
        mask_positive_inf = result == numpy.inf
        mask_negative_inf = result == -numpy.inf
        result[mask_positive_inf] = self.impute_max
        result[mask_negative_inf] = self.impute_min
        return result
    @property
    def lag(self):
        return self.shift


class WindowRollImpulse:
    def __init__(self, func, smaller_window, bigger_window, pct, shift, operation, log=True):
        self.func = func
        self.smaller_window = smaller_window
        self.bigger_window = bigger_window
        self.operation = operation
        self.pct = pct
        self.shift = shift
        self.value_type = ValueTypes.CONTINUOUS
        self.impute_max = numpy.nan
        self.impute_min = numpy.nan
        self.log = log
        self.train_stored = None
    @property
    def parametrization(self):
        dictated = tuple([my_hex('WindowRollImpulse'),
                          my_func_hex(self.func),
                          my_hex(self.smaller_window),
                          my_hex(self.bigger_window),
                          my_hex(self.operation),
                          my_hex(self.pct),
                          my_hex(self.shift),
                          my_hex(self.value_type),
                          my_hex(self.log)])
        return dictated
    @property
    def parametrization_hash(self):
        hashed = my_hex(self.parametrization)
        return hashed
    def project_first(self, series_dict):
        assert len(list(series_dict.keys())) == 1
        result = series_dict[list(series_dict.keys())[0]].copy()
        if self.pct:
            result = result.pct_change(periods=self.shift)
            if self.log:
                result = (result + 1).apply(func=numpy.log)
        mask_positive_inf = result == numpy.inf
        mask_negative_inf = result == -numpy.inf
        self.impute_max = result[~mask_positive_inf].max()
        self.impute_min = result[~mask_negative_inf].min()
        result[mask_positive_inf] = self.impute_max
        result[mask_negative_inf] = self.impute_min
        if self.operation == 'division':
            def dister(x):
                smaller_p = stats.percentileofscore(self.train_stored.values, x['smaller']) / 100
                bigger_p = stats.percentileofscore(self.train_stored.values, x['bigger']) / 100
                r = smaller_p - bigger_p
                return r
            self.train_stored = result[~mask_positive_inf].dropna().copy()
            result_smaller = result.rolling(window=self.smaller_window).apply(func=self.func)
            result_bigger = result.rolling(window=self.bigger_window).apply(func=self.func)
            result = pandas.DataFrame(data={'smaller': result_smaller, 'bigger': result_bigger}).apply(func=dister, axis=1)
            """
            from matplotlib import pyplot
            fig, ax = pyplot.subplots()
            reg = pandas.DataFrame(data={'origin': result, 'smaller': result_smaller, 'bigger': result_bigger, 'final': pandas.DataFrame(data={'smaller': result_smaller, 'bigger': result_bigger}).apply(func=dister, axis=1)})
            reg['zero'] = 0
            reg.plot(y=['origin', 'smaller', 'bigger', 'zero'], ax=ax)
            reg.plot(y=['final', 'zero'], ax=ax, secondary_y=True)
            pyplot.savefig("squares.png")
            """
        elif self.operation == 'quantile':
            def quantilizer(x):
                r = stats.percentileofscore(x.values, self.func(x.values[-self.smaller_window:])) / 100
                return r
            result = result.rolling(window=self.bigger_window).apply(func=quantilizer)
        else:
            raise Exception("Invalid operation specified")
        return result
    def project_second(self, series_dict):
        assert len(list(series_dict.keys())) == 1
        result = series_dict[list(series_dict.keys())[0]].copy()
        if self.pct:
            result = result.pct_change(periods=self.shift)
            if self.log:
                result = (result + 1).apply(func=numpy.log)
        mask_positive_inf = result == numpy.inf
        mask_negative_inf = result == -numpy.inf
        result[mask_positive_inf] = self.impute_max
        result[mask_negative_inf] = self.impute_min
        if self.operation == 'division':
            def dister(x):
                smaller_p = stats.percentileofscore(self.train_stored.values, x['smaller']) / 100
                bigger_p = stats.percentileofscore(self.train_stored.values, x['bigger']) / 100
                r = smaller_p - bigger_p
                return r
            result_smaller = result.rolling(window=self.smaller_window).apply(func=self.func)
            result_bigger = result.rolling(window=self.bigger_window).apply(func=self.func)
            result = pandas.DataFrame(data={'smaller': result_smaller, 'bigger': result_bigger}).apply(func=dister, axis=1)
        elif self.operation == 'quantile':
            def quantilizer(x):
                r = stats.percentileofscore(x, self.func(x[:self.smaller_window])) / 100
                return r
            result = result.rolling(window=self.bigger_window).apply(func=quantilizer)
        else:
            raise Exception("Invalid operation specified")
        return result
    @property
    def lag(self):
        return self.bigger_window


class WindowAppGenerator:
    def __init__(self, func, window):
        self.func = func
        self.window = window
        self.value_type = ValueTypes.CONTINUOUS
    @property
    def parametrization(self):
        dictated = tuple([my_hex('WindowAppGenerator'),
                          my_func_hex(self.func),
                          my_hex(self.window),
                          my_hex(self.value_type)])
        return dictated
    @property
    def parametrization_hash(self):
        hashed = my_hex(self.parametrization)
        return hashed
    def project_first(self, series_dict):
        assert len(list(series_dict.keys())) == 1
        s = series_dict[list(series_dict.keys())[0]].copy()
        result = s.rolling(window=self.window).apply(func=self.func)
        return result
    def project_second(self, series_dict):
        assert len(list(series_dict.keys())) == 1
        s = series_dict[list(series_dict.keys())[0]].copy()
        result = s.rolling(window=self.window).apply(func=self.func)
        return result
    @property
    def lag(self):
        return self.window


class PctChanger:
    def __init__(self):
        self.window = 1
        self.value_type = ValueTypes.CONTINUOUS
    @property
    def parametrization(self):
        dictated = tuple([my_hex('PctChanger'),
                          my_hex(self.window),
                          my_hex(self.value_type)])
        return dictated
    @property
    def parametrization_hash(self):
        hashed = my_hex(self.parametrization)
        return hashed
    def project_first(self, series_dict):
        assert len(list(series_dict.keys())) == 1
        s = series_dict[list(series_dict.keys())[0]].copy()
        result = s.pct_change()
        mask_positive_inf = result == numpy.inf
        mask_negative_inf = result == -numpy.inf
        result[mask_positive_inf] = result[~mask_positive_inf].max()
        result[mask_negative_inf] = result[~mask_negative_inf].min()
        return result
    def project_second(self, series_dict):
        assert len(list(series_dict.keys())) == 1
        s = series_dict[list(series_dict.keys())[0]].copy()
        result = s.pct_change()
        mask_positive_inf = result == numpy.inf
        mask_negative_inf = result == -numpy.inf
        result[mask_positive_inf] = result[~mask_positive_inf].max()
        result[mask_negative_inf] = result[~mask_negative_inf].min()
        return result
    @property
    def lag(self):
        return self.window


class SimpleAggregator:
    def __init__(self, func):
        self.func = func
        self.window = 0
        self.value_type = ValueTypes.CONTINUOUS
    @property
    def parametrization(self):
        dictated = tuple([my_hex('SimpleAggregator'),
                          my_func_hex(self.func),
                          my_hex(self.window),
                          my_hex(self.value_type)])
        return dictated
    @property
    def parametrization_hash(self):
        hashed = my_hex(self.parametrization)
        return hashed
    def project_first(self, series_dict):
        assert len(list(series_dict.keys())) == 1
        s = series_dict[list(series_dict.keys())[0]].copy()
        result = s.reset_index().groupby(by='index').agg(self.func)
        result = result[result.columns[0]]
        return result
    def project_second(self, series_dict):
        assert len(list(series_dict.keys())) == 1
        s = series_dict[list(series_dict.keys())[0]].copy()
        result = s.reset_index().groupby(by='index').agg(self.func)
        result = result[result.columns[0]]
        return result
    @property
    def lag(self):
        return self.window
    def decide_on_missing(self, n_missing):
        if n_missing > 0:
            return 1
        else:
            return 0


class SimpleCasterAggMonth:
    def __init__(self):
        self.ts_frequency = 'MS'
        self.ts_delta = pandas.to_datetime("2010-02-01").to_period("M") - pandas.to_datetime("2010-01-01").to_period("M")
    @property
    def parametrization(self):
        dictated = tuple([my_hex('SimpleCasterAggMonth'),
                          my_hex(self.ts_frequency)])
        return dictated
    @property
    def parametrization_hash(self):
        hashed = my_hex(self.parametrization)
        return hashed
    def cast(self, x):
        casted = datetime.datetime(year=pandas.to_datetime(x).year, month=pandas.to_datetime(x).month, day=1, tzinfo=datetime.timezone.utc).isoformat()
        return casted


class UGMSklearnClass:
    def __init__(self, model, model_kwargs, window, forward=1, log=True, pca=False):
        self._model = model
        self.model_kwargs = model_kwargs
        self.model = None
        self.window = window
        self.forward = forward
        self.log = log
        self.pca = pca
        self.value_type = ValueTypes.CONTINUOUS
        self.impute_max = numpy.nan
        self.impute_min = numpy.nan
    @property
    def parametrization(self):
        dictated = tuple([my_hex('UGMSklearnClass'),
                          my_func_hex(self._model),
                          my_dict_hex(self.model_kwargs),
                          my_hex(self.window),
                          my_hex(self.forward),
                          my_hex(self.log),
                          my_hex(self.pca),
                          my_hex(self.value_type)])
        return dictated
    @property
    def parametrization_hash(self):
        hashed = my_hex(self.parametrization)
        return hashed
    def project_first(self, series_dict):
        assert len(list(series_dict.keys())) == 1
        result = series_dict[list(series_dict.keys())[0]].copy()
        result.name = 'BASE'
        name = result.name
        if self.log:
            result = result.pct_change()
            result = (result + 1).apply(func=numpy.log)
        mask_positive_inf = result == numpy.inf
        mask_negative_inf = result == -numpy.inf
        self.impute_max = result[~mask_positive_inf].max()
        self.impute_min = result[~mask_negative_inf].min()
        result[mask_positive_inf] = self.impute_max
        result[mask_negative_inf] = self.impute_min
        result = pandas.DataFrame(result)
        for j in range(self.window):
            result['{0}__LAG{1}'.format(name, j)] = result[name].shift(j)
        result['{0}__Y'.format(name)] = result[name].shift(-self.forward)
        missing_mask_xy = pandas.isna(result).any(axis=1)
        missing_mask_x = pandas.isna(result[[c for c in result.columns if '__LAG' in c]]).any(axis=1)
        result_nona = result[~missing_mask_xy].copy()
        x = result_nona[[c for c in result.columns if '__LAG' in c]].values
        y = result_nona['{0}__Y'.format(name)].values
        self.model = self._model(**self.model_kwargs)
        self.model.fit(X=x, y=y)
        xx = result[~missing_mask_x][[c for c in result.columns if '__LAG' in c]].values
        y_hat = self.model.predict(X=xx)
        result = result[name].copy()
        result[missing_mask_x] = numpy.nan
        result[~missing_mask_x] = y_hat.astype(dtype=result.dtype)
        return result
    def project_second(self, series_dict):
        assert len(list(series_dict.keys())) == 1
        result = series_dict[list(series_dict.keys())[0]].copy()
        result.name = 'BASE'
        name = result.name
        if self.log:
            result = result.pct_change()
            result = (result + 1).apply(func=numpy.log)
        mask_positive_inf = result == numpy.inf
        mask_negative_inf = result == -numpy.inf
        result[mask_positive_inf] = self.impute_max
        result[mask_negative_inf] = self.impute_min
        result = pandas.DataFrame(result)
        for j in range(self.window):
            result['{0}__LAG{1}'.format(name, j)] = result[name].shift(j)
        result['{0}__Y'.format(name)] = result[name].shift(-self.forward)
        missing_mask_xy = pandas.isna(result).any(axis=1)
        missing_mask_x = pandas.isna(result[[c for c in result.columns if '__LAG' in c]]).any(axis=1)
        result_nona = result[~missing_mask_xy].copy()
        x = result_nona[[c for c in result.columns if '__LAG' in c]].values
        y = result_nona['{0}__Y'.format(name)].values
        xx = result[~missing_mask_x][[c for c in result.columns if '__LAG' in c]].values
        y_hat = self.model.predict(X=xx)
        result = result[name].copy()
        result[missing_mask_x] = numpy.nan
        result[~missing_mask_x] = y_hat.astype(dtype=result.dtype)
        return result
    @property
    def lag(self):
        return self.window


class UGMARIMAClass:
    def __init__(self, model, model_kwargs, window, forward=1, log=True, pca=False):
        self._model = model
        self.model_kwargs = model_kwargs
        self.model = None
        self.window = window
        self.forward = forward
        self.log = log
        self.pca = pca
        self.value_type = ValueTypes.CONTINUOUS
        self.impute_max = numpy.nan
        self.impute_min = numpy.nan
        self.y_project_first_series = None
    @property
    def parametrization(self):
        dictated = tuple([my_hex('UGMARIMAClass'),
                          my_func_hex(self._model),
                          my_dict_hex(self.model_kwargs),
                          my_hex(self.window),
                          my_hex(self.forward),
                          my_hex(self.log),
                          my_hex(self.pca),
                          my_hex(self.value_type)])
        return dictated
    @property
    def parametrization_hash(self):
        hashed = my_hex(self.parametrization)
        return hashed
    def project_first(self, series_dict):
        run_time = time.time()
        assert len(list(series_dict.keys())) == 1
        result = series_dict[list(series_dict.keys())[0]].copy()
        name = result.name
        self.y_project_first_series = result.copy()
        if self.log:
            result = result.pct_change()
            result = (result + 1).apply(func=numpy.log)
        mask_positive_inf = result == numpy.inf
        mask_negative_inf = result == -numpy.inf
        self.impute_max = result[~mask_positive_inf].max()
        self.impute_min = result[~mask_negative_inf].min()
        result[mask_positive_inf] = self.impute_max
        result[mask_negative_inf] = self.impute_min
        y = result.iloc[1:].copy()
        self.model = self._model(window=self.window, **self.model_kwargs)
        run_time = time.time() - run_time
        # print('prep', run_time)
        run_time = time.time()
        self.model.fit(y=y)
        run_time = time.time() - run_time
        # print('fit', run_time)
        run_time = time.time()
        y_hat = self.model.predict(y=y)[1:]
        y_forecast = self.model.forecast(y=y)
        y_hat = numpy.concatenate((y_hat, numpy.array([y_forecast])))
        result = result.copy()
        result.iloc[1:] = y_hat.astype(dtype=result.dtype)
        run_time = time.time() - run_time
        # print('cast', run_time)
        return result
    def project_second(self, series_dict):
        assert len(list(series_dict.keys())) == 1
        result = series_dict[list(series_dict.keys())[0]].copy()
        name = result.name
        result_ = pandas.concat((self.y_project_first_series, result), ignore_index=False)
        if self.log:
            result_ = result_.pct_change()
            result_ = (result_ + 1).apply(func=numpy.log)
        mask_positive_inf = result_ == numpy.inf
        mask_negative_inf = result_ == -numpy.inf
        result_[mask_positive_inf] = self.impute_max
        result_[mask_negative_inf] = self.impute_min
        appendix = result_.iloc[self.y_project_first_series.shape[0]:]
        model_copy = self.model.copy()
        forecasted = []
        for j in range(appendix.shape[0]):
            model_copy.update(appendix.iloc[[j]])
            y_hat_i = model_copy.forecast(None)
            forecasted.append(y_hat_i)
        result = result.copy()
        result = pandas.Series(data=forecasted, index=result.index)
        return result
    @property
    def lag(self):
        return self.window


def refactor_trend_code(trend):
    if trend in ['n', 'c', 't', 'ct']:
        return trend
    else:
        if trend[0] != 't':
            raise ValueError()
        else:
            order = int(trend[1])
            x = [0 for j in range(order)] + [1]
            return x


class AutoArima:
    """
    AutoArima implementation from pmdarima

    No seasonal component is considered

    """
    def __init__(self, window, max_d=2):
        self.max_window = window
        self.max_d = max_d
        self._arima = ARIMA
        self.arima = None
        self.fitted_p = None
        self.fitted_d = None
        self.fitted_q = None
        self.fitted_trend = None
    def fit(self, y):

        model = pmdarima.auto_arima(y.values, start_p=0, start_q=0,
                                    max_order=self.max_window, seasonal=False,
                                    stepwise=True, suppress_warnings=True,
                                    error_action='warn', method='nm')

        # self.fitted_p = current_p
        # self.fitted_d = d
        # self.fitted_q = current_q
        # self.fitted_trend = current_trend
        # self.arima = self._arima(endog=y.values, order=(current_p, d, current_q), trend=refactor_trend_code(current_trend),
        #                          seasonal_order=(0, 0, 0, 0)).fit()

        self.arima = copy.deepcopy(model)

    def predict(self, y):

        prediction = self.arima.predict_in_sample(dynamic=False)
        return prediction

    def update(self, y):

        self.arima.update(y)

    def forecast(self, y):

        forecasted = self.arima.predict(n_periods=1)[-1]
        return forecasted

    def copy(self):

        return copy.deepcopy(self)


class DefiniteArima:
    def __init__(self, window, p, q, max_d=2):
        assert window >= max(p, q, max_d)
        self.p = p
        self.q = q
        self.max_d = max_d
        self.d = None
        self.trend = None
        self._arima = pmdarima.ARIMA
        self.arima = None
    def fit(self, y):

        # identify d

        # codes: t_d0, c_d0, c_d1, n(c)_d2
        kpss_result = pandas.DataFrame(index=['t_d0', 'c_d0', 'c_d1', 'n_d2'],
                                       columns=['kpss_values', 'kpss_trend', 'd'],
                                       data=[[numpy.nan, 'ct', 0],
                                             [numpy.nan, 'c', 0],
                                             [numpy.nan, 'c', 1],
                                             [numpy.nan, 'c', 2]])
        for ix in kpss_result.index:
            yy = y.copy()
            kpss_d = kpss_result.loc[ix, 'd']
            for j in range(kpss_d):
                yy = yy.diff()
            yy = yy.dropna().values
            kpss_trend = kpss_result.loc[ix, 'kpss_trend']
            try:
                kpss_output = kpss(x=yy, regression=kpss_trend, nlags='auto')
            except OverflowError:
                try:
                    kpss_output = kpss(x=yy, regression=kpss_trend, nlags='legacy')
                except Exception as e:
                    raise e
            except Exception as e:
                raise e
            # kpss_result.loc[ix, 'kpss_values'] = kpss_output[1]
            # due to undesired properties of the implementation of kpss test in the part how p-values are projected,
            # a decision was made to consider test statistic values themselves
            kpss_result.loc[ix, 'kpss_values_stat'] = kpss_output[0]
            kpss_result.loc[ix, 'kpss_values_p1'] = kpss_output[3]['1%']
            kpss_result.loc[ix, 'kpss_values_p10'] = kpss_output[3]['10%']
        lesser_mask = kpss_result['kpss_values_stat'] <= kpss_result['kpss_values_p10']
        if lesser_mask.sum() > 0:
            conditional_min = kpss_result.loc[lesser_mask, 'kpss_values_stat'].min()
            i = kpss_result['kpss_values_stat'].values.tolist().index(conditional_min)
        else:
            i = kpss_result['kpss_values_stat'].argmin()
        d, alternative_trend = kpss_result['d'].values[i], kpss_result['kpss_trend'].values[i]

        self.d = d
        if d == 0:
            self.trend = alternative_trend
        elif d == 1:
            self.trend = 'n' if alternative_trend == 'n' else 't'
        else:
            self.trend = 'n' if alternative_trend == 'n' else 't2'

        self.arima = self._arima(order=(self.p, self.d, self.q), trend=refactor_trend_code(self.trend),
                                 seasonal_order=(0, 0, 0, 0), method='nm', suppress_warnings=True)
        self.arima.fit(y.values)

    def predict(self, y):

        prediction = self.arima.predict_in_sample(dynamic=False)
        return prediction

    def update(self, y):

        self.arima.update(y)

    def forecast(self, y):

        forecasted = self.arima.predict(n_periods=1)[-1]
        return forecasted

    def copy(self):

        return copy.deepcopy(self)
