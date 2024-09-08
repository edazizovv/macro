#
import json
import datetime

#
import numpy
import pandas
from scipy import stats
from sklearn.preprocessing import KBinsDiscretizer
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import kpss


#
from new_base import UnholyVice
from new_constants import ValueTypes


#
class VincentClassMobsterS:

    def __init__(self, x_factors_in, target, target_source, target_transform, name_list, param_list, projector, performer=None, stabilizer=None):

        self.mobsters = {x_factors_in[j]: VincentClassMobster(name=x_factors_in[j], target=target, target_source=target_source, target_transform=target_transform, name_list=name_list, param_list=param_list, projector=projector, performer=performer, stabilizer=stabilizer) for j in range(len(x_factors_in))}

    def pull(self, fg, sources, timeaxis):

        for key in self.mobsters.keys():

            print('{0} / {1}'.format(list(self.mobsters.keys()).index(key), len(list(self.mobsters.keys()))))

            fg_local = fg.copy()

            local_path_vertices, local_path_matrix, local_path_pseudo_edges = self.mobsters[key].supply(fg=fg)
            fg_local.init_path(local_path_vertices, local_path_matrix, local_path_pseudo_edges)

            for fold_n in fg_local.folds:
                local_sources = [x for x in sources if (x.name == self.mobsters[key].target_source) or (x.name == self.mobsters[key].name)]
                data_train, data_test = fg_local.fold(local_sources, self.mobsters[key].features + [self.mobsters[key].name] + [self.mobsters[key].target], timeaxis, fold_n=fold_n)
                x_train, y_train = data_train[[x for x in data_train.columns if x != self.mobsters[key].target]].iloc[:-1, :], data_train[self.mobsters[key].target].iloc[1:]
                x_test, y_test = data_test[[x for x in data_test.columns if x != self.mobsters[key].target]].iloc[:-1, :], data_test[self.mobsters[key].target].iloc[1:]

                self.mobsters[key].pull(fold_n=fold_n, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)

    def collapse(self):

        collapsed = []
        collapsed_stats = {}

        for key in self.mobsters.keys():

            print('{0} / {1}'.format(list(self.mobsters.keys()).index(key), len(list(self.mobsters.keys()))))
            c, s = self.mobsters[key].collapse()
            collapsed += c
            collapsed_stats[key] = s

        collapsed = numpy.unique(collapsed).tolist()

        return collapsed, collapsed_stats


class VincentClassMobster:

    def __init__(self, name, target, target_source, target_transform, name_list, param_list, projector, performer=None, stabilizer=None):

        self.name = name
        self.target = target
        self.target_source = target_source
        self.target_transform = target_transform
        self.name_list = name_list
        self.param_list = param_list
        self._projector = projector
        self.performer = performer
        self.stabilizer = stabilizer

        self.features = ['{0}__{1}'.format(self.name, self.name_list[j]) for j in range(len(self.name_list))]
        self.local_resulted = None
        self.global_resulted = None
        self.global_resulted_agg = None

    def supply(self, fg):

        """
        local_path_vertices = [self.target_source] + [self.target] + [self.name] + self.features
        local_path_matrix = numpy.zeros(shape=((len(self.name_list) + 3), (len(self.name_list) + 3)))
        local_path_matrix[0, 1] = 1
        local_path_matrix[2, 3:] = 1
        local_path_pseudo_edges = [None, self.target_transform, None] + [self._projector(**self.param_list[j]) for j in range(len(self.param_list))]

        return local_path_vertices, local_path_matrix, local_path_pseudo_edges
        """

        target_components = [x for x in fg.path.path_vertices if self.target_source in x]
        target_components_mask = numpy.isin(fg.path.path_vertices, target_components)
        target_components_ix = numpy.arange(fg.path.path_matrix.shape[0])[target_components_mask]
        path_matrix_sub_target = fg.path.path_matrix[target_components_ix[:, numpy.newaxis], target_components_ix]
        path_vertices_sub_target = numpy.array(fg.path.path_vertices)[target_components_ix].tolist()
        path_pseudo_edges_sub_target = fg.path.path_pseudo_edges[target_components_ix].tolist()
        n_targets = target_components_mask.sum()

        local_path_vertices = path_vertices_sub_target + [self.name] + self.features
        local_path_matrix = numpy.zeros(shape=((len(self.name_list) + 1 + n_targets), (len(self.name_list) + 1 + n_targets)))

        targets_mask = numpy.arange(n_targets)
        local_path_matrix[targets_mask[:, numpy.newaxis], targets_mask] = path_matrix_sub_target

        local_path_matrix[n_targets, 3:] = 1

        local_path_pseudo_edges = path_pseudo_edges_sub_target + [None] + [self._projector(**self.param_list[j]) for j in range(len(self.param_list))]

        return local_path_vertices, local_path_matrix, local_path_pseudo_edges


    def pull(self, fold_n, x_train, y_train, x_test, y_test):

        self.local_resulted = []
        for j in range(len(self.name_list)):

            if self.performer is not None:
                performed = self.performer(x=x_test.iloc[:, j].values, y=y_test.values)
            else:
                performed = numpy.nan
            if self.stabilizer is not None:
                stabilized = self.stabilizer(x=x_test.iloc[:, j].values, y=y_test.values)
            else:
                stabilized = numpy.nan
            self.local_resulted.append([fold_n, self.name_list[j], performed, stabilized])
        self.local_resulted = pandas.DataFrame(data=self.local_resulted, columns=['fold_n', 'transform', 'performed', 'stabilized'])

        base_performed = self.performer(x=x_test.iloc[:, len(self.name_list)].values, y=y_test.values)
        self.local_resulted['base_performed'] = numpy.abs(base_performed)

        # xxl = pandas.DataFrame(data={'x': x_test.iloc[:, len(self.name_list)].values, 'y': y_test.values})

        if self.global_resulted is None:
            self.global_resulted = self.local_resulted.copy()
        else:
            self.global_resulted = pandas.concat((self.global_resulted, self.local_resulted), axis=0, ignore_index=False)

    def collapse(self):

        # adjust for potential change of the signs
        n_performed = (self.global_resulted['performed'] >= 0).sum()
        if n_performed < (self.global_resulted['performed'].shape[0] / 2):
            self.global_resulted['performed'] = self.global_resulted['performed'] * (-1)
        n_base = (self.global_resulted['base_performed'] >= 0).sum()
        if n_base < (self.global_resulted['base_performed'].shape[0] / 2):
            self.global_resulted['base_performed'] = self.global_resulted['base_performed'] * (-1)

        # https://www.statisticssolutions.com/free-resources/directory-of-statistical-analyses/paired-sample-t-test/
        self.global_resulted['perf_diff'] = self.global_resulted['performed'] - self.global_resulted['base_performed']
        global_resulted_agg_part = self.global_resulted.groupby(by='transform')
        global_resulted_agg_mean = global_resulted_agg_part[['perf_diff']].mean().rename(columns={'perf_diff': 'mean'})
        global_resulted_agg_std = global_resulted_agg_part[['perf_diff']].std().rename(columns={'perf_diff': 'std'})
        global_resulted_agg_count = global_resulted_agg_part[['perf_diff']].count().rename(columns={'perf_diff': 'n'})
        global_resulted_agg = global_resulted_agg_mean.merge(right=global_resulted_agg_std, left_index=True, right_index=True, how='outer')
        global_resulted_agg = global_resulted_agg.merge(right=global_resulted_agg_count, left_index=True, right_index=True, how='outer')

        def tester(x):
            arg = x['mean'] / (x['std'] / (x['n'] ** 0.5))
            pv = 1 - stats.t.cdf(x=arg, df=x['n'] - 1)
            return pv

        global_resulted_agg['test_result'] = global_resulted_agg.apply(func=tester, axis=1)
        global_resulted_agg = global_resulted_agg.sort_values(by='test_result')
        self.global_resulted_agg = global_resulted_agg.copy()

        cmp_alpha_thresh = 0.05

        global_resulted_filtered = global_resulted_agg[global_resulted_agg['test_result'] <= cmp_alpha_thresh].copy()
        if global_resulted_filtered.shape[0] > 0:
            the_chosen_one = ['{0}__{1}'.format(self.name, global_resulted_filtered.index.values[0])]
        else:
            the_chosen_one = []

        return the_chosen_one, (self.global_resulted_agg, self.global_resulted)


class Stayer:
    def __init__(self, method):
        self.method = method
        self.value_type = ValueTypes.CONTINUOUS
        self.impute_max = numpy.nan
        self.impute_min = numpy.nan
        self.wards = None
    @property
    def parametrization(self):
        dictated = tuple([hash('STAYER'),
                          hash(self.value_type)])
        return dictated
    @property
    def parametrization_hash(self):
        hashed = hash(self.parametrization)
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
        dictated = tuple([hash('Binner'),
                          hash(str(self.n_bins)),
                          hash(self.method),
                          hash(self.value_type),
                          hash(self._binner),
                          hash(self._binner_kwargs.values())])
        return dictated
    @property
    def parametrization_hash(self):
        hashed = hash(self.parametrization)
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
        dictated = tuple([hash('RangedPct'),
                          hash(str(self.shift)),
                          hash(self.value_type),
                          hash(str(self.log))])
        return dictated
    @property
    def parametrization_hash(self):
        hashed = hash(self.parametrization)
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
        dictated = tuple([hash('WindowRollImpulse'),
                          hash(self.func),
                          hash(str(self.smaller_window)),
                          hash(str(self.bigger_window)),
                          hash(self.operation),
                          hash(str(self.pct)),
                          hash(str(self.shift)),
                          hash(self.value_type),
                          hash(str(self.log))])
        return dictated
    @property
    def parametrization_hash(self):
        hashed = hash(self.parametrization)
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
        dictated = tuple([hash('SimpleAggregator'),
                          hash(self.func),
                          hash(str(self.window)),
                          hash(self.value_type)])
        return dictated
    @property
    def parametrization_hash(self):
        hashed = hash(self.parametrization)
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


class SimpleCasterAggMonth:
    def __init__(self):
        self.ts_frequency = 'MS'
    @property
    def parametrization(self):
        dictated = tuple([hash('SimpleCasterAggMonth'),
                          hash(self.ts_frequency)])
        return dictated
    @property
    def parametrization_hash(self):
        hashed = hash(self.parametrization)
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
        dictated = tuple([hash('UGMSklearnClass'),
                          hash(self._model),
                          hash(self.model_kwargs.values()),
                          hash(str(self.window)),
                          hash(str(self.forward)),
                          hash(str(self.log)),
                          hash(str(self.pca)),
                          hash(self.value_type)])
        return dictated
    @property
    def parametrization_hash(self):
        hashed = hash(self.parametrization)
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
        dictated = tuple([hash('UGMARIMAClass'),
                          hash(self._model),
                          hash(self.model_kwargs.values()),
                          hash(str(self.window)),
                          hash(str(self.forward)),
                          hash(str(self.log)),
                          hash(str(self.pca)),
                          hash(self.value_type)])
        return dictated
    @property
    def parametrization_hash(self):
        hashed = hash(self.parametrization)
        return hashed
    def project_first(self, series_dict):
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
        self.model.fit(y=y)
        y_hat = self.model.predict(y=y)[1:]
        y_forecast = self.model.forecast(y=y)
        y_hat = numpy.concatenate((y_hat, numpy.array([y_forecast])))
        result = result.copy()
        result.iloc[1:] = y_hat.astype(dtype=result.dtype)
        return result
    def project_second(self, series_dict):
        assert len(list(series_dict.keys())) == 1
        result = series_dict[list(series_dict.keys())[0]].copy()
        name = result.name
        forecasted = []
        for i in range(result.shape[0]):
            result_ = pandas.concat((self.y_project_first_series.iloc[i+1:],
                                     result.iloc[:i+1]), ignore_index=False)
            if self.log:
                result_ = result_.pct_change()
                result_ = (result_ + 1).apply(func=numpy.log)
            mask_positive_inf = result_ == numpy.inf
            mask_negative_inf = result_ == -numpy.inf
            result_[mask_positive_inf] = self.impute_max
            result_[mask_negative_inf] = self.impute_min
            y = result_.iloc[1:].copy()
            y_hat_i = self.model.forecast(y=y)
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
    AutoArima implementation following the algorithm outlined in https://otexts.com/fpp2/arima-r.html
    (with some minor adjustments)

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
            kpss_result.loc[ix, 'kpss_values'] = kpss_output[1]
        i = kpss_result['kpss_values'].argmax()
        d, alternative_trend = kpss_result['d'].values[i], kpss_result['kpss_trend'].values[i]

        # first examination

        # codes: 0_d_0, 2_d_2, 1_d_0, 0_d_1, [0_d_0 -n]

        if alternative_trend == 'ct':
            arima_results = pandas.DataFrame(columns=['aic', 'p', 'q', 'trend'],
                                             data=[[numpy.nan, 0, 0, 'ct'],
                                                   [numpy.nan, 2, 2, 'ct'],
                                                   [numpy.nan, 1, 0, 'ct'],
                                                   [numpy.nan, 0, 1, 'ct'],
                                                   [numpy.nan, 0, 0, 'c'],
                                                   [numpy.nan, 2, 2, 'c'],
                                                   [numpy.nan, 1, 0, 'c'],
                                                   [numpy.nan, 0, 1, 'c'],
                                                   [numpy.nan, 0, 0, 'n']])
        elif d == 0:
            arima_results = pandas.DataFrame(columns=['aic', 'p', 'q', 'trend'],
                                             data=[[numpy.nan, 0, 0, 'c'],
                                                   [numpy.nan, 2, 2, 'c'],
                                                   [numpy.nan, 1, 0, 'c'],
                                                   [numpy.nan, 0, 1, 'c'],
                                                   [numpy.nan, 0, 0, 'n']])
        elif d == 1:
            arima_results = pandas.DataFrame(columns=['aic', 'p', 'q', 'trend'],
                                             data=[[numpy.nan, 0, 0, 't'],
                                                   [numpy.nan, 2, 2, 't'],
                                                   [numpy.nan, 1, 0, 't'],
                                                   [numpy.nan, 0, 1, 't'],
                                                   [numpy.nan, 0, 0, 'n']])
        else:
            arima_results = pandas.DataFrame(columns=['aic', 'p', 'q', 'trend'],
                                             data=[[numpy.nan, 0, 0, 't2'],
                                                   [numpy.nan, 2, 2, 't2'],
                                                   [numpy.nan, 1, 0, 't2'],
                                                   [numpy.nan, 0, 1, 't2']])

        for i in range(arima_results.shape[0]):
            p = arima_results['p'].values[i]
            q = arima_results['q'].values[i]
            trend = arima_results['trend'].values[i]
            arima = self._arima(endog=y.values, order=(p, d, q), trend=refactor_trend_code(trend), seasonal_order=(0, 0, 0, 0))
            arima_res = arima.fit()
            arima_results.loc[arima_results.index[i], 'aic'] = arima_res.aic
        i = arima_results['aic'].argmin()
        current_aic = arima_results['aic'].values[i]
        current_p = arima_results['p'].values[i]
        current_q = arima_results['q'].values[i]
        current_trend = arima_results['trend'].values[i]

        # loops

        finish = False
        while not finish:

            max_pq = self.max_window - max(d, current_p, current_q)

            # codes: (p+1, q), (p-1, q), (p, q+1), (p, q-1), all those without c
            data = []
            if current_p > 1:
                data_append = [[numpy.nan, current_p - 1, current_q, current_trend]]
                data += data_append
            if current_p < max_pq:
                data_append = [[numpy.nan, current_p + 1, current_q, current_trend]]
                data += data_append
            if current_q > 1:
                data_append = [[numpy.nan, current_p, current_q - 1, current_trend]]
                data += data_append
            if current_q < max_pq:
                data_append = [[numpy.nan, current_p, current_q + 1, current_trend]]
                data += data_append
            if current_trend != 'n':
                data_append_n = []
                for z in data:
                    zz = list(z)
                    zz[-1] = 'n'
                    data_append_n += [zz]
                data_append_c = []
                if current_trend == 'ct':
                    for z in data:
                        zz = list(z)
                        zz[-1] = 'c'
                        data_append_c += [zz]
                data += data_append_n
                data += data_append_c
            else:
                data_append_c = []
                for z in data:
                    zz = list(z)
                    zz[-1] = 'c' if d == 0 else 't' if d == 1 else 't2'
                    data_append_c += [zz]
                data += data_append_c
            arima_results = pandas.DataFrame(columns=['aic', 'p', 'q', 'trend'],
                                             data=data)

            for i in range(arima_results.shape[0]):
                p = arima_results['p'].values[i]
                q = arima_results['q'].values[i]
                trend = arima_results['trend'].values[i]
                arima = self._arima(endog=y.values, order=(p, d, q), trend=refactor_trend_code(trend), seasonal_order=(0, 0, 0, 0))
                arima_res = arima.fit()
                arima_results.loc[arima_results.index[i], 'aic'] = arima_res.aic
            i = arima_results['aic'].argmin()
            candidate_aic = arima_results['aic'].values[i]

            if candidate_aic <= current_aic:
                current_aic = candidate_aic
                current_p = arima_results['p'].values[i]
                current_q = arima_results['q'].values[i]
                current_trend = arima_results['trend'].values[i]
            else:
                finish = True

        self.fitted_p = current_p
        self.fitted_d = d
        self.fitted_q = current_q
        self.fitted_trend = current_trend
        self.arima = self._arima(endog=y.values, order=(current_p, d, current_q), trend=refactor_trend_code(current_trend),
                                 seasonal_order=(0, 0, 0, 0)).fit()

    def predict(self, y):

        self.arima = self._arima(endog=y.values, order=(self.fitted_p, self.fitted_d, self.fitted_q), trend=refactor_trend_code(self.fitted_trend),
                                 seasonal_order=(0, 0, 0, 0)).fit()

        prediction = self.arima.predict()
        return prediction

    def forecast(self, y):

        self.arima = self._arima(endog=y.values, order=(self.fitted_p, self.fitted_d, self.fitted_q), trend=refactor_trend_code(self.fitted_trend),
                                 seasonal_order=(0, 0, 0, 0)).fit()

        forecasted = self.arima.forecast()[-1]
        return forecasted


class DefiniteArima:
    def __init__(self, window, p, q, max_d=2):
        assert window >= max(p, q, max_d)
        self.p = p
        self.q = q
        self.max_d = max_d
        self.d = None
        self.trend = None
        self._arima = ARIMA
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
            kpss_result.loc[ix, 'kpss_values'] = kpss_output[1]
        i = kpss_result['kpss_values'].argmax()
        d, alternative_trend = kpss_result['d'].values[i], kpss_result['kpss_trend'].values[i]

        self.d = d
        if d == 0:
            self.trend = alternative_trend
        elif d == 1:
            self.trend = 'n' if alternative_trend == 'n' else 't'
        else:
            self.trend = 'n' if alternative_trend == 'n' else 't2'

        self.arima = self._arima(endog=y.values, order=(self.p, self.d, self.q), trend=refactor_trend_code(self.trend),
                                 seasonal_order=(0, 0, 0, 0)).fit()

    def predict(self, y):

        self.arima = self._arima(endog=y.values, order=(self.p, self.d, self.q), trend=refactor_trend_code(self.trend),
                                 seasonal_order=(0, 0, 0, 0)).fit()

        prediction = self.arima.predict()
        return prediction

    def forecast(self, y):

        self.arima = self._arima(endog=y.values, order=(self.p, self.d, self.q), trend=refactor_trend_code(self.trend),
                                 seasonal_order=(0, 0, 0, 0)).fit()

        forecasted = self.arima.forecast()[-1]
        return forecasted
