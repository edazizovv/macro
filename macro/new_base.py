#
import hashlib
import _pickle
import pickle
import time

#
import numpy
import pandas


#
from macro.new_base_utils import new_read, my_hex    # TODO: to be replaced with the correct new read
from macro._new_constants import DataReadingConstants, SystemFilesSignatures, Routing   # TODO: to be refactored
from macro.new_constants import (
    PROJECT_STRUCTURE as PJST,
    LOADING_DOCK as LD,
)
from macro.new_utils.readers import READERS

#
class Phaser:

    def __init__(self, fg, sources, target, timeaxis, master_mobster, roller_mobsters):

        self.fg = fg
        self.sources = sources
        self.target = target
        self.timeaxis = timeaxis
        self.master_mobster = master_mobster
        self.roller_mobsters = roller_mobsters

    def pump(self):

        # here should be some adjustment to fg stellar path based on mobsters' requests
        # fg.do_something(mobsters)
        mobsters_features = ...

        for fold_n in self.fg.folds:
            data_train, data_test = self.fg.fold(self.sources, mobsters_features + [self.target], self.timeaxis, fold_n=fold_n)
            x_train, y_train = data_train[[x for x in data_train.columns if x != self.target]].iloc[:-1, :], data_train[self.target].iloc[1:]
            x_test, y_test = data_test[[x for x in data_test.columns if x != self.target]].iloc[:-1, :], data_test[self.target].iloc[1:]

            self.master_mobster.pull(fold_n=fold_n, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)

    def dump(self):

        out_features, out_features_stats = self.master_mobster.collapse()

        return out_features, out_features_stats



class TSCreator:

    @property
    def frequency(self):
        raise NotImplemented()

    def cast(self):
        raise NotImplemented()


class Vice:
    def __init__(self, name=None, values=None, index=None, value_type=None, ts_frequency=None, ts_delta=None, publication_lag=None, start_history=None):

        self.name = name
        self.values = values
        self.index = index
        self.value_type = value_type
        self.ts_frequency = ts_frequency
        self.ts_delta = ts_delta
        self.publication_lag = publication_lag
        self.start_history = start_history
        self.series = pandas.Series(data=self.values, index=self.index)

    def to_unholy_vice(self, lag_first_start_dt, first_start_dt, first_end_dt, lag_second_start_dt, second_start_dt, second_end_dt):

        av = UnholyVice(name=self.name, values=self.values, index=self.index, value_type=self.value_type,
                        ts_frequency=self.ts_frequency, ts_delta=self.ts_delta,
                        publication_lag=self.publication_lag, start_history=self.start_history,
                        lag_first_start_dt=lag_first_start_dt, first_start_dt=first_start_dt, first_end_dt=first_end_dt,
                        lag_second_start_dt=lag_second_start_dt, second_start_dt=second_start_dt, second_end_dt=second_end_dt)
        return av


class UnholyVice:
    def __init__(self, name=None, values=None, index=None, value_type=None,
                 ts_frequency=None, ts_delta=None,
                 publication_lag=None, start_history=None,
                 lag_first_start_dt=None, first_start_dt=None, first_end_dt=None, lag_second_start_dt=None, second_start_dt=None, second_end_dt=None):

        self.name = name
        self._values = values
        if index is not None:
            self._index = pandas.to_datetime(index).to_series().apply(func=lambda x: x.isoformat()).values
        else:
            self._index = index
        self.value_type = value_type
        self.ts_frequency = ts_frequency
        self.ts_delta = ts_delta
        self._series = pandas.Series(data=self._values, index=self._index)

        self.publication_lag = publication_lag
        self.start_history = start_history

        if lag_first_start_dt is not None:
            self.lag_first_start_dt = pandas.to_datetime(lag_first_start_dt).isoformat()
        else:
            self.lag_first_start_dt = None
        if first_start_dt is not None:
            self.first_start_dt = pandas.to_datetime(first_start_dt).isoformat()
        else:
            self.first_start_dt = None
        if first_end_dt is not None:
            self.first_end_dt = pandas.to_datetime(first_end_dt).isoformat()
        else:
            self.first_end_dt = None
        if lag_second_start_dt is not None:
            self.lag_second_start_dt = pandas.to_datetime(lag_second_start_dt).isoformat()
        else:
            self.lag_second_start_dt = None
        if second_start_dt is not None:
            self.second_start_dt = pandas.to_datetime(second_start_dt).isoformat()
        else:
            self.second_start_dt = None
        if second_end_dt is not None:
            self.second_end_dt = pandas.to_datetime(second_end_dt).isoformat()
        else:
            self.second_end_dt = None

        self._signature = None

    @property
    def parametrization(self):
        dictated = tuple([my_hex(self.lag_first_start_dt),
                          my_hex(self.first_start_dt),
                          my_hex(self.first_end_dt),
                          my_hex(self.lag_second_start_dt),
                          my_hex(self.second_start_dt),
                          my_hex(self.second_end_dt),
                          my_hex(self.publication_lag),
                          my_hex(self.start_history),
                          my_hex(self.ts_frequency),
                          my_hex(self.ts_delta)])
        return dictated
    @property
    def parametrization_hash(self):
        hashed = my_hex(self.parametrization)
        return hashed
    @property
    def signature(self):
        if self._signature is None:
            raise Exception("Unholy Vice is not signed")
        else:
            return self._signature

    def sign(self, incoming_chain, projector_hash):
        signed = my_hex(incoming_chain + projector_hash)
        self._signature = signed

    def dump(self):
        with open('../data/data_folds_hash/{0}.pkl'.format(self.signature), 'wb') as f:
            pickle.dump(self, f)

    def pump(self):
        with open('../data/data_folds_hash/{0}.pkl'.format(self.signature), 'rb') as f:
            return pickle.load(f)

    def needs_pump(self):
        assert self.signature
        result = self._values is None
        return result

    @property
    def base_values(self):
        result = self._values.copy()
        return result

    @property
    def base_index(self):
        result = self._index.copy()
        return result

    @property
    def base_series(self):
        result = self._series.copy()
        return result

    @property
    def ix_lag_mask_first(self):
        result = (self._index >= self.lag_first_start_dt) * (self._index <= self.first_end_dt)
        return result

    @property
    def lag_values_first(self):
        result = self._values[self.ix_lag_mask_first].copy()
        return result

    @property
    def lag_index_first(self):
        result = self._index[self.ix_lag_mask_first].copy()
        return result

    @property
    def lag_series_first(self):
        result = self._series[self.ix_lag_mask_first].copy()
        return result

    @property
    def ix_current_mask_first(self):
        result = (self._index >= self.first_start_dt) * (self._index <= self.first_end_dt)
        return result

    @property
    def current_values_first(self):
        result = self._values[self.ix_current_mask_first].copy()
        return result

    @property
    def current_index_first(self):
        result = self._index[self.ix_current_mask_first].copy()
        return result

    @property
    def current_series_first(self):
        result = self._series[self.ix_current_mask_first].copy()
        return result

    @property
    def ix_lag_mask_second(self):
        result = (self._index >= self.lag_second_start_dt) * (self._index <= self.second_end_dt)
        return result

    @property
    def lag_values_second(self):
        result = self._values[self.ix_lag_mask_second].copy()
        return result

    @property
    def lag_index_second(self):
        result = self._index[self.ix_lag_mask_second].copy()
        return result

    @property
    def lag_series_second(self):
        result = self._series[self.ix_lag_mask_second].copy()
        return result

    @property
    def ix_current_mask_second(self):
        result = (self._index >= self.second_start_dt) * (self._index <= self.second_end_dt)
        return result

    @property
    def current_values_second(self):
        result = self._values[self.ix_current_mask_second].copy()
        return result

    @property
    def current_index_second(self):
        result = self._index[self.ix_current_mask_second].copy()
        return result

    @property
    def current_series_second(self):
        result = self._series[self.ix_current_mask_second].copy()
        return result

    def to_vice(self):

        v = Vice(name=self.name,
                 values=self._values, index=self._index,
                 value_type=self.value_type, ts_frequency=self.ts_frequency)
        return v


class Projector:
    def __init__(self, ts_creator, role,
                 agg_function=None, agg_function_kwg=None,
                 fill_function=None, fill_function_kwg=None,
                 app_function=None, app_function_kwg=None):
        self.ts_creator = ts_creator
        self.role = role
        self._agg_function = agg_function
        self._fill_function = fill_function
        self._app_function = app_function
        self.agg_function_kwg = agg_function_kwg
        self.fill_function_kwg = fill_function_kwg
        self.app_function_kwg = app_function_kwg
        self.agg_function = None
        self.fill_function = None
        self.app_function = None
    """
        new_lag_start = v.ix_lag_mask_first.shift(self.window, freq=v.ts_frequency)[0]
        new_lag_mid = v.ix_lag_mask_second.shift(self.window, freq=v.ts_frequency)[0]
        result_vice = UnholyVice(values=result.values, index=result.index,
                                 value_type=v.value_type, ts_frequency=v.ts_frequency,
                                 lag_start_dt=new_lag_start, start_dt=v.start_dt,
                                 lag_mid_dt=new_lag_mid, mid_dt=v.mid_dt,
                                 end_dt=v.end_dt)
                                 """

    @property
    def parametrization(self):
        agg_function = self._agg_function(**self.agg_function_kwg) if self._agg_function is not None else None
        fill_function = self._fill_function(**self.fill_function_kwg) if self._fill_function is not None else None
        app_function = self._app_function(**self.app_function_kwg) if self._app_function is not None else None
        dictated = tuple([self.ts_creator.parametrization_hash if self.ts_creator is not None else my_hex(None),
                          my_hex(self.role),
                          agg_function.parametrization_hash if agg_function is not None else my_hex(None),
                          fill_function.parametrization_hash if fill_function is not None else my_hex(None),
                          app_function.parametrization_hash if app_function is not None else my_hex(None)])
        return dictated

    @property
    def parametrization_hash(self):
        hashed = my_hex(self.parametrization)
        return hashed

    def downcast_tx(self, vices):
        """
        Recalculate start_history wrt to the aggregation

        :param vices:
        :return:
        """

        assert len(list(vices.keys())) == 1
        name = list(vices.keys())[0]
        vice = vices[name]

        new_histories = {}

        agg = self._agg_function(**self.agg_function_kwg)

        for history_piece in vice.start_history.keys():

            start_history = vice.start_history[history_piece]

            start_history_casted = pandas.Series(data=[start_history]).apply(func=self.ts_creator.cast).values[0]
            start_history_casted_prev = pandas.to_datetime([start_history_casted]).shift(-1, freq=self.ts_creator.ts_frequency)[0]

            ts_ranged = pandas.date_range(start=start_history_casted_prev, end=start_history, freq=vice.ts_frequency)
            ts_ranged_casted = ts_ranged.to_series().apply(func=self.ts_creator.cast)
            ts_ranged = ts_ranged[(ts_ranged_casted == start_history_casted) & (ts_ranged < start_history)]

            n_missing = ts_ranged.shape[0]

            decision_lag = agg.decide_on_missing(n_missing=n_missing)

            new_start_history = pandas.to_datetime([start_history_casted]).shift(decision_lag + agg.lag, freq=self.ts_creator.ts_frequency)[0].isoformat()

            new_histories[history_piece] = new_start_history

        result_vice = UnholyVice(values=None, index=None,
                                 value_type=agg.value_type, ts_frequency=self.ts_creator.ts_frequency,
                                 lag_first_start_dt=None, first_start_dt=None, first_end_dt=None,
                                 lag_second_start_dt=None, second_start_dt=None, second_end_dt=None,
                                 start_history=new_histories,)

        return result_vice

    def downcast(self, vices):
        """
        Decrease frequency with agg function

        :param vices:
        :return:
        """

        input_hash = my_hex(tuple([x.signature for x in vices.values()]))

        name = list(vices.keys())[0]
        vice = vices[name]

        new_series_first_index = vice.lag_series_first.index.to_series().apply(func=self.ts_creator.cast)
        new_series_second_index = vice.lag_series_second.index.to_series().apply(func=self.ts_creator.cast)
        new_series_first = pandas.Series(index=new_series_first_index, data=vice.lag_series_first.values)
        new_series_second = pandas.Series(index=new_series_second_index, data=vice.lag_series_second.values)

        # there should be duplicates but no missing
        self.control_missing(new_series_first)
        self.control_missing(new_series_second)

        self.agg_function = self._agg_function(**self.agg_function_kwg)

        new_series_first_agg = self.agg_function.project_first({name: new_series_first})
        new_series_second_agg = self.agg_function.project_second({name: new_series_second})

        new_lag_first_start = vice.lag_series_first.index.to_series().apply(func=self.ts_creator.cast).shift(self.agg_function.lag, freq=self.ts_creator.ts_frequency)[0]
        new_first_start = vice.first_start_dt
        new_first_end = vice.first_end_dt
        new_lag_second_start = vice.lag_series_second.index.to_series().apply(func=self.ts_creator.cast).shift(self.agg_function.lag, freq=self.ts_creator.ts_frequency)[0]
        new_second_start = vice.second_start_dt
        new_second_end = vice.second_end_dt

        joint = pandas.concat((new_series_first_agg[new_series_first_agg.index <= new_first_end], new_series_second_agg[new_series_second_agg.index >= new_second_start]), axis=0)

        result_vice = UnholyVice(values=joint.values, index=joint.index,
                                 value_type=self.agg_function.value_type, ts_frequency=self.ts_creator.ts_frequency,
                                 lag_first_start_dt=new_lag_first_start, first_start_dt=new_first_start, first_end_dt=new_first_end,
                                 lag_second_start_dt=new_lag_second_start, second_start_dt=new_second_start, second_end_dt=new_second_end)
        result_vice.sign(incoming_chain=input_hash, projector_hash=self.parametrization_hash)
        return result_vice
    def control_missing(self, series):
        if pandas.isna(series).any():
            raise Exception("Missing values found: for recast should contain no missings")
    def upcast(self, vices):
        """
        Increase frequency with fill function

        :param vices:
        :return:
        """

        input_hash = my_hex(tuple([x.signature for x in vices.values()]))

        name = list(vices.keys())[0]
        vice = vices[name]

        new_series_first = pandas.Series(index=self.ts_creator.cast(start=vice.lag_start_dt, end=vice.mid_dt)).merge(right=vice.lag_series_first, how='left')
        new_series_second = pandas.Series(index=self.ts_creator.cast(start=vice.lag_mid_dt, end=vice.end_dt)).merge(right=vice.lag_series_second, how='left')

        # there should be missing but no duplicates
        self.control_duplicates(new_series_first)
        self.control_duplicates(new_series_second)

        self.fill_function = self._fill_function(**self.fill_function_kwg)

        new_series_first_fill = self.fill_function.project_first({name: new_series_first})
        new_series_second_fill = self.fill_function.project_second({name: new_series_second})

        casted_lag_first = self.ts_creator.cast(start=vice.lag_start_dt, end=vice.mid_dt)
        casted_lag_second = self.ts_creator.cast(start=vice.lag_mid_dt, end=vice.end_dt)
        casted_first = self.ts_creator.cast(start=vice.start_dt, end=vice.mid_dt)
        casted_second = self.ts_creator.cast(start=vice.mid_dt, end=vice.end_dt)

        new_lag_start = pandas.to_datetime(casted_lag_first.index).shift(self.fill_function.lag, freq=self.ts_creator.ts_frequency)[0]
        new_start = casted_first.index[0]
        new_lag_mid = pandas.to_datetime(casted_lag_second.index).shift(self.fill_function.lag, freq=self.ts_creator.ts_frequency)[0]
        new_mid = casted_second.index[0]
        new_end = casted_second.index[-1]

        joint = pandas.concat((new_series_first_fill, new_series_second_fill[new_series_second_fill.index >= new_mid]), axis=0)
        result_vice = UnholyVice(values=joint.values, index=joint.index,
                                 value_type=self.fill_function.value_type, ts_frequency=self.ts_creator.ts_frequency,
                                 lag_start_dt=new_lag_start, start_dt=new_start,
                                 lag_mid_dt=new_lag_mid, mid_dt=new_mid,
                                 end_dt=new_end)
        result_vice.sign(incoming_chain=input_hash, projector_hash=self.parametrization_hash)
        return result_vice

    def control_duplicates(self, series):
        ss = series.copy().reset_index()
        if not (ss.groupby(by='index').count() == 1).all().all():
            raise Exception("Duplicates found: for recast should contain no duplicates")

    def recast_tx(self, vices):
        """
        Recalculate start_history wrt to the aggregation

        :param vices:
        :return:
        """

        joint_freq = numpy.unique([vices[name].ts_frequency for name in vices.keys()])[0]

        app = self._app_function(**self.app_function_kwg)

        new_histories = {}
        for vice_key in vices.keys():
            vice = vices[vice_key]

            for history_piece in vice.start_history.keys():

                start_history = vice.start_history[history_piece]

                new_start_history = pandas.to_datetime([start_history]).shift(app.lag, freq=joint_freq)[0].isoformat()

                if history_piece not in new_histories.keys():
                    new_histories[history_piece] = new_start_history
                else:
                    new_start_history = max(new_histories[history_piece], new_start_history)
                    new_histories[history_piece] = new_start_history

        result_vice = UnholyVice(values=None, index=None,
                                 value_type=app.value_type, ts_frequency=self.ts_creator.ts_frequency,
                                 lag_first_start_dt=None, first_start_dt=None, first_end_dt=None,
                                 lag_second_start_dt=None, second_start_dt=None, second_end_dt=None,
                                 start_history=new_histories,)

        return result_vice

    def recast(self, vices):
        """
        Make representation of same frequency with app function

        :param vices:
        :return:
        """

        # run_time = time.time()
        input_hash = my_hex(tuple([x.signature for x in vices.values()]))
        # run_time = time.time() - run_time
        # print('hash', run_time)

        # run_time = time.time()
        # there should be no duplicates and no missing
        for name in vices.keys():
            self.control_missing(vices[name].lag_series_first)
            self.control_missing(vices[name].lag_series_second)
            self.control_duplicates(vices[name].lag_series_first)
            self.control_duplicates(vices[name].lag_series_second)
        # run_time = time.time() - run_time
        # print('no_duplicates', run_time)

        # run_time = time.time()
        self.app_function = self._app_function(**self.app_function_kwg)
        # run_time = time.time() - run_time
        # print('app_function', run_time)

        run_time = time.time()
        new_series_first_app = self.app_function.project_first({name: vices[name].lag_series_first for name in vices.keys()})
        run_time = time.time() - run_time
        # print('two_series: first', run_time)
        run_time = time.time()
        new_series_second_app = self.app_function.project_second({name: vices[name].lag_series_second for name in vices.keys()})
        run_time = time.time() - run_time
        # print('two_series: second', run_time)

        # run_time = time.time()
        assert numpy.unique([vices[name].first_start_dt for name in vices.keys()]).shape[0] == 1
        assert numpy.unique([vices[name].first_end_dt for name in vices.keys()]).shape[0] == 1
        assert numpy.unique([vices[name].second_start_dt for name in vices.keys()]).shape[0] == 1
        assert numpy.unique([vices[name].second_end_dt for name in vices.keys()]).shape[0] == 1

        assert numpy.unique([vices[name].ts_frequency for name in vices.keys()]).shape[0] == 1
        # run_time = time.time() - run_time
        # print('check_assert', run_time)
        # run_time = time.time()
        joint_freq = numpy.unique([vices[name].ts_frequency for name in vices.keys()])[0]
        # run_time = time.time() - run_time
        # print('joint_freq', run_time)

        # run_time = time.time()
        new_lag_first_start_dt = pandas.to_datetime(new_series_first_app.index).shift(self.app_function.lag, freq=joint_freq)[0].isoformat()
        new_first_start_dt = numpy.unique([vices[name].first_start_dt for name in vices.keys()])[0]
        new_first_end_dt = numpy.unique([vices[name].first_end_dt for name in vices.keys()])[0]
        new_lag_second_start_dt = pandas.to_datetime(new_series_second_app.index).shift(self.app_function.lag, freq=joint_freq)[0].isoformat()
        new_second_start_dt = numpy.unique([vices[name].second_start_dt for name in vices.keys()])[0]
        new_second_end_dt = numpy.unique([vices[name].second_end_dt for name in vices.keys()])[0]
        # run_time = time.time() - run_time
        # print('lag_recalc', run_time)

        assert new_lag_first_start_dt in new_series_first_app.index
        assert pandas.to_datetime([new_first_end_dt]).shift(-1, freq=joint_freq)[0].isoformat() in new_series_first_app.index
        assert new_second_start_dt in new_series_second_app.index
        assert pandas.to_datetime([new_second_end_dt]).shift(-1, freq=joint_freq)[0].isoformat() in new_series_second_app.index


        # run_time = time.time()
        joint = pandas.concat((
            new_series_first_app[new_series_first_app.index >= new_lag_first_start_dt],
            new_series_second_app[new_series_second_app.index >= new_second_start_dt],
        ),
                              axis=0)
        # run_time = time.time() - run_time
        # print('concat', run_time)
        # run_time = time.time()
        result_vice = UnholyVice(values=joint.values, index=joint.index,
                                 value_type=self.app_function.value_type, ts_frequency=joint_freq,
                                 lag_first_start_dt=new_lag_first_start_dt, first_start_dt=new_first_start_dt, first_end_dt=new_first_end_dt,
                                 lag_second_start_dt=new_lag_second_start_dt, second_start_dt=new_second_start_dt, second_end_dt=new_second_end_dt)
        result_vice.sign(incoming_chain=input_hash, projector_hash=self.parametrization_hash)
        # run_time = time.time() - run_time
        # print('unholy_vice', run_time)
        return result_vice

    def find_lag(self, vs):
        if self.role == 'downcast':
            return self.downcast_tx(vices=vs)
        elif self.role == 'upcast':
            raise Exception()
        elif self.role == 'recast':
            return self.recast_tx(vices=vs)
        else:
            raise Exception()
    def fit_transform(self, vs):
        self.fit = False
        if self.role == 'downcast':
            return self.downcast(vices=vs)
        elif self.role == 'upcast':
            return self.upcast(vices=vs)
        elif self.role == 'recast':
            return self.recast(vices=vs)
        else:
            raise Exception()
    def transform(self, vs):
        self.fit = True
        if self.role == 'downcast':
            return self.downcast(vices=vs)
        elif self.role == 'upcast':
            return self.upcast(vices=vs)
        elif self.role == 'recast':
            return self.recast(vices=vs)
        else:
            raise Exception()


class Item:
    def __init__(self, name):

        self.name = name

        self.pod_summary = None

        self.series = None
        self.value_type = None
        self.ts_frequency = None
        self.ts_delta = None
        # TODO: calendar logic to be implemented in future versions
        # self.ts_calendar = None

        self.reader = None

        self.publication_lag = None
        self.start_history = None

        self._load_pod()
        self._process_series()
    @property
    def parametrization(self):
        date_hash = hashlib.sha256(pandas.util.hash_pandas_object(self.series['DATE'], index=False).values).hexdigest()
        value_hash = hashlib.sha256(pandas.util.hash_pandas_object(self.series[self.name], index=False).values).hexdigest()
        dictated = tuple([my_hex(self.name),
                          # my_hex(self.series['DATE'].values),
                          # my_hex(self.series[self.name].values),
                          date_hash,
                          value_hash,
                          my_hex(self.value_type),
                          my_hex(self.ts_frequency),
                          my_hex(self.ts_delta),
                          my_hex(self.start_history)])
        return dictated
    @property
    def parametrization_hash(self):
        hashed = my_hex(self.parametrization)
        return hashed
    def _load_pod(self):
        self.pod_summary = pandas.read_excel(PJST.LOADING_DOCK.CTRL_FPATH, sheet_name=LD.DC_POD.OUTPUT_SHEET)
        assert all([x in self.pod_summary.columns.values for x in SystemFilesSignatures.CONTROLLER_SIGNATURE])
    def _process_series(self):

        ok_summary = self.pod_summary[self.pod_summary['status'] == "OK"].copy()
        assert self.name in ok_summary['name'].values

        name_in_pod = ok_summary.loc[ok_summary['name'] == self.name]

        self.value_type = name_in_pod['value_type'].values[0]
        self.ts_frequency = name_in_pod['ts_frequency'].values[0]
        self.publication_lag = name_in_pod['publication_lag'].values[0]

        reader = name_in_pod['reader'].values[0]
        reader = READERS[reader]
        reader = reader()
        self.reader = reader

        dock_d = PJST.LOADING_DOCK.CTRL_FPATH
        dock = pandas.read_excel(dock_d)
        dock = dock.set_index(LD.DC_POD.DOCK_INDEX_COLUMN)
        dock_path = dock.loc[LD.DC_POD.POD_ROW, LD.DC_POD.LOCATOR_COLUMN]

        # self.series = pandas.read_csv(Routing.DATA_SOURCE_FORMATTER.format(self.name),
        #                               sep=DataReadingConstants.CSV_SEPARATOR)

        self.series = new_read(source_formatter=f"{dock_path}{LD.DC_POD.POD_FOLDER}",
                               name=self.name, reader=self.reader, value_type=self.value_type)

        self.series[self.name] = self.series[self.name].shift(self.publication_lag)
        self.series = self.series.iloc[self.publication_lag:].copy()
        self.start_history = self.series[DataReadingConstants.DATE_COLUMN].min()

        hashed = hashlib.sha256(pandas.util.hash_pandas_object(self.series).values).hexdigest()

        hashed_control = name_in_pod['hashed'].values[0]
        assert hashed == hashed_control

        # TODO: calendar logic to be implemented in future versions
        # self.ts_calendar = name_in_pod['ts_calendar'].values[0]

    def to_vice(self):

        result = Vice(name=self.name,
                      values=self.series[self.name].values,
                      index=self.series[DataReadingConstants.DATE_COLUMN].values,
                      value_type=self.value_type,
                      ts_frequency=self.ts_frequency,
                      ts_delta=self.ts_delta,
                      publication_lag=self.publication_lag,
                      start_history={self.name: self.start_history})
        return result


class Collection:

    def __init__(self, source_items, stellar_paths, fold_generator, target, timeaxis_boundaries=None):

        self.source_items = source_items
        # TODO: once to be upgraded to stellar_map
        self.stellar_paths = stellar_paths

        # TODO: once to be upgraded to _route_stellar_map
        self._route_stellar_paths()

        self.fold_generator = fold_generator
        self.fold_generator.set_lag(joint_lag=self.stellar_paths.gather_lag())

        self._target_name = target
        if self.target:
            pass

        self.timeaxis = [None, None]

        self._check_frequencies_alignment()
        self._gather_timeaxis(timeaxis_boundaries=timeaxis_boundaries)

    @property
    def target(self):

        checked = self.target in self.source_items.names
        if not checked:
            raise Exception("Inconsistent stellar paths with respect to target")

        return self.source_items[self._target_name]

    def _route_stellar_paths(self):

        checked = all([y in self.source_items.names for x in self.stellar_paths for y in x.sources()])
        if not checked:
            raise Exception("Inconsistent stellar paths present")

    def _check_frequencies_alignment(self):

        checked = all([x.final_frequency() == self.target.ts_frequency for x in self.stellar_paths])
        if not checked:
            raise Exception("Inconsistent frequencies present")

    def _gather_timeaxis(self, timeaxis_boundaries):

        if timeaxis_boundaries[0] is None:
            self.timeaxis[0] = self.target.series.index.values.min()
        else:
            self.timeaxis[0] = timeaxis_boundaries[0]
        if timeaxis_boundaries[1] is None:
            self.timeaxis[1] = self.target.series.index.values.max()
        else:
            self.timeaxis[1] = timeaxis_boundaries[1]

    def manifold(self, fold_n, features):

        fold = self.fold_generator.fold(timeaxis=self.timeaxis, fold_n=fold_n)
        # TODO: once to be upgraded to stellar_mapping
        train_sample, test_sample = fold.stellar_pathing(source_items=self.source_items,
                                                         stellar_paths=self.stellar_paths,
                                                         features=features)

        return train_sample, test_sample


class SourceItemStack:
    ...

class StellarPaths:
    ...


class TimeAxe:
    ...

class FoldGenerator:
    def __init__(self, n_folds, joint_lag=None, val_rate=0.5, overlap_rate=0.5, freq=None, verbose=False):
        self.n_folds = n_folds
        self.joint_lag = joint_lag
        self.current_fold = -1
        self.val_rate = val_rate
        self.overlap_rate = overlap_rate
        self.freq = freq
        self.path = None
        self.verbose = verbose
    def copy(self):
        f = FoldGenerator(n_folds=self.n_folds, joint_lag=self.joint_lag, val_rate=self.val_rate, overlap_rate=self.overlap_rate)
        f.init_path(path_vertices=self.path.path_vertices, path_matrix=self.path.path_matrix, path_pseudo_edges=self.path.path_pseudo_edges, save_features=self.path.save_features)
        return f
    def set_lag(self, joint_lag):
        self.joint_lag = joint_lag
    def set_lag_from_delta(self, lag_delta, timeaxis):
        n_lagged = timeaxis[timeaxis <= (timeaxis[0] + lag_delta)].shape[0]
        self.set_lag(joint_lag=n_lagged)
    def init_path(self, path_vertices, path_matrix, path_pseudo_edges, save_features):
        self.path = Path(path_vertices, path_matrix, path_pseudo_edges, save_features)
    @property
    def folds(self):
        return list(range(self.n_folds))
    def find_lags(self, sources, features, target):
        self.path.find_lags(sources=sources,
                        features=features)
        history_pieces = numpy.unique([x for name in features for x in self.path.stock[name].start_history.keys()])
        sources_starts = {source.name: source.start_history for source in sources}

        unique_histories = {}
        diffs = []
        for history_piece in history_pieces:
            for name in features:
                if history_piece in self.path.stock[name].start_history.keys():
                    diff = (pandas.to_datetime(self.path.stock[name].start_history[history_piece])
                            - pandas.to_datetime(sources_starts[history_piece]))
                    diffs.append(diff)
                    if history_piece in unique_histories.keys():
                        new_history = max(unique_histories[history_piece], self.path.stock[name].start_history[history_piece])
                        unique_histories[history_piece] = new_history
                    else:
                        unique_histories[history_piece] = self.path.stock[name].start_history[history_piece]

        max_diff = max(diffs)
        max_history = max(list(unique_histories.values()))
        end_history = min([source.series[DataReadingConstants.DATE_COLUMN].values[-1] for source in sources])

        ts_frequency = self.path.stock[target].ts_frequency

        suggested_start = pandas.to_datetime(max_history)
        suggested_lag = max_diff

        self.freq.delta = max_diff

        date_range = pandas.date_range(start=suggested_start, end=end_history, freq=ts_frequency)
        return date_range, suggested_lag
    def fold(self, sources, features, timeaxis, fold_n=None):
        if self.joint_lag is None:
            raise Exception("Fold generation impossible: joint_lag not specified")
        if self.path is None:
            raise Exception("Fold generation impossible: path not initialized")
        if fold_n is None:
            if (self.current_fold + 1) < self.n_folds:
                self.current_fold += 1
            else:
                print("Folds exhausted; starting over")
                self.current_fold = 0
        else:
            self.current_fold = fold_n

        fold_size = timeaxis.shape[0] / (1 + (1 - self.overlap_rate) * (self.n_folds - 1))

        start_int = int(fold_size * (1 - self.overlap_rate) * self.current_fold)
        mid_int = int(fold_size * ((1 - self.overlap_rate) * self.current_fold + self.val_rate))
        end_int = min(int(fold_size * ((1 - self.overlap_rate) * self.current_fold + 1)), timeaxis.shape[0] - 1)

        lag_first_start_dt = pandas.to_datetime(timeaxis[start_int]) - self.freq.delta
        first_start_dt = pandas.to_datetime(timeaxis[start_int])
        first_end_dt = pandas.to_datetime([timeaxis[mid_int]]).shift(-1, freq=self.freq.to_end)[0]
        lag_second_start_dt = pandas.to_datetime(timeaxis[mid_int]) - self.freq.delta
        second_start_dt = pandas.to_datetime(timeaxis[mid_int])
        second_end_dt = pandas.to_datetime([timeaxis[end_int]]).shift(-1, freq=self.freq.to_end)[0]

        if self.current_fold > 0:
            lag_first_start_dt = pandas.to_datetime([lag_first_start_dt]).shift(-1, freq=self.freq.freq)[0]
        lag_second_start_dt = pandas.to_datetime([lag_second_start_dt]).shift(-1, freq=self.freq.freq)[0]

        if self.verbose:

            print()
            # print("Folding: {0} / {1} \n\n\tlag_start=\t{2}; \n\tstart=\t\t{3}; \n\tlag_mid=\t{4}; \n\tmid=\t\t{5}; \n\tend=\t\t{6}; \n\tval_rate=\t{7:.2f}\n\tsize=\t{8}".format(
            #     self.current_fold, self.n_folds, lag_start_dt, start_dt, lag_mid_dt, mid_dt, end_dt, self.val_rate, end_int - mid_int
            # ))


        self.path.route(sources=sources,
                        features=features,
                        lag_first_start_dt=lag_first_start_dt,
                        first_start_dt=first_start_dt,
                        first_end_dt=first_end_dt,
                        lag_second_start_dt=lag_second_start_dt,
                        second_start_dt=second_start_dt,
                        second_end_dt=second_end_dt)

        x_train = []
        for name in features:
            if self.path.stock[name].needs_pump():
                self.path.stock[name] = self.path.stock[name].pump()
            snippet = self.path.stock[name].current_series_first
            snippet.name = name
            x_train.append(snippet)
        x_train = pandas.concat(x_train, axis=1, ignore_index=False)
        x_test = []
        for name in features:
            snippet = self.path.stock[name].current_series_second
            snippet.name = name
            x_test.append(snippet)
        x_test = pandas.concat(x_test, axis=1, ignore_index=False)

        print(f"fold_sizes: {x_train.shape[0]}, {x_test.shape[0]}")
        # pandas.isna(x_train).any(), pandas.isna(x_test).any()

        assert not pandas.isna(x_train).any().any()
        assert not pandas.isna(x_test).any().any()

        return x_train, x_test


class PathPreView:
    def __init__(self, path_vertices, path_matrix, path_pseudo_edges, save_features, selection):
        self.path_vertices = numpy.array(path_vertices)
        self.path_matrix = path_matrix.astype(dtype=bool)
        self.path_pseudo_edges = numpy.array(path_pseudo_edges)
        self.save_features = save_features
        self.selection = selection
    @property
    def ix(self):
        result = self.path_vertices.tolist().index(self.selection)
        return result
    @property
    def parents(self):
        result = self.path_vertices[self.path_matrix[:, self.ix]]
        return result
    @property
    def children(self):
        result = self.path_vertices[self.path_matrix[self.ix, :]]
        return result
    def find_lag_local(self, stock):
        parental_stock = {s: stock[s] for s in stock.keys() if s in self.parents}
        resulting_unh_vice = self.path_pseudo_edges[self.ix].find_lag(vs=parental_stock)
        return resulting_unh_vice
    def grow_local(self, stock):
        parental_stock = {s: stock[s] for s in stock.keys() if s in self.parents}
        input_hash = my_hex(tuple([x.signature for x in parental_stock.values()]))
        projector_hash = self.path_pseudo_edges[self.ix].parametrization_hash

        if not self.save_features[self.ix]:
            resulting_unh_vice = self.path_pseudo_edges[self.ix].fit_transform(vs=parental_stock)
        elif self.chainlink_checker(input_hash=input_hash, projector_hash=projector_hash):
            resulting_unh_vice = UnholyVice()
            resulting_unh_vice.sign(incoming_chain=input_hash, projector_hash=projector_hash)
        else:
            resulting_unh_vice = self.path_pseudo_edges[self.ix].fit_transform(vs=parental_stock)
            resulting_hash = resulting_unh_vice.signature
            self.chainlink_writer(input_hash=input_hash, projector_hash=projector_hash, resulting_hash=resulting_hash)
            resulting_unh_vice.dump()
        return resulting_unh_vice
    def chainlink_checker(self, input_hash, projector_hash):
        input_hash = str(input_hash)
        projector_hash = str(projector_hash)
        d = '../data/other/chain_link.xlsx'
        chain_link = pandas.read_excel(d)
        chain_link['input_hash'] = chain_link['input_hash'].astype(dtype=str)
        chain_link['projector_hash'] = chain_link['projector_hash'].astype(dtype=str)
        mask = (chain_link['input_hash'] == input_hash) * (
                chain_link['projector_hash'] == projector_hash
        )
        if mask.sum() == 0:
            return False
        elif mask.sum() == 1:
            return True
        else:
            raise Exception("Too many records in the chainlink for {0} and {1}".format(input_hash, projector_hash))
    def chainlink_writer(self, input_hash, projector_hash, resulting_hash):
        input_hash = str(input_hash)
        projector_hash = str(projector_hash)
        resulting_hash = str(resulting_hash)
        d = '../data/other/chain_link.xlsx'
        chain_link = pandas.read_excel(d)
        chain_link['input_hash'] = chain_link['input_hash'].astype(dtype=str)
        chain_link['projector_hash'] = chain_link['projector_hash'].astype(dtype=str)
        chain_link['resulting_hash'] = chain_link['resulting_hash'].astype(dtype=str)
        mask = (chain_link['input_hash'] == input_hash) * (
                chain_link['projector_hash'] == projector_hash
        )
        if mask.sum() == 0:
            appendix = pandas.DataFrame(data={'input_hash': [input_hash],
                                              'projector_hash': [projector_hash],
                                              'resulting_hash': [resulting_hash]})
            chain_link = pandas.concat((chain_link, appendix), axis=0, ignore_index=True)
            chain_link.to_excel(d, index=False)
        elif mask.sum() == 1:
            raise Exception('Trying to write while the record exists: {0}, {1}'.format(input_hash, projector_hash))
        else:
            raise Exception("Too many records in the chainlink for {0} and {1}".format(input_hash, projector_hash))


class Path:
    def __init__(self, path_vertices, path_matrix, path_pseudo_edges, save_features):
        self.path_vertices = path_vertices
        self.path_matrix = path_matrix
        self.path_pseudo_edges = path_pseudo_edges
        self.save_features = save_features

        self.stock = {}
        self.stock_new = {}
        self.seeds = {}
        self.seeds_ung = {}
        self.finish = False

    def path(self, selection):
        result = PathPreView(path_vertices=self.path_vertices,
                             path_matrix=self.path_matrix,
                             path_pseudo_edges=self.path_pseudo_edges,
                             save_features=self.save_features,
                             selection=selection)
        return result
    def find_lags(self, sources, features):

        self.stock = {source.name: source.to_vice().to_unholy_vice(lag_first_start_dt=None, first_start_dt=None, first_end_dt=None, lag_second_start_dt=None, second_start_dt=None, second_end_dt=None) for source in sources}
        self.stock_new = {source.name: source.to_vice().to_unholy_vice(lag_first_start_dt=None, first_start_dt=None, first_end_dt=None, lag_second_start_dt=None, second_start_dt=None, second_end_dt=None) for source in sources}
        self.finish = False

        while not self.finish:

            self.seeds = {}
            self.seeds_ung = {}

            for sn in self.stock_new.keys():
                children = self.path(sn).children
                for child in children:
                    if child not in self.stock.keys():
                        parents = self.path(child).parents
                        if numpy.isin(parents, list(self.stock.keys())).all():
                            self.seeds[child] = {}
                        else:
                            self.seeds_ung[child] = {}
            self.stock_new = {}
            for seed in self.seeds_ung.keys():
                parents = self.path(seed).parents
                if numpy.isin(parents, list(self.stock.keys())).all():
                    self.seeds[seed] = {}
                    del self.seeds_ung[seed]

            for seed in self.seeds:
                run_time = time.time()
                self.stock_new[seed] = self.path(seed).find_lag_local(
                    stock=self.stock
                )
                self.stock[seed] = self.stock_new[seed]
                run_time = time.time() - run_time

            if numpy.isin(features, list(self.stock.keys())).all():
                self.finish = True

    def route(self, sources, features, lag_first_start_dt, first_start_dt, first_end_dt, lag_second_start_dt, second_start_dt, second_end_dt):

        self.stock = {source.name: source.to_vice().to_unholy_vice(lag_first_start_dt=lag_first_start_dt, first_start_dt=first_start_dt, first_end_dt=first_end_dt, lag_second_start_dt=lag_second_start_dt, second_start_dt=second_start_dt, second_end_dt=second_end_dt) for source in sources}
        self.stock_new = {source.name: source.to_vice().to_unholy_vice(lag_first_start_dt=lag_first_start_dt, first_start_dt=first_start_dt, first_end_dt=first_end_dt, lag_second_start_dt=lag_second_start_dt, second_start_dt=second_start_dt, second_end_dt=second_end_dt) for source in sources}
        self.finish = False

        for source in sources:
            self.stock[source.name].sign(incoming_chain=source.parametrization_hash, projector_hash=self.stock[source.name].parametrization_hash)
            self.stock_new[source.name].sign(incoming_chain=source.parametrization_hash, projector_hash=self.stock_new[source.name].parametrization_hash)

        while not self.finish:

            self.seeds = {}
            self.seeds_ung = {}

            for sn in self.stock_new.keys():
                children = self.path(sn).children
                for child in children:
                    if child not in self.stock.keys():
                        parents = self.path(child).parents
                        if numpy.isin(parents, list(self.stock.keys())).all():
                            self.seeds[child] = {}
                        else:
                            self.seeds_ung[child] = {}
            self.stock_new = {}
            for seed in self.seeds_ung.keys():
                parents = self.path(seed).parents
                if numpy.isin(parents, list(self.stock.keys())).all():
                    self.seeds[seed] = {}
                    del self.seeds_ung[seed]

            for seed in self.seeds:
                run_time = time.time()
                self.stock_new[seed] = self.path(seed).grow_local(
                    stock=self.stock
                )
                self.stock[seed] = self.stock_new[seed]
                run_time = time.time() - run_time

            if numpy.isin(features, list(self.stock.keys())).all():
                self.finish = True
