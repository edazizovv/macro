
import os
import numpy
import pandas
import json

def axer(x, ax):
    resp = ax[x <= ax][0] if (x <= ax).sum() > 0 else None
    return resp


class Element:
    def __init__(self, source):
        self.source = source

        self.time_axis = None
        self.name = None

        _ = self.load_raw()
    def load_raw(self):
        frame = pandas.read_csv(self.source, na_values=['.'])
        assert frame.shape[1] == 2
        assert frame.columns[0] == 'DATE'
        self.name = frame.columns[1]
        return frame
    def aggregate(self, frame, target_axis, gb_func):
        self.time_axis = target_axis
        frame['jx'] = frame['DATE'].apply(func=axer, args=(target_axis.values,))
        frame_agg = frame.groupby(by='jx')[self.name].apply(gb_func)
        return frame_agg
    @property
    def frame(self):
        frame = self.load_raw()
        return frame
    def represent(self, representor, representor_kwg, origin, target_axis, gb_func, target_axis_subx=None):
        if origin:
            resulted = representor.represent(self.frame, **representor_kwg)
            resulted = self.aggregate(resulted, target_axis=target_axis, gb_func=gb_func)
        else:
            resulted = self.aggregate(self.frame, target_axis=target_axis, gb_func=gb_func)
            resulted = representor.represent(resulted, **representor_kwg)
        if target_axis_subx is not None:
            resulted = resulted[target_axis_subx]
        return resulted


class ElementOld:
    def __init__(self, group, source, lag):
        self.group = group
        self.source = source

        self.lag = lag

        self.time_axis = None
        self.name = None

        _ = self.load_raw()
    def load_raw(self):
        frame = pandas.read_csv(self.source, na_values=['.'])
        assert frame.shape[1] == 2
        assert frame.columns[0] == 'DATE'
        self.name = frame.columns[1]
        frame = frame.sort_values(by='DATE')
        return frame
    def aggregate(self, target_axis, gb_func):
        frame = self.load_raw()
        self.time_axis = target_axis
        frame['jx'] = frame['DATE'].apply(func=axer, args=(target_axis.values,))
        frame_agg = frame.groupby(by='jx')[self.name].apply(gb_func)
        frame_agg = frame_agg.sort_index()
        return frame_agg
    @property
    def frame(self):
        frame = self.load_raw()
        return frame


class Frame:
    def __init__(self, d='../data/data/'):
        self.d = d
        self.elements = []

        self.time_axis = None
        for f in os.listdir('{0}'.format(d)):
            fx = '{0}{1}'.format(d, f)
            element = Element(source=fx)
            self.elements.append(element)

    def __getitem__(self, item):
        for j in range(len(self.elements)):
            if self.elements[j].name == item:
                return self.elements[j]
        raise KeyError("Invalid name requested: {0}".format(item))

    @property
    def names(self):
        return [e.name for e in self.elements]

    @property
    def n(self):
        return len(self.names)


class FrameOld:
    def __init__(self, d='../data/data/'):
        self.d = d
        self.elements = []

        with open('{0}lags.json'.format(d), 'r') as f:
            self.lags = json.load(f)
        
        self.cast_data = None
        self.repr_data = None
        self.last_casted_freq = None
        self.names_repr = None
        self.min_date = None
        self.max_date = None

        for f in [x for x in os.listdir('{0}'.format(d)) if '.csv' in x]:
            fx = '{0}{1}'.format(d, f)
            name = f[:f.index('.csv')]
            if name in self.lags.keys():
                element = ElementOld(group=None, source=fx, lag=self.lags[name])
                self.elements.append(element)
            else:
                print('Not found in lags: {0}'.format(name))

    def __getitem__(self, item):
        for j in range(len(self.elements)):
            if self.elements[j].name == item:
                return self.elements[j]
        raise KeyError("Invalid name requested: {0}".format(item))

    def tighten_dates(self, cutoff_date):
        data = self.repr_data.copy()
        data = data[data.index >= cutoff_date]
        data = data.dropna(axis=1)

        return data

    def cast(self, time_axis, gb_funcs, fill_values):

        data = []
        if isinstance(gb_funcs, list):
            for j in range(len(self.elements)):
                e = self.elements[j].aggregate(target_axis=time_axis, gb_func=gb_funcs[j])
                data.append(e)
        else:
            for j in range(len(self.elements)):
                e = self.elements[j].aggregate(target_axis=time_axis, gb_func=gb_funcs)
                data.append(e)
        data = pandas.concat(data, axis=1, ignore_index=False)
        data = data.sort_index()
        if isinstance(gb_funcs, list):
            for j in range(data.shape[1]):
                if fill_values[j] == 'ffill':
                    data[data.columns[j]] = data[data.columns[j]].ffill()
                else:
                    data[data.columns[j]] = data[data.columns[j]].fillna(fill_values[j])
        else:
            if fill_values == 'ffill':
                data = data.ffill()
            else:
                data = data.fillna(fill_values)
        for c in data.columns:
            data[c] = data[c].shift(self.lags[c])
        self.cast_data = data.copy()

        return data

    def represent(self, factors, win_type_s, win_kwgs_s):

        data = self.cast_data.copy()
        names_repr = []

        if len(factors) > 0:
            for j in range(len(factors)):

                fg = factors[j]
                win_type = win_type_s[j]
                win_kwgs = win_kwgs_s[j]
                for i in range(len(win_type)):
                    if win_type[i] == 'full_apply':
                        names_add = ['{0}__full_{1}'.format(x, win_kwgs[i]['func'].__name__) for x in fg]
                        names_repr += names_add
                        data[names_add] = win_kwgs[i]['func'](data[fg])
                    elif win_type[i] == 'rolling':
                        names_add = ['{0}__rolling_{1}__{2}__{3}'.format(x, win_kwgs[i]['window'], win_kwgs[i]['win_func'], win_kwgs[i]['agg_func']) for x in fg]
                        names_repr += names_add
                        data[names_add] = data[fg].rolling(window=win_kwgs[i]['window'], win_type=win_kwgs[i]['win_func']).agg(win_kwgs[i]['agg_func'])
                    elif win_type[i] == 'ewm':
                        names_add = ['{0}__ewm_{1}__{2}'.format(x, win_kwgs[i]['alpha'], win_kwgs[i]['agg_func']) for x in fg]
                        names_repr += names_add
                        data[names_add] = data[fg].ewm(alpha=win_kwgs[i]['alpha']).agg(win_kwgs[i]['agg_func'])
                    elif win_type[i] == 'pct':
                        names_add = ['{0}__pct'.format(x) for x in fg]
                        names_repr += names_add
                        data[names_add] = data[fg].pct_change()
                    elif win_type[i] == 'switchblade':
                        names_add = ['{0}__switchblade_{1}__{2}__{3}'.format(x, win_kwgs[i]['window'], win_kwgs[i]['win_func'], win_kwgs[i]['agg_func'].__name__) for x in fg]
                        names_repr += names_add
                        data[names_add] = data[fg].shift(-win_kwgs[i]['window']).rolling(window=win_kwgs[i]['window'], win_type=win_kwgs[i]['win_func']).agg(win_kwgs[i]['agg_func'])
                    else:
                        raise ValueError('win_type: {0}'.format(win_type[i]))
                    fg = names_add

        for c in data.columns:
            data.loc[pandas.isna(data[c]), c] = data[c].mean()
            data.loc[data[c] == numpy.inf, c] = data.loc[data[c] != numpy.inf, c].max()
            data.loc[data[c] == -numpy.inf, c] = data.loc[data[c] != -numpy.inf, c].min()
        self.names_repr = names_repr

        self.repr_data = data.copy()
        return data

"""
f = FrameOld()
target = 'TLT'
target_e = f[target]
time_axis = target_e.frame['DATE'].copy()
_ = f.cast(time_axis=time_axis, gb_funcs='last', fill_values='ffill')

factors = [['T10YIEM'],
           ['T10YIEM']]
windows = [4, 8, 16]
win_funcs = [None, None]
funcs = ['mean', 'mean']
# _ = f.represent(factors=factors, windows=windows, win_funcs=win_funcs, funcs=funcs)

# result = f.tighten_dates(cutoff_date='2007-01-01')
"""
