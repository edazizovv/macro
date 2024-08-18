#


#
import numpy
import pandas
from scipy import stats


#


#
def trans_diff(series):
    return series.diff()


def trans_pct(series):
    return series.pct_change()


def repr_will_double(series):
    def geometric_mean(s):
        return numpy.log(2) / numpy.log(s[-1])
    return (series / 100 + 1).rolling(window=24).apply(geometric_mean, raw=True)


def repr_n_for_double(series):
    def geometric_mean(s):
        j = 0
        for j in range(s.shape[0]):
            if s[:j+1].prod() >= 2:
                break
        return j + 1
    return (series / 100 + 1).rolling(window=24).apply(geometric_mean, raw=True)


def repr_dot_alfa_generator(w):
    def estimate_alfa(series):
        if (series.shape[0] - 1) % 2 == 0:
            mid_ix = (series.shape[0] - 1) / 2
            mid_value = (series[((series.shape[0] - 2) / 2)] + series[(series.shape[0] / 2)]) / 2
        else:
            mid_ix = (series.shape[0] - 1) // 2
            mid_value = series[mid_ix]
        beta = (series[-1] - series[0]) / (series.shape[0] - 1)
        alfa = mid_value - beta * mid_ix
        return alfa
    def result_func(series):
        return (series / 100 + 1).rolling(window=w).apply(estimate_alfa, raw=True)
    return result_func


def repr_dot_beta_generator(w):
    def estimate_beta(series):
        return (series[-1] - series[0]) / (series.shape[0] - 1)
    def result_func(series):
        return (series / 100 + 1).rolling(window=w).apply(estimate_beta, raw=True)
    return result_func


def repr_geometric_mean_base_pct_generator(w):
    def geometric_mean(series):
        return numpy.power(series.prod(), 1/series.shape[0])
    def result_func(series):
        return (series / 100 + 1).rolling(window=w).apply(geometric_mean, raw=True)
    return result_func


def repr_lag_generator(n_lag):
    def result_func(series):
        return series.shift(n_lag)
    return result_func


def repr_mean_window_generator(w):
    def result_func(series):
        return series.shift(1).rolling(window=w).mean()
    return result_func


def measure_pearsonr(a, b):
    if pandas.infer_freq(a.index) == pandas.infer_freq(b.index):
        if (a.index == b.index).all():
            result = stats.pearsonr(a.values, b.values)
            return result.statistic
        else:
            return numpy.nan
    else:
        return numpy.nan


def measure_kendalltau(a, b):
    if pandas.infer_freq(a.index) == pandas.infer_freq(b.index):
        if (a.index == b.index).all():
            result = stats.kendalltau(a.values, b.values)
            return result.statistic
        else:
            return numpy.nan
    else:
        return numpy.nan


def cast_to_frequency(series, freq):
    ...


def upcast_to_frequency(series, freq, splitter=None):
    se = pandas.Series(data=series, index=pandas.date_range(start=series.index.min(), end=series.index.max(), freq=freq))
    if splitter is not None:
        se = se.fillna(splitter(se))
    se = se.fillna(method='ffill')
    return se


def downcast_to_frequency(series, freq, filler):
    d_range = pandas.date_range(start=series.index.min(), end=series.index.max(), freq=freq)
    assert d_range.max() <= series.index.max()
    if d_range.min() > series.index.min():
        d_range = pandas.date_range(start=series.index.min() - pandas.tseries.frequencies.to_offset(freq),
                                    end=series.index.max(), freq=freq)
    # if d_range.max() < series.index.max():
    #     d_range = pandas.date_range(start=series.index.min(), end=series.index.max() + pandas.tseries.frequencies.to_offset(freq), freq=freq)
    binned = numpy.digitize(series.index.values.view('i8'), d_range.values.view('i8'))
    # binned[-1] = binned[-2]
    se = pandas.DataFrame(series)
    se['binned'] = binned
    se = se.sort_values(by='binned')
    if filler is None:
        filler = 'last'
    se = se.groupby(by='binned').apply(filler)
    d_range.name = 'date'
    se.index = pandas.Series(se.index).apply(func=lambda x: d_range[x-1])
    assert se.shape[1] == 1
    se = se[se.columns[0]]
    return se


def upcast_generator(freq, splitter=None):
    def result(series):
        return upcast_to_frequency(series=series, freq=freq, splitter=splitter)
    return result


def downcast_generator(freq, filler=None):
    def result(series):
        return downcast_to_frequency(series=series, freq=freq, filler=filler)
    return result
