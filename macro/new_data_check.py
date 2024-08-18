#
import hashlib


#
import numpy
import pandas


#
from new_base_utils import new_read
from new_constants import DQControlsErrors, DataReadingConstants, ValueTypes, SystemFilesSignatures, Routing  # TSTypes,


#
# TODO: ts_calendar to be implemented
def controller_view(loader, name):
    loaded = pandas.read_excel(loader)

    source_formatter = Routing.DATA_SOURCE_FORMATTER
    i = loaded['name'].values.tolist().index(name)
    value_type = loaded['value_type'].values[i]
    ts_frequency = loaded['ts_frequency'].values[i]

    series = pandas.read_csv(source_formatter.format(name), sep=DataReadingConstants.CSV_SEPARATOR)

    original_series = series.copy()

    series.loc[numpy.isin(series[name].values, ValueTypes.MISSING), name] = numpy.nan

    if value_type == ValueTypes.CONTINUOUS:
        try:
            series[name] = pandas.to_numeric(series[name], errors='raise', downcast=ValueTypes.CONTINUOUS_DOWNCAST)
        except ValueError:
            raise Exception()
    elif value_type == ValueTypes.ORDINAL:
        try:
            series[name] = pandas.to_numeric(series[name], errors='raise', downcast=ValueTypes.ORDINAL_DOWNCAST)
        except ValueError:
            raise Exception()
    elif value_type == ValueTypes.CATEGORICAL:
        series[name] = series[name].astype(dtype=ValueTypes.CATEGORICAL_DOWNCAST)
    else:
        raise Exception("Invalid value_type specified: {0}; "
                        "should be one of the following: CONTINUOUS, ORDINAL, CATEGORICAL".format(value_type))

    series[DataReadingConstants.DATE_COLUMN] = \
        pandas.to_datetime(series[DataReadingConstants.DATE_COLUMN]).dt.tz_localize('UTC') \
            .apply(func=lambda x: x.isoformat())

    # TODO: timezones to be controlled
    # TODO: higher frequencies to be implemented
    min_date, max_date = series[DataReadingConstants.DATE_COLUMN].min(), series[DataReadingConstants.DATE_COLUMN].max()
    """
    if ts_frequency == TSTypes.DAILY:
        bench_ts_range = pandas.date_range(start=min_date, end=max_date, freq='D', tz='UTC')
    elif ts_frequency == TSTypes.MONTHLY:
        bench_ts_range = pandas.date_range(start=min_date, end=max_date, freq='MS', tz='UTC')
    elif ts_frequency == TSTypes.QUARTERLY:
        bench_ts_range = pandas.date_range(start=min_date, end=max_date, freq='QS', tz='UTC')
    elif ts_frequency == TSTypes.YEARLY:
        bench_ts_range = pandas.date_range(start=min_date, end=max_date, freq='YS', tz='UTC')
    else:
        raise Exception("Invalid ts_frequency specified: {0}; "
                        "should be one of the following: D, M, Q, A".format(ts_frequency))
    """
    # https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases
    bench_ts_range = pandas.date_range(start=min_date, end=max_date, freq=ts_frequency, tz='UTC')
    bench_ts_range = pandas.Series(bench_ts_range).apply(func=lambda x: x.isoformat()).values
    series_ts_controlled = pandas.DataFrame(data={DataReadingConstants.DATE_COLUMN: bench_ts_range})
    series_ts_controlled = series_ts_controlled.merge(right=series,
                                                      left_on=DataReadingConstants.DATE_COLUMN,
                                                      right_on=DataReadingConstants.DATE_COLUMN,
                                                      how='left')
    type_controlled_series = series.copy()

    return original_series, type_controlled_series, series_ts_controlled


def control_item(source_formatter, name, value_type, ts_frequency):

    series = new_read(source_formatter=source_formatter, name=name, value_type=value_type)

    # TODO: timezones to be controlled
    # TODO: higher frequencies to be implemented
    min_date, max_date = series[DataReadingConstants.DATE_COLUMN].min(), series[DataReadingConstants.DATE_COLUMN].max()
    """
    if ts_frequency == TSTypes.DAILY:
        bench_ts_range = pandas.date_range(start=min_date, end=max_date, freq='D', tz='UTC')
    elif ts_frequency == TSTypes.MONTHLY:
        bench_ts_range = pandas.date_range(start=min_date, end=max_date, freq='MS', tz='UTC')
    elif ts_frequency == TSTypes.QUARTERLY:
        bench_ts_range = pandas.date_range(start=min_date, end=max_date, freq='QS', tz='UTC')
    elif ts_frequency == TSTypes.YEARLY:
        bench_ts_range = pandas.date_range(start=min_date, end=max_date, freq='YS', tz='UTC')
    else:
        raise Exception("Invalid ts_frequency specified: {0}; "
                        "should be one of the following: D, M, Q, A".format(ts_frequency))
    """
    # https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases
    bench_ts_range = pandas.date_range(start=min_date, end=max_date, freq=ts_frequency, tz='UTC')
    bench_ts_range = pandas.Series(bench_ts_range).apply(func=lambda x: x.isoformat()).values
    series_ts_controlled = pandas.DataFrame(data={DataReadingConstants.DATE_COLUMN: bench_ts_range})
    series_ts_controlled = series_ts_controlled.merge(right=series,
                                                      left_on=DataReadingConstants.DATE_COLUMN,
                                                      right_on=DataReadingConstants.DATE_COLUMN,
                                                      how='left')
    check_before_value_miss = pandas.isna(series[name]).sum()
    check_after_value_miss = pandas.isna(series_ts_controlled[name]).sum()

    hashed = hashlib.sha256(pandas.util.hash_pandas_object(series).values).hexdigest()

    n_total = series_ts_controlled.shape[0]

    return DQControlsErrors.OK, n_total, check_before_value_miss, check_after_value_miss, hashed


def control(loader):

    loaded = pandas.read_excel(loader)
    assert all([x in loaded.columns for x in SystemFilesSignatures.LOADER_SIGNATURE])

    statuses, before_value_misses, after_value_misses, hashed_values = [], [], [], []
    n_total, before_value_misses_pct, after_value_misses_pct = [], [], []
    for i in range(loaded.shape[0]):
        status, nn, before_value_miss, after_value_miss, hashed = control_item(
            source_formatter=Routing.DATA_SOURCE_FORMATTER,
            name=loaded['name'].values[i],
            value_type=loaded['value_type'].values[i],
            ts_frequency=loaded['ts_frequency'].values[i]
        )
        statuses.append(status)
        before_value_misses.append(before_value_miss)
        after_value_misses.append(after_value_miss)
        hashed_values.append(hashed)
        n_total.append(nn)
        before_value_misses_pct.append(before_value_miss / nn)
        after_value_misses_pct.append(after_value_miss / nn)
    result = pandas.DataFrame(data={'name': loaded['name'].values,
                                    'status': statuses,
                                    'n_total': n_total,
                                    'before_value_miss': before_value_misses,
                                    'before_value_miss_pct': before_value_misses_pct,
                                    'after_value_miss': after_value_misses,
                                    'after_value_miss_pct': after_value_misses_pct,
                                    'hashed': hashed_values,
                                    'value_type': loaded['value_type'].values,
                                    'ts_frequency': loaded['ts_frequency'].values})
    assert result.columns.values.tolist() == SystemFilesSignatures.CONTROLLER_SIGNATURE
    return result
