#
import hashlib


#
import numpy
import pandas


#
from macro._new_constants import DQControlsErrors, DataReadingConstants, ValueTypes, SystemFilesSignatures, Routing  # TSTypes,


#
def my_hex(x):
    return hashlib.sha256(str(x).encode('UTF-8')).hexdigest()


def my_dict_hex(dx):
    return my_hex(tuple([dx[x] for x in sorted(dx.keys())]))


def my_func_hex(f):
    return my_hex(f.__name__)


def new_read(source_formatter, name, reader, value_type):

    series = reader.read(source_formatter=source_formatter, name=name)
    if not (series.shape[1] == 2):
        return DQControlsErrors.DIM_ERROR, None, None, None
    if not (series.columns[0] == DataReadingConstants.DATE_COLUMN):
        return DQControlsErrors.DATE_COL_ERROR, None, None, None

    series = series.sort_values(by=DataReadingConstants.DATE_COLUMN)
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

    return series
