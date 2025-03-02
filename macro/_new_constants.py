#


#
import numpy


#


#
class DataReadingConstants:
    DATE_COLUMN = 'DATE'
    CSV_SEPARATOR = ','


class ValueTypes:
    CONTINUOUS = 'CONTINUOUS'
    CONTINUOUS_DOWNCAST = 'float'
    ORDINAL = 'ORDINAL'
    ORDINAL_DOWNCAST = 'int'
    CATEGORICAL = 'CATEGORICAL'
    CATEGORICAL_DOWNCAST = 'object'
    MISSING = ['.']

"""
class TSTypes:
    DAILY = 'D'
    MONTHLY = 'M'
    QUARTERLY = 'Q'
    YEARLY = 'A'
"""

class DQControlsErrors:
    OK = 'OK'
    DIM_ERROR = 'DIM_ERROR'
    DATE_COL_ERROR = 'DATE_COL_ERROR'


class Routing:
    DATA_SOURCE_FORMATTER = '../data/data/{0}.csv'


class SystemFilesSignatures:
    LOADER_SIGNATURE = ['name', 'value_type', 'reader', 'ts_frequency', 'publication_lag']
    CONTROLLER_SIGNATURE = ['name', 'status',
                            'n_total', 'before_value_miss', 'before_value_miss_pct',
                            'after_value_miss', 'after_value_miss_pct',
                            'hashed',
                            'value_type', 'ts_frequency', 'publication_lag',
                            'reader']
