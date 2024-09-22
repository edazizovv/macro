#
import json


#
import numpy
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, ElasticNet


#
from macro.new_base_test_projectors import SimpleAggregator, WindowRollImpulse, RangedPct, Stayer, Binner, UGMARIMAClass, UGMSklearnClass, AutoArima, DefiniteArima


#
a_function_translator = {
    'SimpleAggregator': SimpleAggregator,
    'WindowRollImpulse': WindowRollImpulse,
    'RangedPct': RangedPct,
    'Stayer': Stayer,
    'Binner': Binner,
    'UGMARIMAClass': UGMARIMAClass,
    'UGMSklearnClass': UGMSklearnClass,
}


def a_function_kwg_translator(a):
    b = json.loads(a)
    if 'func' in b.keys():
        b['func'] = a_function_kwg_translator_func[b['func']]
    if 'model' in b.keys():
        b['model'] = a_function_kwg_translator_model[b['model']]
    return b


def p10(x):
    return numpy.quantile(x, 0.1)


def p90(x):
    return numpy.quantile(x, 0.9)


a_function_kwg_translator_func = {
    'none': None,
    'mean': numpy.mean,
    'median': numpy.median,
    'std': numpy.std,
    'min': numpy.min,
    'max': numpy.max,
    'p10': p10,
    'p90': p90,
}

a_function_kwg_translator_model = {
    'arima_auto': AutoArima,
    'arima_def': DefiniteArima,
    'DecisionTreeRegressor': DecisionTreeRegressor,
    'RandomForestRegressor': RandomForestRegressor,
    'HistGradientBoostingRegressor': HistGradientBoostingRegressor,
    'KNeighborsRegressor': KNeighborsRegressor,
    'LinearRegression': LinearRegression,
    'ElasticNet': ElasticNet,
}
