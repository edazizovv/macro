import os
import numpy
import pandas
import seaborn
from matplotlib import pyplot
from macro.new_base_test_projectors import UGMARIMAClass, AutoArima

d = '../data/data_folds/'
os.listdir(d)

os.chdir('../../')
os.listdir('../')

dd = './data/data_folds/data_{0}_{1}.xlsx'

j = 9

role = 'train'
data_train = pandas.read_excel(dd.format(role, j))
data_train = data_train.rename(columns={'Unnamed: 0': 'date'})
data_train['date'] = pandas.to_datetime(data_train['date'])
data_train = data_train.set_index('date')
data_train.head(5)

role = 'test'
data_test = pandas.read_excel(dd.format(role, j))
data_test = data_test.rename(columns={'Unnamed: 0': 'date'})
data_test['date'] = pandas.to_datetime(data_test['date'])
data_test = data_test.set_index('date')
data_test.head(5)

"""
fig, ax = pyplot.subplots(figsize=(10, 10))
seaborn.lineplot(
    x=data_train["date"],
    y=data_train["AAA"],
    color='blue',
    ax=ax
)
seaborn.lineplot(
    x=data_test["date"],
    y=data_test["AAA"],
    color='orange',
    ax=ax
)
ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
"""

arima_kwg = {"model": AutoArima, "model_kwargs": {}, "window": 6}

model = UGMARIMAClass(**arima_kwg)

series_first = {'AAA': data_train['AAA']}
series_second = {'AAA': data_test['AAA']}

result_first = model.project_first(series_first)

result_second = model.project_second(series_second)

print(model.model.max_window, model.model.max_d, model.model.arima, model.model.fitted_p, model.model.fitted_d, model.model.fitted_q, model.model.fitted_trend)

v0 = data_train["AAA"].values[0]
result_first_ = numpy.exp(result_first)
result_first_[0] = v0
result_first_ = result_first_.cumprod()

v0 = data_test["AAA"].values[0]
result_second_ = numpy.exp(result_second)
result_second_[0] = v0
result_second_ = result_second_.cumprod()

"""
fig, ax = pyplot.subplots(figsize=(10, 10))
seaborn.lineplot(
    x=data_train["date"],
    y=data_train["AAA"],
    color='navy',
    ax=ax
)
seaborn.lineplot(
    x=data_train["date"],
    y=result_first_,
    color='blue',
    ax=ax
)
seaborn.lineplot(
    x=data_test["date"],
    y=result_second_,
    color='yellow',
    ax=ax
)
seaborn.lineplot(
    x=data_test["date"],
    y=data_test["AAA"],
    color='orange',
    ax=ax
)
ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
"""
