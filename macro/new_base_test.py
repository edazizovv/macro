#


#
import numpy
import pandas


#
from new_base import Path, Projector, Item, FoldGenerator
from new_base_test_projectors import WindowAppGenerator
from new_data_check import control, controller_view


#
loader_source = './data_meta/loader_pitch.xlsx'
controller_source = './controller_pitch.xlsx'

controller = control(loader_source)
controller.to_excel(controller_source, index=False)

# check_original, check_typec, check_tsc = controller_view(loader=loader_source, name='BAMLC0A0CM')

# TODO:
# QS does not work properly when there are non-standard months available (ofc it does not, we need a solution here)

m_mean = lambda s: s.mean()
m_median = lambda s: s.median()

path_vertices = ['AAA', 'AAA_mean3', 'AAA_median6']
path_matrix = numpy.array([[0, 1, 1], [0, 0, 0], [0, 0, 0]])
path_pseudo_edges = [None,
                     Projector(ts_creator=None,
                               role='recast',
                               app_function=WindowAppGenerator,
                               app_function_kwg={'func': m_mean,
                                                 'window': 3}),
                     Projector(ts_creator=None,
                               role='recast',
                               app_function=WindowAppGenerator,
                               app_function_kwg={'func': m_median,
                                                 'window': 6})
                     ]
'''
path = Path(path_vertices, path_matrix, path_pseudo_edges)

sources = [Item(name='AAA', loader_source=loader_source, controller_source=controller_source)]
features = ['AAA_mean3', 'AAA_median6']

lag_start_dt = '1999-01-01'
start_dt = '2000-01-01'
lag_mid_dt = '2003-01-01'
mid_dt = '2004-01-01'
end_dt = '2010-01-01'
path.route(sources=sources,
           features=features,
           lag_start_dt=lag_start_dt,
           start_dt=start_dt,
           lag_mid_dt=lag_mid_dt,
           mid_dt=mid_dt,
           end_dt=end_dt)
'''

n_folds = 10
joint_lag = 12
val_rate = 0.5
fg = FoldGenerator(n_folds=n_folds, joint_lag=joint_lag, val_rate=val_rate)

fg.init_path(path_vertices, path_matrix, path_pseudo_edges)

sources = [Item(name='AAA', loader_source=loader_source, controller_source=controller_source)]
features = ['AAA_mean3', 'AAA_median6']
timeaxis = sources[0].series['DATE'].values
fold_n = None
# digger = sources[0].series.copy().set_index('DATE')
# ax = digger.plot()  # s is an instance of Series
# fig = ax.get_figure()
# fig.savefig('./pic.png')
# raise Exception("Diggers")
fg.fold(sources, features, timeaxis, fold_n=fold_n)

x_train = []
for name in features:
    snippet = fg.path.stock[name].current_series_first
    snippet.name = name
    x_train.append(snippet)
x_train = pandas.concat(x_train, axis=1, ignore_index=False)
x_test = []
for name in features:
    snippet = fg.path.stock[name].current_series_second
    snippet.name = name
    x_test.append(snippet)
x_test = pandas.concat(x_test, axis=1, ignore_index=False)


"""
import pandas
zex_1 = path.stock['AAA'].series
zex_2 = path.stock['AAA_mean3'].series
zex_3 = path.stock['AAA_median6'].series
zex = pandas.DataFrame(data={'zex_1': zex_1, 'zex_2': zex_2, 'zex_3': zex_3})
import matplotlib
matplotlib.use('TkAgg',force=True)
from matplotlib import pyplot
print("Switched to:",matplotlib.get_backend())
zex.plot()
"""
