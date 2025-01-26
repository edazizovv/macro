#


#
import numpy
import pandas
from sklearn.linear_model import lasso_path, LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot
from scipy.stats import multivariate_normal, norm, ecdf


#
from macro.new_base import Path, Projector, Item, FoldGenerator
from macro.new_base_test_projectors import WindowAppGenerator
from macro.new_data_check import control, controller_view
from macro.new_base_truegarage import r2_metric, kendalltau_metric, somersd_metric, BasicLinearModel as MicroModel, BasicLassoSelectorModel as SelectorModel
from scipy import stats
from macro.new_base_trueuse_pods import features, path_pseudo_edges, path_matrix, path_vertices, sources, name_list, param_list
from macro.functional import sd_metric, pv_metric
from macro.new_base_bootstrap_multivariate_feature_selectors import bootstrap_direct_perf_improvement_sampler as pperf
from macro.new_base_trueuse_phase2_garage import lm_proxy, dt_proxy, rf_proxy, kn_proxy


#
loader_source = '../data/data_meta/loader_pitch.xlsx'
controller_source = '../data/other/controller_pitch.xlsx'

controller = control(loader_source)
controller.to_excel(controller_source, index=False)

"""
Stage -1: Fold generator set-up
"""

n_folds = 10
# n_folds = 2
joint_lag = 12
val_rate = 0.5
overlap_rate = 0.15
fg = FoldGenerator(n_folds=n_folds, joint_lag=joint_lag, val_rate=val_rate, overlap_rate=overlap_rate)

target = 'TLT_aggmean_pct'

savers = numpy.ones(shape=(len(path_vertices),)).astype(dtype=bool)
fg.init_path(path_vertices, path_matrix, path_pseudo_edges, savers)

timeaxis = sources[1].series['DATE'].values[sources[1].series['DATE'].values >= pandas.to_datetime('2007-01-01').isoformat()]

"""
Stage 2.1: LASSO-based skeleton
"""
# """
#                                         I'd recommend direct perf improvement with the parameters from below
performer = pv_metric
signif_alpha = 0.05                     # for impr-based: 0.05; for ks-based (both): 0.01; for impr-based direct: 0.05
n_bootstrap_samples = 1000
bootstrap_sample_size_rate = 1.0

perf_incr_threshold = 0
in_selection_thresh = 0.10              # for impr-based: 0.00; for ks-based (both): 0.50; for impr-based direct: 0.20

# threshold filter

performer = pv_metric
parameters = {
    'signif_alpha': signif_alpha,
    'n_bootstrap_samples': n_bootstrap_samples,
    'bootstrap_sample_size_rate': bootstrap_sample_size_rate,
    'perf_incr_threshold': perf_incr_threshold,
    'in_selection_thresh': in_selection_thresh,
    'pperf': pperf,
}

selected_features, selected_summary, inter_perf, final_perf = lm_proxy(
    fg=fg,
    sources=sources,
    features=features,
    target=target,
    timeaxis=timeaxis,
    performer=performer,
    parameters=parameters,
)

final_perf.to_excel('./final_perf_lm.xlsx')
# """

"""
Stage 2.1: Decision tree-based skeleton
"""

"""

performer = pv_metric
parameters = {
    'max_leaf_nodes': 5,
}

selected_features, selected_summary, inter_perf, final_perf = dt_proxy(
    fg=fg,
    sources=sources,
    features=features,
    target=target,
    timeaxis=timeaxis,
    performer=performer,
    parameters=parameters,
)

final_perf.to_excel('./final_perf_dt.xlsx')
"""


"""
Stage 2.1: Random forest-based skeleton
"""

"""

performer = pv_metric
parameters = {}

selected_features, selected_summary, inter_perf, final_perf = rf_proxy(
    fg=fg,
    sources=sources,
    features=features,
    target=target,
    timeaxis=timeaxis,
    performer=performer,
    parameters=parameters,
)

final_perf.to_excel('./final_perf_rf.xlsx')
"""


"""
Stage 2.1: Nearest neighbors-based skeleton
"""

"""

performer = pv_metric
parameters = {
    'n': 4,
    'm': 10,
    'w': 'distance',
}

selected_features, selected_summary, inter_perf, final_perf = kn_proxy(
    fg=fg,
    sources=sources,
    features=features,
    target=target,
    timeaxis=timeaxis,
    performer=performer,
    parameters=parameters,
)

final_perf.to_excel('./final_perf_kn.xlsx')
"""
