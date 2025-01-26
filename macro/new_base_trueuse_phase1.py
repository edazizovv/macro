#


#
import numpy
import pandas
from scipy.stats import multivariate_normal, norm
from sklearn.linear_model import LinearRegression


#
from macro.new_base import Path, Projector, Item, FoldGenerator
from macro.new_base_test_projectors import WindowAppGenerator, VincentClassMobsterS
from macro.new_data_check import control, controller_view
from functional import pa_metric, r2_metric
from macro.new_base_truegarage import r2_metric, kendalltau_metric, somersd_metric, BasicLinearModel as MicroModel, BasicLassoSelectorModel as SelectorModel
from scipy import stats
from macro.new_base_trueuse_pods import features, path_pseudo_edges, path_matrix, path_vertices, sources, name_list, param_list
from macro.functional import sd_metric, pv_metric

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

target = 'IVV_aggmean_pct'

savers = numpy.ones(shape=(len(path_vertices),)).astype(dtype=bool)
fg.init_path(path_vertices, path_matrix, path_pseudo_edges, savers)

timeaxis = sources[1].series['DATE'].values[sources[1].series['DATE'].values >= pandas.to_datetime('2007-01-01').isoformat()]

"""
Stage 1: Univariate selection
"""

absolute_threshold = 0.1
significant_share_threshold = 0.5
performer = pv_metric

random_state = 999
n_bootstrap_samples = 100
bootstrap_sample_size_rate = 1.0
alpha = 0.05

# threshold filter

check_results, perf_test_agg = vincent_class_feature_selection_mechanism(
    fg=fg,
    sources=sources,
    features=features,
    target=target,
    timeaxis=timeaxis,
    performer=performer,
    alpha=alpha,
    n_bootstrap_samples=n_bootstrap_samples,
    bootstrap_sample_size_rate=bootstrap_sample_size_rate,
)

check_results_agg = check_results.groupby(by='feature')[['perf', 'perf_test', 'perf_pass']].mean()
check_results_agg = check_results_agg.sort_values(by=['perf_pass', 'perf'], ascending=False)

check_results_agg_median = check_results.groupby(by='feature')[['perf', 'perf_test', 'perf_pass']].median()
check_results_agg_median = check_results_agg_median.sort_values(by=['perf_pass', 'perf'], ascending=False)

# ['RBUSBIS__median1x6_div_pct', 'GS20__pct_shift1', 'FEDFUNDS__pct_shift1', 'GS10__pct_shift1]
# ['RBUSBIS__mean1x3_div_pct', 'USEPUINDXM__predictors_arima_auto', 'USTRADE__std1x12_div_pct', 'MRTSSM44112USN__std1x12_div_pct', 'PCU4841214841212__max3x6_div_pct', 'GS10__p903x6_div_pct', 'PCEDG__std1x12_div_pct']
"""
feature = 'PCEDG__std1x12_div_pct'
checky = check_results[check_results['feature'] == feature]
"""

chosen = check_results_agg.index[
    (check_results_agg['perf_pass'] >= 0.3) &
    (0.5 <= (check_results_agg['perf_test'] / check_results_agg['perf']))].tolist()

chosen = util_prepare_chosen(
    chosen=chosen,
    perf_test_agg=perf_test_agg,
    check_results_agg=check_results_agg,
)

chosen.to_excel('./ab.xlsx')

'''
# conservatism procedure

for fold_n in fg.folds:
    data_train, data_test = fg.fold(sources, features + [target], timeaxis, fold_n=fold_n)
    x_train, y_train = data_train[[x for x in data_train.columns if x != target]].iloc[:-1, :], data_train[target].iloc[1:]
    x_test, y_test = data_test[[x for x in data_test.columns if x != target]].iloc[:-1, :], data_test[target].iloc[1:]

    ...
'''