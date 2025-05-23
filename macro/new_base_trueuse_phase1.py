#


#
import numpy
import pandas
from scipy.stats import multivariate_normal, norm
from sklearn.linear_model import LinearRegression


#
from macro.new_base import Path, Projector, Item, FoldGenerator
from macro.new_base_test_projectors import WindowAppGenerator
from macro.new_data_check import pod_loader, controller_view
from functional import pa_metric, r2_metric
from macro.new_base_truegarage import r2_metric, kendalltau_metric, somersd_metric, BasicLinearModel as MicroModel, BasicLassoSelectorModel as SelectorModel
from scipy import stats
# from macro.new_base_trueuse_pods import features, path_pseudo_edges, path_matrix, path_vertices, sources, name_list, param_list
from macro.functional import sd_metric, pv_metric
from raise_fold_generator import fg, start_date, end_date
from new_base_trueuse_phase1_garage import vincent_class_feature_selection_mechanism, util_prepare_chosen
from macro.graph_loader import ElementLoader

#
target = 'IVV_aggmean_pct'

# timeaxis = sources[1].series['DATE'].values[sources[1].series['DATE'].values >= pandas.to_datetime(start_date).isoformat()]

loader = ElementLoader(vector_path='../data/data_meta/vector.xlsx')

save_features = numpy.ones(shape=(len(loader.path_vertices),)).astype(dtype=bool)
fg.init_path(loader.path_vertices, loader.path_matrix, loader.path_pseudo_edges, save_features)

date_range, suggested_lag = fg.find_lags(sources=loader.sources, features=loader.features, target=target)
# fg.set_lag_from_delta(lag_delta=suggested_lag, timeaxis=date_range)

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

x_factors = [x for x in loader.features if x != target]

# threshold filter

check_results, perf_test_agg = vincent_class_feature_selection_mechanism(
    fg=fg,
    sources=loader.sources,
    predictors=x_factors,
    target=target,
    timeaxis=date_range,
    performer=performer,
    alpha=alpha,
    n_bootstrap_samples=n_bootstrap_samples,
    bootstrap_sample_size_rate=bootstrap_sample_size_rate,
)

check_results.to_excel("../data/data_folds/perf_results_ph1.xlsx")

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
    (check_results_agg['perf_pass'] >= 0.2) &
    (0.5 <= (check_results_agg['perf_test'] / check_results_agg['perf']))].tolist()

chosen = util_prepare_chosen(
    chosen=chosen,
    perf_test_agg=perf_test_agg,
    check_results_agg=check_results_agg,
)

chosen.to_excel('./phase_1_chosen.xlsx')

'''
# conservatism procedure

for fold_n in fg.folds:
    data_train, data_test = fg.fold(sources, features + [target], timeaxis, fold_n=fold_n)
    x_train, y_train = data_train[[x for x in data_train.columns if x != target]].iloc[:-1, :], data_train[target].iloc[1:]
    x_test, y_test = data_test[[x for x in data_test.columns if x != target]].iloc[:-1, :], data_test[target].iloc[1:]

    ...
'''