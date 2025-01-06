"""
MULTICOLLINEARITY DROPPED

No evidence found in favour of reducing multicollinearity by the means of excluding correlated variables.
Any of these should be handled manually after the conservative selection procedure.

Below there are some commented sections which stood for multicollinearity topic yet have been rejected.

At the moment, the module consists only of some basic cut-offs aimed at reducing number of potential drivers
to be considered.
"""
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
This is three-phase build-up:
    1. Univariate thresh cut-off
    2. Finding LASSO-based skeleton
    3. Finding complement 
"""

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

The idea:

We define a reasonable pair of thresholds: 
    > absolute threshold for average absolute performance (over folds)
    > share of significant point estimates of the performance (over folds)
Those which surpass the thresholds shall be passed into a conservatism procedure with multicollinearity considered.
There may be two options for the multicollinearity measurement:
    > correlation matrix (pairwise correlations) with a given measure (standard case: take the performer)
    > VIF-like measure (group correlations) with a given measure (standard case: performer-based)

Conservatism procedure is performed so that feature sets produced are passed into a simple linear regression 
which performance is estimated. Dynamics of the performance is considered over the tightening series of the procedure,
and the minimum tight is chosen.
"""
print("Stage 1")

absolute_threshold = 0.1
significant_share_threshold = 0.5
performer = pv_metric

random_state = 999
n_bootstrap_samples = 100
bootstrap_sample_size_rate = 1.0
alpha = 0.05

# threshold filter

check_results = []
perf_test = []
for fold_n in fg.folds:
    data_train, data_test = fg.fold(sources, features + [target], timeaxis, fold_n=fold_n)
    x_train, y_train = data_train[[x for x in data_train.columns if x != target]].iloc[:-1, :], data_train[target].iloc[1:]
    x_test, y_test = data_test[[x for x in data_test.columns if x != target]].iloc[:-1, :], data_test[target].iloc[1:]

    # calculate significances with bootstrap

    marginal_empirical_distributions = numpy.concatenate(
        (
            x_train.values,
            y_train.values.reshape(-1, 1),
        ),
        axis=1,
    )
    joint_cov = pandas.DataFrame(marginal_empirical_distributions).cov()

    joint_cov_zero = joint_cov.copy()
    target_self_var = joint_cov_zero.iloc[-1, -1]
    joint_cov_zero.iloc[-1, :] = 0
    joint_cov_zero.iloc[:, -1] = 0
    joint_cov_zero.iloc[-1, -1] = target_self_var

    joint_names = x_train.columns.tolist() + [y_train.name]
    n = x_train.shape[0]
    m = joint_cov.shape[0]
    zero_means = numpy.zeros(shape=m)
    generator_normal = multivariate_normal(
        mean=zero_means,
        cov=joint_cov_zero,
        allow_singular=True,
    )

    perf_boostrapped_summary = []
    for i in range(n_bootstrap_samples):
        sample_bootstrapped = {}
        normal_sample = generator_normal.rvs(size=int(n * bootstrap_sample_size_rate))
        for k in range(m):
            scale = joint_cov_zero.iloc[k, k]
            if scale != 0:
                univariate_normal = norm(loc=0, scale=scale)
                q_values = univariate_normal.cdf(x=normal_sample[:, k])
                x_values = [
                    numpy.quantile(
                        a=marginal_empirical_distributions[:, k],
                        q=q,
                    )
                    for q in q_values
                ]
                x_values = numpy.array(x_values)
            else:
                x_values = numpy.zeros(shape=(marginal_empirical_distributions.shape[0],))
            sample_bootstrapped[joint_names[k]] = x_values
        sample_bootstrapped = pandas.DataFrame(sample_bootstrapped)
        perf_boostrap = [performer(x=sample_bootstrapped.iloc[:, k].values, y=sample_bootstrapped.iloc[:, m - 1].values)
                         for k in range(m - 1)]
        perf_boostrapped_summary.append(perf_boostrap)
    perf_boostrapped_summary = pandas.DataFrame(
        data=perf_boostrapped_summary,
        columns=features,
        index=list(range(n_bootstrap_samples)),
    )

    for feature in features:

        ci_thresh_lower = numpy.quantile(
            a=perf_boostrapped_summary[feature].values,
            q=(alpha / 2),
        )
        ci_thresh_upper = numpy.quantile(
            a=perf_boostrapped_summary[feature].values,
            q=(1 - (alpha / 2)),
        )

        perf = performer(x=x_train[feature].values, y=y_train.values)
        perf_ts = performer(x=x_test[feature].values, y=y_test.values)

        perf_pass = not ((ci_thresh_lower <= perf) and (perf <= ci_thresh_upper))

        check_results.append(
            [
                fold_n,
                feature,
                perf,
                perf_ts,
                ci_thresh_lower,
                ci_thresh_upper,
                perf_pass,
            ]
        )

    perf_test_fold = [
        performer(x=x_test[feature].values, y=y_test.values)
        for feature in features
    ]
    perf_test.append(perf_test_fold)

check_results = pandas.DataFrame(
    data=check_results,
    columns=['fold_n', 'feature', 'perf', 'perf_test', 'ci_thresh_lower', 'ci_thresh_upper', 'perf_pass'],
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

perf_test = pandas.DataFrame(data=perf_test, columns=features)
perf_test_agg = perf_test.mean(axis=0).T
perf_test_agg = pandas.DataFrame(data=perf_test_agg, columns=['perf_test'])
perf_test_agg = perf_test_agg.reset_index()
perf_test_agg = perf_test_agg.rename(columns={'index': 'feature'})

chosen = pandas.DataFrame(data={'feature': chosen})

chosen = chosen.merge(
    right=perf_test_agg,
    left_on='feature',
    right_on='feature',
    how='left',
)

chosen = chosen.merge(
    right=check_results_agg.reset_index(),
    left_on='feature',
    right_on='feature',
    how='left',
)

from macro.new_base_trueuse_pods import vxl

chosen['sources'] = chosen['feature'].apply(func=lambda x: x.split('__')[0])
chosen['transforms'] = chosen['feature'].apply(func=lambda x: x.split('__')[1])
chosen = chosen.merge(
    right=vxl,
    left_on=['transforms'],
    right_on=['transform_name'],
    how='left',
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