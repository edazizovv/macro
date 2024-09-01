#


#
import numpy
import pandas


#
from new_base import Path, Projector, Item, FoldGenerator
from new_base_test_projectors import WindowAppGenerator, VincentClassMobsterS
from new_data_check import control, controller_view
from new_base_truegarage import r2_metric, kendalltau_metric, somersd_metric, BasicLinearModel as MicroModel, BasicLassoSelectorModel as SelectorModel
from scipy import stats
from new_base_trueuse_pods import features, path_pseudo_edges, path_matrix, path_vertices, sources, name_list, param_list
from functional import SomersD, pv_metric

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
joint_lag = 12
val_rate = 0.5
overlap_rate = 0.15
fg = FoldGenerator(n_folds=n_folds, joint_lag=joint_lag, val_rate=val_rate, overlap_rate=overlap_rate)

target = 'TLT_aggmean_pct'

fg.init_path(path_vertices, path_matrix, path_pseudo_edges)

timeaxis = sources[1].series['DATE'].values[sources[1].series['DATE'].values >= pandas.to_datetime('2007-01-01').isoformat()]

"""
Stage 1: Univariate thresh cut-off
"""
print("Stage 1")

fold_n = None
# uni_measure = SomersD
uni_measure = pv_metric

drivers, measure_collector = [], []
folds = []
for fold_n in fg.folds:
    data_train, data_test = fg.fold(sources, features + [target], timeaxis, fold_n=fold_n)
    data_train.to_excel('../data/data_folds/data_train_{0}.xlsx'.format(fold_n))
    data_test.to_excel('../data/data_folds/data_test_{0}.xlsx'.format(fold_n))
    x_train, y_train = data_train[[x for x in data_train.columns if x != target]].iloc[:-1, :], data_train[target].iloc[1:]
    x_test, y_test = data_test[[x for x in data_test.columns if x != target]].iloc[:-1, :], data_test[target].iloc[1:]

    for feature in features:
        measured = uni_measure(x=x_train[feature], y=y_train)
        measure_collector.append(measured)
        drivers.append(feature)
        folds.append(fold_n)

uni_thresh = 0.1
same_sign_thresh = 0.9
measured_stats = pandas.DataFrame(data={'fold': folds,
                                        'feature': drivers,
                                        'measured': measure_collector})
measured_stats['measure_sign'] = (measured_stats['measured'] > 0).astype(dtype=int).astype(dtype=float)
measured_stats_agg_mean = measured_stats.groupby(by='feature')[['measure_sign']].mean()
measured_stats_agg_median = measured_stats.groupby(by='feature')[['measured']].median()
measured_stats_agg = measured_stats_agg_mean.merge(right=measured_stats_agg_median, left_index=True, right_index=True, how='outer')
measured_stats_agg['measure_sign'] = measured_stats_agg['measure_sign'].apply(func=lambda x: max(x, 1 - x))
# measured_stats_mask = ((measured_stats_agg['measured'].apply(func=lambda x: numpy.abs(x)) >= uni_thresh) *
#                        (measured_stats_agg['measure_sign'] >= same_sign_thresh))
measured_stats_mask = ((measured_stats_agg['measured'].apply(func=lambda x: x) >= uni_thresh) *
                       (measured_stats_agg['measure_sign'] >= same_sign_thresh))

uni_features = measured_stats_agg[measured_stats_mask].index.values.tolist()
xx = measured_stats_agg[measured_stats_mask]
xx = xx.sort_values(by='measured')

# raise Exception("ghoul?")

"""
Stage 2: Finding LASSO-based skeleton
"""
# '''
print("Stage 2")

sm_kwg = {}
# mm_kwg = {'score': somersd_metric, 'metrics': {'r2_metric': r2_metric, 'kendalltau_metric': kendalltau_metric}}
mm_kwg = {'score': r2_metric, 'metrics': {'somersd_metric': somersd_metric, 'kendalltau_metric': kendalltau_metric}}

total_scores, total_metrics, total_ranks = [], [], []
total_improvements, total_features = [], []
folds = []
for fold_n in fg.folds:
    data_train, data_test = fg.fold(sources, uni_features + [target], timeaxis, fold_n=fold_n)
    x_train, y_train = data_train[[x for x in data_train.columns if x != target]].iloc[:-1, :], data_train[target].iloc[1:]
    x_test, y_test = data_test[[x for x in data_test.columns if x != target]].iloc[:-1, :], data_test[target].iloc[1:]

    sm = SelectorModel(**sm_kwg)
    sm.fit(x=x_train, y=y_train)

    previous_score = 0
    for j in sm.ranking:
        sub_features = sm.ranking_features[:(j + 1)]
        mm = MicroModel(**mm_kwg)
        mm.fit(x=x_train[sub_features], y=y_train)
        # mm_score = numpy.abs(mm.score(x=x_test[sub_features], y=y_test))
        mm_score = mm.score(x=x_test[sub_features], y=y_test)
        mm_metrics = mm.metrics(x=x_test[sub_features], y=y_test)
        if mm_score <= previous_score:
            improvement = numpy.nan
            previous_score = numpy.nan
        else:
            improvement = mm_score - previous_score
            previous_score = mm_score

        total_scores.append(mm_score)
        total_metrics.append(mm_metrics)
        total_improvements.append(improvement)
        total_features.append(sm.ranking_features[j])

        folds.append(fold_n)

    total_ranks += sm.ranking

skeleton_stats = pandas.DataFrame(data={'fold': folds,
                                        'score': total_scores})

total_metrics = [pandas.DataFrame(data={key: [total_metrics[i][key]] for key in total_metrics[i].keys()}) for i in range(len(total_metrics))]
total_metrics = pandas.concat(total_metrics, axis=0, ignore_index=True)
for key in total_metrics.columns:
    skeleton_stats[key] = total_metrics[key].values

skeleton_stats['improvement'] = total_improvements
skeleton_stats['feature'] = total_features
skeleton_stats['selected'] = (~pandas.isna(skeleton_stats['improvement'])).astype(dtype=int)

skeleton_stats_agg = skeleton_stats.groupby(by='feature')[total_metrics.columns.values.tolist() + ['improvement']].median()
skeleton_stats_agg = skeleton_stats_agg.merge(right=skeleton_stats.groupby(by='feature')[['selected']].mean(),
                                              left_index=True, right_index=True, how='outer')

skeleton_stats_agg = skeleton_stats_agg.sort_values(by='selected')
# xxx = measured_stats[measured_stats['feature'] == 'GEPUPPP_pct']

mask_skeleton_filtered = skeleton_stats_agg[(skeleton_stats_agg['selected'] >= 0.5) *
                                            (skeleton_stats_agg['improvement'] >= 0.01)]

mask_skeleton_filtered = mask_skeleton_filtered.sort_values(by='improvement')
skeleton_top_n = 30
skeleton_features = mask_skeleton_filtered.index.values[:skeleton_top_n].tolist()

raise Exception("ghoul?")

# '''
"""
Stage 3: Finding complement 
"""
# '''

sm_kwg = {}
mm_kwg = {'score': somersd_metric, 'metrics': {'r2_metric': r2_metric, 'kendalltau_metric': kendalltau_metric}}

cmp_alpha_thresh = 0.05

left_features = [x for x in uni_features if x not in skeleton_features]
cmp_selection = list(skeleton_features)

while True:

    print("iterating cmp...")

    base_scores, candidate_scores, candidate_features = [], [], []
    folds = []
    for fold_n in fg.folds:
        data_train, data_test = fg.fold(sources, uni_features + [target], timeaxis, fold_n=fold_n)
        x_train, y_train = data_train[[x for x in data_train.columns if x != target]].iloc[:-1, :], data_train[target].iloc[1:]
        x_test, y_test = data_test[[x for x in data_test.columns if x != target]].iloc[:-1, :], data_test[target].iloc[1:]

        x_train_sk_features = x_train[cmp_selection]
        x_test_sk_features = x_test[cmp_selection]

        mm = MicroModel(**mm_kwg)
        mm.fit(x=x_train_sk_features, y=y_train)
        mm_score = mm.score(x=x_test_sk_features, y=y_test)
        base_scores += [mm_score] * len(left_features)

        for feature in left_features:

            x_train_ext_features = x_train[cmp_selection + [feature]]
            x_test_ext_features = x_test[cmp_selection + [feature]]

            mm = MicroModel(**mm_kwg)
            mm.fit(x=x_train_ext_features, y=y_train)
            mm_score = mm.score(x=x_test_ext_features, y=y_test)
            candidate_scores.append(mm_score)
            candidate_features.append(feature)
            folds.append(fold_n)

    complement_stats = pandas.DataFrame(data={'fold': folds,
                                              'base_score': base_scores,
                                              'candidate_score': candidate_scores,
                                              'feature': candidate_features})

    def charger(x):
        # https://www.statisticssolutions.com/free-resources/directory-of-statistical-analyses/paired-sample-t-test/
        tested = stats.ttest_rel(x['base_score'].values,
                                 x['candidate_score'].values,
                                 alternative='less')
        return tested.pval


    # https://www.statisticssolutions.com/free-resources/directory-of-statistical-analyses/paired-sample-t-test/
    complement_stats['candidate_diff'] = complement_stats['candidate_score'] - complement_stats['base_score']
    complement_stats_agg_part = complement_stats.groupby(by='feature')
    complement_stats_agg_mean = complement_stats_agg_part[['candidate_diff']].mean().rename(columns={'candidate_diff': 'mean'})
    complement_stats_agg_std = complement_stats_agg_part[['candidate_diff']].std().rename(columns={'candidate_diff': 'std'})
    complement_stats_agg_count = complement_stats_agg_part[['candidate_diff']].count().rename(columns={'candidate_diff': 'n'})
    complement_stats_agg = complement_stats_agg_mean.merge(right=complement_stats_agg_std, left_index=True, right_index=True, how='outer')
    complement_stats_agg = complement_stats_agg.merge(right=complement_stats_agg_count, left_index=True, right_index=True, how='outer')

    # yy = complement_stats[complement_stats['feature'] == 'DFXARC1M027SBEA_median6_pct']

    def tester(x):
        arg = x['mean'] / (x['std'] / (x['n'] ** 0.5))
        pv = 1 - stats.t.cdf(x=arg, df=x['n'] - 1)
        return pv

    complement_stats_agg['test_result'] = complement_stats_agg.apply(func=tester, axis=1)
    complement_stats_agg = complement_stats_agg.sort_values(by='test_result')

    complement_stats_filtered = complement_stats_agg[complement_stats_agg['test_result'] <= cmp_alpha_thresh].copy()
    if complement_stats_filtered.shape[0] == 0:
        break
    else:
        if len(left_features) == 1:
            break
        else:
            complement_stats_filtered = complement_stats_filtered.sort_values(by='mean')
            new_feature = complement_stats_filtered.index.values[0]
            cmp_selection.append(new_feature)
            left_features = [x for x in left_features if x != new_feature]
            print("next iteration")
# '''

"""
Stage 4: Final validation
"""
# '''

sm_kwg = {}
mm_kwg = {'score': somersd_metric, 'metrics': {'r2_metric': r2_metric, 'kendalltau_metric': kendalltau_metric}}

print("Final run")

base_scores = []
coeffs = []
folds = []
train_points_global, test_points_global = {}, {}
for fold_n in fg.folds:
    data_train, data_test = fg.fold(sources, uni_features + [target], timeaxis, fold_n=fold_n)
    x_train, y_train = data_train[[x for x in data_train.columns if x != target]].iloc[:-1, :], data_train[target].iloc[1:]
    x_test, y_test = data_test[[x for x in data_test.columns if x != target]].iloc[:-1, :], data_test[target].iloc[1:]

    x_train_sk_features = x_train[cmp_selection]
    x_test_sk_features = x_test[cmp_selection]

    mm = MicroModel(**mm_kwg)
    mm.fit(x=x_train_sk_features, y=y_train)
    mm_score = mm.score(x=x_test_sk_features, y=y_test)
    base_scores.append(mm_score)
    folds.append(fold_n)

    coeffs.append([mm.model.intercept_] + mm.model.coef_.tolist())

    train_points = pandas.DataFrame(data={'y': y_train, 'y_hat': mm.predict(x=x_train_sk_features)})
    test_points = pandas.DataFrame(data={'y': y_test, 'y_hat': mm.predict(x=x_test_sk_features)})

    train_points_global[fold_n] = train_points
    test_points_global[fold_n] = test_points

final_stats = pandas.DataFrame(data=coeffs, columns=['const'] + cmp_selection)
final_stats['fold'] = folds
final_stats['base_score'] = base_scores
for c in (['const'] + cmp_selection):
    final_stats[c] = final_stats[c] / final_stats[c].std()

"""
from matplotlib import pyplot
fig, ax = pyplot.subplots()
final_stats[['const'] + cmp_selection].plot(ax=ax)
final_stats['base_score'].plot(ax=ax, secondary_y=True)
ax.legend(ax.get_lines() + ax.right_ax.get_lines(),\
           ['const'] + cmp_selection + ['base_score'])
fig.savefig('xx.png')
"""

"""
from matplotlib import pyplot
fig, ax = pyplot.subplots()
k = 0
train_points_global[k].plot(ax=ax)
fig.savefig('xx_train.png')
del fig
del ax
fig, ax = pyplot.subplots()
test_points_global[k].plot(ax=ax)
fig.savefig('xx_test.png')
"""

# xxx = measured_stats[numpy.isin(measured_stats['feature'].values, ['const'] + cmp_selection)]
# xxx = xxx.sort_values(by=['feature', 'fold'])

# '''
