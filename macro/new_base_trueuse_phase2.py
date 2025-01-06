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

#
from macro.new_base import Path, Projector, Item, FoldGenerator
from macro.new_base_test_projectors import WindowAppGenerator, VincentClassMobsterS
from macro.new_data_check import control, controller_view
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

target = 'TLT_aggmean_pct'

savers = numpy.ones(shape=(len(path_vertices),)).astype(dtype=bool)
fg.init_path(path_vertices, path_matrix, path_pseudo_edges, savers)

timeaxis = sources[1].series['DATE'].values[sources[1].series['DATE'].values >= pandas.to_datetime('2007-01-01').isoformat()]

"""
Stage 2.1: LASSO-based skeleton
"""
"""
performer = pv_metric
alpha = 0.05
n_bootstrap_samples = 1000
bootstrap_sample_size_rate = 1.0

# threshold filter

summary_fold_n = []
summary_feature = []
summary_alpha_enters = []
summary_perf_increase = []
summary_perf = []
summary_perf_test = []
summary_in_selection = []
summary_perf_univ = []
summary_perf_univ_test = []
xxx = []
for fold_n in fg.folds:
    print(fold_n)
    data_train, data_test = fg.fold(sources, features + [target], timeaxis, fold_n=fold_n)
    x_train, y_train = data_train[[x for x in data_train.columns if x != target]].iloc[:-1, :], data_train[target].iloc[1:]
    x_test, y_test = data_test[[x for x in data_test.columns if x != target]].iloc[:-1, :], data_test[target].iloc[1:]

    # we need to standardize before lasso

    sc = StandardScaler()
    x_train = pandas.DataFrame(
        data=sc.fit_transform(X=x_train.values),
        index=x_train.index,
        columns=x_train.columns,
    )
    x_test = pandas.DataFrame(
        data=sc.transform(X=x_test.values),
        index=x_test.index,
        columns=x_test.columns,
    )

    # lasso

    num = 100
    # alpha_start = 0.02
    # alpha_step = 0.99
    alpha_start = 0.02
    alpha_step = 0.95
    alphas = numpy.array([alpha_start * (alpha_step ** k) for k in range(num)])
    alphas_lasso, coefs_lasso, _ = lasso_path(
        X=x_train.values,
        y=y_train.values,
        alphas=alphas,
    )

    xx = pandas.DataFrame(data=coefs_lasso.T, index=alphas_lasso, columns=features)
    xx['n_distinct'] = xx.apply(func=lambda x: (x != 0).sum(), axis=1)
    xx.index.name = 'alphas_lasso'
    xx = xx.reset_index()
    xx_distinct = xx.groupby(by='n_distinct').first()
    xx_distinct = xx_distinct.reset_index().set_index('alphas_lasso')
    xx_distinct = xx_distinct[xx_distinct['n_distinct'] > 0].copy()
    if xx_distinct.shape[0] == 0:
        raise Exception("No entries in the path found, consider changing the alphas")
    xx_distinct['features_selected'] = xx_distinct.apply(func=lambda x: numpy.array(features)[x[features] != 0], axis=1)
    xx_distinct['perf_train'] = numpy.nan
    xx_distinct['perf_test'] = numpy.nan
    xx_distinct['perf_ci_lower_train'] = numpy.nan
    xx_distinct['perf_ci_upper_train'] = numpy.nan
    xx_distinct['perf_ci_lower_test'] = numpy.nan
    xx_distinct['perf_ci_upper_test'] = numpy.nan
    for j in range(xx_distinct.shape[0]):

        x = xx_distinct.iloc[j, :]
        x_ix = xx_distinct.index[j]

        jth_model = LinearRegression()
        jth_model.fit(X=x_train[x['features_selected']].values, y=y_train.values)
        y_train_hat = jth_model.predict(X=x_train[x['features_selected']].values)
        y_test_hat = jth_model.predict(X=x_test[x['features_selected']].values)

        perf_train = performer(x=y_train_hat, y=y_train.values)
        perf_test = performer(x=y_test_hat, y=y_test.values)
        xx_distinct.loc[x_ix, 'perf_train'] = perf_train
        xx_distinct.loc[x_ix, 'perf_test'] = perf_test

        '''
        n_train, k_train = y_train.shape[0], x['n_distinct'] + 1
        # f_stat_train = (perf_train / (1 - perf_train)) / ((n_train - k_train) / (k_train - 1))
        f_stat_train = (perf_train / (1 - perf_train)) / (k_train / (n_train - k_train - 1))
        # f_stat follows non-central F distribution
        betas = numpy.array([jth_model.intercept_] + jth_model.coef_.tolist())
        e = numpy.ones(shape=(n_train, 1))
        x_local_train = numpy.concatenate(
            (
                e,
                x_train[x['features_selected']].values
            ),
            axis=1,
        )
        m = betas.shape[0]
        # noncentrality_parameter_train = (betas.reshape(1, -1) @ x_local_train.T @ (numpy.identity(n=n_train) - (1 / n_train) * (e @ e.T)) @ x_local_train @ betas.reshape(-1, 1))[0, 0]
        error = y_train.values - y_train_hat
        beta_var = numpy.linalg.inv((x_local_train.T @ x_local_train))
        vars = numpy.array([beta_var[k, k] for k in range(beta_var.shape[0])])
        noncentrality_parameter_train = sum([(n_train - 1) * vars[t] * (betas[t] ** 2) for t in range(m)]) / error.var()
        # NOTE: probably we're missing standard errors of the coeffs & (n - 1) in the numerator as in that paper,
        #       yet I'd rather keep it as this one because I presume adding those terms would increase the confidence
        #       intervals; consequently, less features would be selected, that is an unwanted scenario as we are
        #       already quite conservative
        dist = stats.ncf(dfn=m, dfd=(n_train - m - 1), nc=noncentrality_parameter_train)
        f_stat_ci_lower_train = dist.ppf(q=(alpha / 2))
        f_stat_ci_upper_train = dist.ppf(q=(1 - (alpha / 2)))

        tss = (y_train.values.reshape(-1, 1).T @ (numpy.identity(n=n_train) - (1 / n_train) * (e @ e.T)) @ y_train.values.reshape(-1, 1))[0, 0]
        h = x_local_train @ numpy.linalg.inv(x_local_train.T @ x_local_train) @ x_local_train.T
        rss = (y_train.values.reshape(-1, 1).T @ (numpy.identity(n=n_train) - h) @ y_train.values.reshape(-1, 1))[0, 0]
        f_ = ((tss - rss) / rss) / (m / (n_train - m - 1))
        r2_ = 1 - (rss / tss)

        pp_ = f_stat_train / (f_stat_train + ((n_train - k_train - 1) / k_train))
        perf_ci_lower_train = f_stat_ci_lower_train / (f_stat_ci_lower_train + ((n_train - k_train - 1) / k_train))
        perf_ci_upper_train = f_stat_ci_upper_train / (f_stat_ci_upper_train + ((n_train - k_train - 1) / k_train))
        '''

        ix_range = numpy.array(range(x_train.shape[0]))
        perf_boostrapped = []
        for i in range(n_bootstrap_samples):
            sub_ix_range = numpy.random.choice(
                a=ix_range,
                size=int(x_train.shape[0] * bootstrap_sample_size_rate),
                replace=True,
            )
            x_mask = x_train.index[sub_ix_range]
            y_mask = y_train.index[sub_ix_range]
            jth_model_boostrap = LinearRegression()
            jth_model_boostrap.fit(X=x_train.loc[x_mask, x['features_selected']].values, y=y_train.loc[y_mask].values)
            y_train_hat_boostrap = jth_model.predict(X=x_train.loc[x_mask, x['features_selected']].values)

            perf_train_boostrap = performer(x=y_train_hat_boostrap, y=y_train.loc[y_mask].values)
            perf_boostrapped.append(perf_train_boostrap)

        perf_ci_lower_train = numpy.quantile(
            a=perf_boostrapped,
            q=(alpha / 2),
        )
        perf_ci_upper_train = numpy.quantile(
            a=perf_boostrapped,
            q=(1 - (alpha / 2)),
        )

        xx_distinct.loc[x_ix, 'perf_ci_lower_train'] = perf_ci_lower_train
        xx_distinct.loc[x_ix, 'perf_ci_upper_train'] = perf_ci_upper_train

    xx_distinct['max_perf'] = xx_distinct['perf_train'].max()
    mask = xx_distinct['max_perf'] <= xx_distinct['perf_ci_upper_train']
    if mask.sum() == 0:
        tightest_alpha_cover = xx_distinct.index.min()
    else:
        tightest_alpha_cover = xx_distinct[mask].index.max()
    mask_below = xx_distinct.index < tightest_alpha_cover
    xx_distinct.loc[mask_below, 'cover'] = 0
    xx_distinct.loc[~mask_below, 'cover'] = 1

    for feature in features:
        mask_enters = xx_distinct[feature] != 0
        if mask_enters.sum() == 0:
            alpha_enters = numpy.nan
            feature_perf_increase = numpy.nan
            feature_entered_perf = numpy.nan
            feature_entered_perf_test = numpy.nan
            feature_covered = 0
            feature_perf_univariate = numpy.nan
            feature_perf_univariate_test = numpy.nan
        else:
            alpha_enters = xx_distinct.loc[mask_enters, feature].index.max()
            alpha_ix = xx_distinct.index.tolist().index(alpha_enters)
            feature_entered_perf = xx_distinct.iloc[alpha_ix, :]['perf_train']
            feature_entered_perf_test = xx_distinct.iloc[alpha_ix, :]['perf_test']
            if alpha_ix == 0:
                feature_before_perf = 0
            else:
                feature_before_perf = xx_distinct.iloc[alpha_ix - 1, :]['perf_train']
            feature_perf_increase = feature_entered_perf - feature_before_perf
            feature_covered = xx_distinct.iloc[alpha_ix, :]['cover']
            feature_perf_univariate = performer(x=x_train[feature].values, y=y_train.values)
            feature_perf_univariate_test = performer(x=x_test[feature].values, y=y_test.values)

        summary_fold_n.append(fold_n)
        summary_feature.append(feature)
        summary_alpha_enters.append(alpha_enters)
        summary_perf_increase.append(feature_perf_increase)
        summary_perf.append(feature_entered_perf)
        summary_perf_test.append(feature_entered_perf_test)
        summary_perf_univ.append(feature_perf_univariate)
        summary_perf_univ_test.append(feature_perf_univariate_test)
        summary_in_selection.append(feature_covered)

    xxx.append(xx_distinct)


summary = pandas.DataFrame(data={'fold_n': summary_fold_n,
                                 'feature': summary_feature,
                                 'alpha_enters': summary_alpha_enters,
                                 'perf_increase': summary_perf_increase,
                                 'perf': summary_perf,
                                 'perf_test': summary_perf_test,
                                 'perf_univ': summary_perf_univ,
                                 'perf_univ_test': summary_perf_univ_test,
                                 'in_selection': summary_in_selection})

summary_agg = summary.groupby(by='feature')[['alpha_enters', 'perf_increase', 'perf', 'perf_test', 'perf_univ', 'perf_univ_test', 'in_selection']].mean()
summary_agg_fiter = summary_agg[(summary_agg['perf_increase'] > 0) & (summary_agg['in_selection'] > 0)]

# features_selected = summary_agg_fiter.index.tolist()
# features_selected = ['GS20__pct_shift1', 'PCU4841214841212__stayer_percentile', 'RBUSBIS__median1x6_div_pct', 'T20YIEM__pct_shift1']
# features_selected = ['MRTSSM44112USN__std1x12_div_pct', 'PCU4841214841212__max3x6_div_pct', 'RBUSBIS__mean1x3_div_pct', 'T10YIEM__max1x3_div_pct', 'USEPUINDXM__predictors_arima_auto', 'USTRADE__std1x12_div_pct']

print('finalize')
final_perfs_train = []
final_perfs_test = []
for fold_n in fg.folds:

    data_train, data_test = fg.fold(sources, features_selected + [target], timeaxis, fold_n=fold_n)
    x_train, y_train = data_train[[x for x in data_train.columns if x != target]].iloc[:-1, :], data_train[target].iloc[1:]
    x_test, y_test = data_test[[x for x in data_test.columns if x != target]].iloc[:-1, :], data_test[target].iloc[1:]

    lm = LinearRegression()
    lm.fit(X=x_train.values, y=y_train.values)
    y_train_hat = lm.predict(X=x_train.values)
    y_test_hat = lm.predict(X=x_test.values)

    perf_train = performer(x=y_train_hat, y=y_train.values)
    perf_test = performer(x=y_test_hat, y=y_test.values)

    final_perfs_train.append(perf_train)
    final_perfs_test.append(perf_test)

final_perf = pandas.DataFrame(
    data={'perf_train': final_perfs_train,
          'perf_test': final_perfs_test,},
    index=fg.folds,
)

final_perf.to_excel('./final_perf_lm.xlsx')
"""

"""
Stage 2.1: Decision tree-based skeleton
"""

"""

performer = pv_metric
dt_kwg = {
    # 'max_depth': 5,
}

final_perfs_train = []
final_perfs_test = []
models = []
significances = {feature: [] for feature in features}
for fold_n in fg.folds:

    data_train, data_test = fg.fold(sources, features + [target], timeaxis, fold_n=fold_n)
    x_train, y_train = data_train[[x for x in data_train.columns if x != target]].iloc[:-1, :], data_train[target].iloc[1:]
    x_test, y_test = data_test[[x for x in data_test.columns if x != target]].iloc[:-1, :], data_test[target].iloc[1:]

    model = DecisionTreeRegressor(**dt_kwg)
    model.fit(X=x_train, y=y_train)
    y_train_hat = model.predict(X=x_train)
    y_test_hat = model.predict(X=x_test)

    perf_train = performer(x=y_train_hat, y=y_train.values)
    perf_test = performer(x=y_test_hat, y=y_test.values)

    final_perfs_train.append(perf_train)
    final_perfs_test.append(perf_test)

    models.append(model)

    for j in range(len(model.feature_names_in_)):
        significances[model.feature_names_in_[j]].append(model.feature_importances_[j])

inter_perf = pandas.DataFrame(
    data={'perf_train': final_perfs_train,
          'perf_test': final_perfs_test,},
    index=fg.folds,
)

significances = pandas.DataFrame(data=significances)
selected_summary = (significances > 0).sum(axis=0)
selected_features = selected_summary.index[selected_summary == 10].tolist()

performer = pv_metric
dt_kwg = {
    # 'max_depth': 5,
}

print('finalize')
final_perfs_train = []
final_perfs_test = []
for fold_n in fg.folds:

    data_train, data_test = fg.fold(sources, selected_features + [target], timeaxis, fold_n=fold_n)
    x_train, y_train = data_train[[x for x in data_train.columns if x != target]].iloc[:-1, :], data_train[target].iloc[1:]
    x_test, y_test = data_test[[x for x in data_test.columns if x != target]].iloc[:-1, :], data_test[target].iloc[1:]

    model = DecisionTreeRegressor(**dt_kwg)
    model.fit(X=x_train, y=y_train)
    y_train_hat = model.predict(X=x_train)
    y_test_hat = model.predict(X=x_test)

    perf_train = performer(x=y_train_hat, y=y_train.values)
    perf_test = performer(x=y_test_hat, y=y_test.values)

    final_perfs_train.append(perf_train)
    final_perfs_test.append(perf_test)

final_perf = pandas.DataFrame(
    data={'perf_train': final_perfs_train,
          'perf_test': final_perfs_test,},
    index=fg.folds,
)

final_perf.to_excel('./final_perf_dt.xlsx')
"""


"""
Stage 2.1: Random forest-based skeleton
"""

"""

performer = pv_metric
rf_kwg = {
    # 'max_depth': 5,
}

final_perfs_train = []
final_perfs_test = []
models = []
significances = {feature: [] for feature in features}
for fold_n in fg.folds:

    data_train, data_test = fg.fold(sources, features + [target], timeaxis, fold_n=fold_n)
    x_train, y_train = data_train[[x for x in data_train.columns if x != target]].iloc[:-1, :], data_train[target].iloc[1:]
    x_test, y_test = data_test[[x for x in data_test.columns if x != target]].iloc[:-1, :], data_test[target].iloc[1:]

    model = RandomForestRegressor(**rf_kwg)
    model.fit(X=x_train, y=y_train)
    y_train_hat = model.predict(X=x_train)
    y_test_hat = model.predict(X=x_test)

    perf_train = performer(x=y_train_hat, y=y_train.values)
    perf_test = performer(x=y_test_hat, y=y_test.values)

    final_perfs_train.append(perf_train)
    final_perfs_test.append(perf_test)

    models.append(model)

    for j in range(len(model.feature_names_in_)):
        significances[model.feature_names_in_[j]].append(model.feature_importances_[j])

inter_perf = pandas.DataFrame(
    data={'perf_train': final_perfs_train,
          'perf_test': final_perfs_test,},
    index=fg.folds,
)

significances = pandas.DataFrame(data=significances)
selected_summary = (significances > 0.01).sum(axis=0)
selected_features = selected_summary.index[selected_summary == 10].tolist()

performer = pv_metric
rf_kwg = {
    # 'max_depth': 5,
}

print('finalize')
final_perfs_train = []
final_perfs_test = []
for fold_n in fg.folds:

    data_train, data_test = fg.fold(sources, selected_features + [target], timeaxis, fold_n=fold_n)
    x_train, y_train = data_train[[x for x in data_train.columns if x != target]].iloc[:-1, :], data_train[target].iloc[1:]
    x_test, y_test = data_test[[x for x in data_test.columns if x != target]].iloc[:-1, :], data_test[target].iloc[1:]

    model = RandomForestRegressor(**rf_kwg)
    model.fit(X=x_train, y=y_train)
    y_train_hat = model.predict(X=x_train)
    y_test_hat = model.predict(X=x_test)

    perf_train = performer(x=y_train_hat, y=y_train.values)
    perf_test = performer(x=y_test_hat, y=y_test.values)

    final_perfs_train.append(perf_train)
    final_perfs_test.append(perf_test)

final_perf = pandas.DataFrame(
    data={'perf_train': final_perfs_train,
          'perf_test': final_perfs_test,},
    index=fg.folds,
)

final_perf.to_excel('./final_perf_rf.xlsx')
"""


"""
Stage 2.1: Nearest neighbors-based skeleton
"""

"""

perfs_u = {}
perfs_w = {}
m = 35

for z in range(m):
    print(z)

    performer = pv_metric


    kn_kwg = {
        'n_neighbors': z + 1,
        'weights': 'uniform'
    }

    final_perfs_train = []
    final_perfs_test = []
    for fold_n in fg.folds:

        data_train, data_test = fg.fold(sources, features + [target], timeaxis, fold_n=fold_n)
        x_train, y_train = data_train[[x for x in data_train.columns if x != target]].iloc[:-1, :], data_train[target].iloc[1:]
        x_test, y_test = data_test[[x for x in data_test.columns if x != target]].iloc[:-1, :], data_test[target].iloc[1:]

        sc = StandardScaler()
        x_train = sc.fit_transform(X=x_train.values)
        x_test = sc.transform(X=x_test.values)

        model = KNeighborsRegressor(**kn_kwg)
        model.fit(X=x_train, y=y_train)
        y_train_hat = model.predict(X=x_train)
        y_test_hat = model.predict(X=x_test)

        perf_train = performer(x=y_train_hat, y=y_train.values)
        perf_test = performer(x=y_test_hat, y=y_test.values)

        final_perfs_train.append(perf_train)
        final_perfs_test.append(perf_test)

    inter_perf = pandas.DataFrame(
        data={'perf_train': final_perfs_train,
              'perf_test': final_perfs_test,},
        index=fg.folds,
    )

    perfs_u[z] = inter_perf


    kn_kwg = {
        'n_neighbors': z + 1,
        'weights': 'distance'
    }

    final_perfs_train = []
    final_perfs_test = []
    for fold_n in fg.folds:

        data_train, data_test = fg.fold(sources, features + [target], timeaxis, fold_n=fold_n)
        x_train, y_train = data_train[[x for x in data_train.columns if x != target]].iloc[:-1, :], data_train[target].iloc[1:]
        x_test, y_test = data_test[[x for x in data_test.columns if x != target]].iloc[:-1, :], data_test[target].iloc[1:]

        sc = StandardScaler()
        x_train = sc.fit_transform(X=x_train.values)
        x_test = sc.transform(X=x_test.values)

        model = KNeighborsRegressor(**kn_kwg)
        model.fit(X=x_train, y=y_train)
        y_train_hat = model.predict(X=x_train)
        y_test_hat = model.predict(X=x_test)

        perf_train = performer(x=y_train_hat, y=y_train.values)
        perf_test = performer(x=y_test_hat, y=y_test.values)

        final_perfs_train.append(perf_train)
        final_perfs_test.append(perf_test)

    inter_perf = pandas.DataFrame(
        data={'perf_train': final_perfs_train,
              'perf_test': final_perfs_test,},
        index=fg.folds,
    )

    perfs_w[z] = inter_perf

mm = 10
fig, ax = pyplot.subplots(fg.n_folds, 2)
for fold_n in fg.folds:
    perf_u_series = [perfs_u[j].loc[fold_n, 'perf_test'] for j in perfs_u.keys() if j < mm]
    perf_w_series = [perfs_w[j].loc[fold_n, 'perf_test'] for j in perfs_w.keys() if j < mm]
    pandas.Series(perf_u_series).plot(ax=ax[fold_n, 0])
    pandas.Series(perf_w_series).plot(ax=ax[fold_n, 1])
fig.savefig('z.svg')

final_n_neighbors = 4

kn_kwg = {
    'n_neighbors': final_n_neighbors + 1,
    'weights': 'distance'
}

final_perfs_train = []
final_perfs_test = []
for fold_n in fg.folds:

    data_train, data_test = fg.fold(sources, features + [target], timeaxis, fold_n=fold_n)
    x_train, y_train = data_train[[x for x in data_train.columns if x != target]].iloc[:-1, :], data_train[target].iloc[1:]
    x_test, y_test = data_test[[x for x in data_test.columns if x != target]].iloc[:-1, :], data_test[target].iloc[1:]

    sc = StandardScaler()
    x_train = sc.fit_transform(X=x_train.values)
    x_test = sc.transform(X=x_test.values)

    model = KNeighborsRegressor(**kn_kwg)
    model.fit(X=x_train, y=y_train)
    y_train_hat = model.predict(X=x_train)
    y_test_hat = model.predict(X=x_test)

    perf_train = performer(x=y_train_hat, y=y_train.values)
    perf_test = performer(x=y_test_hat, y=y_test.values)

    final_perfs_train.append(perf_train)
    final_perfs_test.append(perf_test)

final_perf = pandas.DataFrame(
    data={'perf_train': final_perfs_train,
          'perf_test': final_perfs_test,},
    index=fg.folds,
)

final_perf.to_excel('./final_perf_kn.xlsx')
"""
