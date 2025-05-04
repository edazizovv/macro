#


#
import numpy
import pandas
from matplotlib import pyplot
from sklearn.linear_model import lasso_path, LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

#


#
def lm_proxy(
    fg,
    sources,
    features,
    target,
    timeaxis,
    performer,
    parameters,
):

    signif_alpha = parameters['signif_alpha']
    n_bootstrap_samples = parameters['n_bootstrap_samples']
    bootstrap_sample_size_rate = parameters['bootstrap_sample_size_rate']
    perf_incr_threshold = parameters['perf_incr_threshold']
    in_selection_thresh = parameters['in_selection_thresh']
    pperf = parameters['pperf']

    summary_fold_n = []
    summary_feature = []
    summary_alpha_enters = []
    summary_perf_increase = []
    summary_perf = []
    summary_perf_test = []
    summary_in_selection = []
    summary_perf_univ = []
    summary_perf_univ_test = []
    inter_perf = []
    for fold_n in fg.folds:
        print(fold_n)
        data_train, data_test = fg.fold(sources, features + [target], timeaxis, fold_n=fold_n)
        x_train, y_train = data_train[[x for x in data_train.columns if x != target]].iloc[:-1, :], data_train[
                                                                                                        target].iloc[1:]
        x_test, y_test = data_test[[x for x in data_test.columns if x != target]].iloc[:-1, :], data_test[target].iloc[
                                                                                                1:]
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

        lasso_summary = pandas.DataFrame(data=coefs_lasso.T, index=alphas_lasso, columns=features)
        lasso_summary['n_distinct'] = lasso_summary.apply(func=lambda x: (x != 0).sum(), axis=1)
        lasso_summary.index.name = 'alphas_lasso'
        lasso_summary = lasso_summary.reset_index()
        lasso_summary_distinct = lasso_summary.groupby(by='n_distinct').first()
        lasso_summary_distinct = lasso_summary_distinct.reset_index().set_index('alphas_lasso')
        lasso_summary_distinct = lasso_summary_distinct[lasso_summary_distinct['n_distinct'] > 0].copy()
        if lasso_summary_distinct.shape[0] == 0:
            raise Exception("No entries in the path found, consider changing the alphas")
        lasso_summary_distinct['selected_features'] = lasso_summary_distinct.apply(
            func=lambda x: numpy.array(features)[x[features] != 0], axis=1)
        lasso_summary_distinct['perf_train'] = numpy.nan
        lasso_summary_distinct['perf_test'] = numpy.nan
        lasso_summary_distinct['perf_impr_train_value'] = numpy.nan
        lasso_summary_distinct['perf_impr_test_value'] = numpy.nan
        lasso_summary_distinct['perf_impr_train_significance'] = numpy.nan

        lasso_summary_distinct = pperf(
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            lasso_summary_distinct=lasso_summary_distinct,
            performer=performer,
            n_bootstrap_samples=n_bootstrap_samples,
            bootstrap_sample_size_rate=bootstrap_sample_size_rate,
            signif_alpha=signif_alpha,
        )

        for feature in features:
            mask_enters = lasso_summary_distinct[feature] != 0
            if mask_enters.sum() == 0:
                alpha_enters = numpy.nan
                feature_perf_increase = numpy.nan
                feature_entered_perf = numpy.nan
                feature_entered_perf_test = numpy.nan
                feature_covered = 0
                feature_perf_univariate = numpy.nan
                feature_perf_univariate_test = numpy.nan
            else:
                alpha_enters = lasso_summary_distinct.loc[mask_enters, feature].index.max()
                alpha_ix = lasso_summary_distinct.index.tolist().index(alpha_enters)
                feature_entered_perf = lasso_summary_distinct.iloc[alpha_ix, :]['perf_train']
                feature_entered_perf_test = lasso_summary_distinct.iloc[alpha_ix, :]['perf_test']
                if alpha_ix == 0:
                    feature_before_perf = 0
                else:
                    feature_before_perf = lasso_summary_distinct.iloc[alpha_ix - 1, :]['perf_train']
                feature_perf_increase = feature_entered_perf - feature_before_perf
                feature_covered = lasso_summary_distinct.iloc[alpha_ix, :]['cover']
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

        inter_perf.append(lasso_summary_distinct)

    summary = pandas.DataFrame(data={'fold_n': summary_fold_n,
                                     'feature': summary_feature,
                                     'alpha_enters': summary_alpha_enters,
                                     'perf_increase': summary_perf_increase,
                                     'perf': summary_perf,
                                     'perf_test': summary_perf_test,
                                     'perf_univ': summary_perf_univ,
                                     'perf_univ_test': summary_perf_univ_test,
                                     'in_selection': summary_in_selection})

    summary_agg = summary.groupby(by='feature')[
        ['alpha_enters', 'perf_increase', 'perf', 'perf_test', 'perf_univ', 'perf_univ_test', 'in_selection']].mean()
    summary_agg_fiter = summary_agg[
        (summary_agg['perf_increase'] > perf_incr_threshold) & (summary_agg['in_selection'] > in_selection_thresh)]

    selected_features, selected_summary = summary_agg_fiter.index.tolist(), summary_agg

    # selected_features = summary_agg_fiter.index.tolist()
    # selected_features = ['GS20__pct_shift1', 'PCU4841214841212__stayer_percentile', 'RBUSBIS__median1x6_div_pct', 'T20YIEM__pct_shift1']
    # selected_features = ['MRTSSM44112USN__std1x12_div_pct', 'PCU4841214841212__max3x6_div_pct', 'RBUSBIS__mean1x3_div_pct', 'T10YIEM__max1x3_div_pct', 'USEPUINDXM__predictors_arima_auto', 'USTRADE__std1x12_div_pct']

    if len(selected_features) == 0:
        raise Exception("no features selected :(")

    print('finalize')
    final_perfs_train = []
    final_perfs_test = []
    for fold_n in fg.folds:
        data_train, data_test = fg.fold(sources, selected_features + [target], timeaxis, fold_n=fold_n)
        x_train, y_train = data_train[[x for x in data_train.columns if x != target]].iloc[:-1, :], data_train[
                                                                                                        target].iloc[1:]
        x_test, y_test = data_test[[x for x in data_test.columns if x != target]].iloc[:-1, :], data_test[target].iloc[
                                                                                                1:]
        lm = LinearRegression()
        lm.fit(X=x_train.values, y=y_train.values)
        y_train_hat = lm.predict(X=x_train.values)
        y_test_hat = lm.predict(X=x_test.values)

        z_train = pandas.concat((x_train, pandas.Series(y_train_hat, name="y_hat", index=y_train.index), y_train), axis=1)
        z_train.to_excel('../data/data_folds/data_train_ph2_{0}.xlsx'.format(fold_n), index=True)
        z_test = pandas.concat((x_test, pandas.Series(y_test_hat, name="y_hat", index=y_test.index), y_test), axis=1)
        z_test.to_excel('../data/data_folds/data_test_ph2_{0}.xlsx'.format(fold_n), index=True)


        perf_train = performer(x=y_train_hat, y=y_train.values)
        perf_test = performer(x=y_test_hat, y=y_test.values)

        final_perfs_train.append(perf_train)
        final_perfs_test.append(perf_test)

    final_perf = pandas.DataFrame(
        data={'perf_train': final_perfs_train,
              'perf_test': final_perfs_test, },
        index=fg.folds,
    )

    return selected_features, selected_summary, inter_perf, final_perf


def dt_proxy(
    fg,
    sources,
    features,
    target,
    timeaxis,
    performer,
    parameters,
):

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

    return selected_features, selected_summary, inter_perf, final_perf


def rf_proxy(
    fg,
    sources,
    features,
    target,
    timeaxis,
    performer,
    parameters,
):

    rf_kwg = {
        # 'max_depth': 5,
    }

    final_perfs_train = []
    final_perfs_test = []
    models = []
    significances = {feature: [] for feature in features}
    for fold_n in fg.folds:

        data_train, data_test = fg.fold(sources, features + [target], timeaxis, fold_n=fold_n)
        x_train, y_train = data_train[[x for x in data_train.columns if x != target]].iloc[:-1, :], data_train[
                                                                                                        target].iloc[1:]
        x_test, y_test = data_test[[x for x in data_test.columns if x != target]].iloc[:-1, :], data_test[target].iloc[
                                                                                                1:]

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
              'perf_test': final_perfs_test, },
        index=fg.folds,
    )

    significances = pandas.DataFrame(data=significances)
    selected_summary = (significances > 0.01).sum(axis=0)
    selected_features = selected_summary.index[selected_summary == 10].tolist()

    rf_kwg = {
        # 'max_depth': 5,
    }

    print('finalize')
    final_perfs_train = []
    final_perfs_test = []
    for fold_n in fg.folds:
        data_train, data_test = fg.fold(sources, selected_features + [target], timeaxis, fold_n=fold_n)
        x_train, y_train = data_train[[x for x in data_train.columns if x != target]].iloc[:-1, :], data_train[
                                                                                                        target].iloc[1:]
        x_test, y_test = data_test[[x for x in data_test.columns if x != target]].iloc[:-1, :], data_test[target].iloc[
                                                                                                1:]

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
              'perf_test': final_perfs_test, },
        index=fg.folds,
    )

    return selected_features, selected_summary, inter_perf, final_perf


def kn_proxy(
    fg,
    sources,
    features,
    target,
    timeaxis,
    performer,
    parameters,
):

    perfs_u = {}
    perfs_w = {}
    m = parameters['m']

    for z in range(m):
        print(z)

        kn_kwg = {
            'n_neighbors': z + 1,
            'weights': 'uniform'
        }

        final_perfs_train = []
        final_perfs_test = []
        for fold_n in fg.folds:
            data_train, data_test = fg.fold(sources, features + [target], timeaxis, fold_n=fold_n)
            x_train, y_train = data_train[[x for x in data_train.columns if x != target]].iloc[:-1, :], data_train[
                                                                                                            target].iloc[
                                                                                                        1:]
            x_test, y_test = data_test[[x for x in data_test.columns if x != target]].iloc[:-1, :], data_test[
                                                                                                        target].iloc[1:]

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
                  'perf_test': final_perfs_test, },
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
            x_train, y_train = data_train[[x for x in data_train.columns if x != target]].iloc[:-1, :], data_train[
                                                                                                            target].iloc[
                                                                                                        1:]
            x_test, y_test = data_test[[x for x in data_test.columns if x != target]].iloc[:-1, :], data_test[
                                                                                                        target].iloc[1:]

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
                  'perf_test': final_perfs_test, },
            index=fg.folds,
        )

        perfs_w[z] = inter_perf

    fig_name = 'nb_performance.svg'

    mm = m
    fig, ax = pyplot.subplots(fg.n_folds, 2)
    for fold_n in fg.folds:
        perf_u_series = [perfs_u[j].loc[fold_n, 'perf_test'] for j in perfs_u.keys() if j < mm]
        perf_w_series = [perfs_w[j].loc[fold_n, 'perf_test'] for j in perfs_w.keys() if j < mm]
        pandas.Series(perf_u_series).plot(ax=ax[fold_n, 0])
        pandas.Series(perf_w_series).plot(ax=ax[fold_n, 1])
    fig.savefig(fig_name)

    kn_kwg = {
        'n_neighbors': parameters['n'] + 1,
        'weights': parameters['w'],
    }

    final_perfs_train = []
    final_perfs_test = []
    for fold_n in fg.folds:
        data_train, data_test = fg.fold(sources, features + [target], timeaxis, fold_n=fold_n)
        x_train, y_train = data_train[[x for x in data_train.columns if x != target]].iloc[:-1, :], data_train[
                                                                                                        target].iloc[1:]
        x_test, y_test = data_test[[x for x in data_test.columns if x != target]].iloc[:-1, :], data_test[target].iloc[
                                                                                                1:]

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
              'perf_test': final_perfs_test, },
        index=fg.folds,
    )

    selected_features = features
    selected_summary = None
    inter_perf = fig_name

    return selected_features, selected_summary, inter_perf, final_perf
