#


#
import numpy
import pandas
from scipy.stats import multivariate_normal, norm


#
from macro.new_base_trueuse_pods import vxl


#

def _bootstrap_empirical_distribution_generator(
    x,
    y,
    features,
    performer,
    n_bootstrap_samples,
    bootstrap_sample_size_rate,
):

    marginal_empirical_distributions = numpy.concatenate(
        (
            x.values,
            y.values.reshape(-1, 1),
        ),
        axis=1,
    )
    joint_cov = pandas.DataFrame(marginal_empirical_distributions).cov()

    joint_cov_zero = joint_cov.copy()
    target_self_var = joint_cov_zero.iloc[-1, -1]
    joint_cov_zero.iloc[-1, :] = 0
    joint_cov_zero.iloc[:, -1] = 0
    joint_cov_zero.iloc[-1, -1] = target_self_var

    joint_cov_zero_corr = joint_cov_zero.copy()

    if not (numpy.linalg.eigvals(joint_cov_zero) > 0).all():
        print("Warning: cov matrix for an iteration appears to be not positive semidefinitive; applying mitigation")
        joint_cov_zero_corr = joint_cov_zero / 2

        for j in range(joint_cov_zero_corr.shape[0]):
            joint_cov_zero_corr.iloc[j, j] = joint_cov_zero.iloc[j, j]

        while not (numpy.linalg.eigvals(joint_cov_zero_corr) >= 0).all():

            joint_cov_zero_corr = joint_cov_zero_corr / 2

            for j in range(joint_cov_zero_corr.shape[0]):
                joint_cov_zero_corr.iloc[j, j] = joint_cov_zero.iloc[j, j]

    joint_names = x.columns.tolist() + [y.name]
    n = x.shape[0]
    m = joint_cov.shape[0]
    zero_means = numpy.zeros(shape=m)
    generator_normal = multivariate_normal(
        mean=zero_means,
        cov=joint_cov_zero_corr,
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
        if any([pandas.isna(a) for a in perf_boostrap]):
            # raise Exception()
            perf_boostrap = [a if ~pandas.isna(a) else 0 for a in perf_boostrap]
            # TODO: to be investigated & resolved
        perf_boostrapped_summary.append(perf_boostrap)
    perf_boostrapped_summary = pandas.DataFrame(
        data=perf_boostrapped_summary,
        columns=features,
        index=list(range(n_bootstrap_samples)),
    )

    return perf_boostrapped_summary


def vincent_class_feature_selection_mechanism(
    fg,
    sources,
    predictors,
    target,
    timeaxis,
    performer,
    alpha,
    n_bootstrap_samples,
    bootstrap_sample_size_rate,
):
    check_results = []
    perf_test = []
    for fold_n in fg.folds:
        data_train, data_test = fg.fold(sources, predictors + [target], timeaxis, fold_n=fold_n)
        x_train, y_train = data_train[[x for x in data_train.columns if x != target]].iloc[:-1, :], data_train[target].iloc[1:]
        x_test, y_test = data_test[[x for x in data_test.columns if x != target]].iloc[:-1, :], data_test[target].iloc[1:]

        z_train = pandas.concat((x_train, y_train), axis=1)
        z_train.to_excel('../data/data_folds/data_train_ph1_{0}.xlsx'.format(fold_n), index=True)
        z_test = pandas.concat((x_test, y_test), axis=1)
        z_test.to_excel('../data/data_folds/data_test_ph1_{0}.xlsx'.format(fold_n), index=True)

        # calculate significances with bootstrap

        perf_boostrapped_summary = _bootstrap_empirical_distribution_generator(
            x=x_train,
            y=y_train,
            features=predictors,
            performer=performer,
            n_bootstrap_samples=n_bootstrap_samples,
            bootstrap_sample_size_rate=bootstrap_sample_size_rate,
        )

        for feature in predictors:

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

            if pandas.isna(perf):
                if x_train[feature].unique().shape[0] == 1:
                    perf = 0
                    ci_thresh_lower = -1
                    ci_thresh_upper = 1
                else:
                    raise Exception("missing perf?")
            if pandas.isna(ci_thresh_lower) or pandas.isna(ci_thresh_upper):
                raise Exception()
                perf_pass = False
            else:
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
            for feature in predictors
        ]
        perf_test.append(perf_test_fold)

    check_results = pandas.DataFrame(
        data=check_results,
        columns=['fold_n', 'feature', 'perf', 'perf_test', 'ci_thresh_lower', 'ci_thresh_upper', 'perf_pass'],
    )

    perf_test = pandas.DataFrame(data=perf_test, columns=predictors)
    perf_test_agg = perf_test.mean(axis=0).T
    perf_test_agg = pandas.DataFrame(data=perf_test_agg, columns=['perf_test'])
    perf_test_agg = perf_test_agg.reset_index()
    perf_test_agg = perf_test_agg.rename(columns={'index': 'feature'})

    return check_results, perf_test_agg


def util_prepare_chosen(
    chosen,
    perf_test_agg,
    check_results_agg,
):

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

    chosen['sources'] = chosen['feature'].apply(func=lambda x: x.split('__')[0])
    chosen['transforms'] = chosen['feature'].apply(func=lambda x: x.split('__')[1])
    chosen = chosen.merge(
        right=vxl,
        left_on=['transforms'],
        right_on=['transform_name'],
        how='left',
    )

    return chosen
