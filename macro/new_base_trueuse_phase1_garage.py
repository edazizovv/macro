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

    joint_names = x.columns.tolist() + [y.name]
    n = x.shape[0]
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

    return perf_boostrapped_summary


def vincent_class_feature_selection_mechanism(
    fg,
    sources,
    features,
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
        data_train, data_test = fg.fold(sources, features + [target], timeaxis, fold_n=fold_n)
        x_train, y_train = data_train[[x for x in data_train.columns if x != target]].iloc[:-1, :], data_train[target].iloc[1:]
        x_test, y_test = data_test[[x for x in data_test.columns if x != target]].iloc[:-1, :], data_test[target].iloc[1:]

        # calculate significances with bootstrap

        perf_boostrapped_summary = _bootstrap_empirical_distribution_generator(
            x=x_train,
            y=y_train,
            features=features,
            performer=performer,
            n_bootstrap_samples=n_bootstrap_samples,
            bootstrap_sample_size_rate=bootstrap_sample_size_rate,
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

    perf_test = pandas.DataFrame(data=perf_test, columns=features)
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
