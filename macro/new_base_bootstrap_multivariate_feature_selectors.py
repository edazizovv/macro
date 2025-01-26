
import numpy
import pandas
from sklearn.linear_model import LinearRegression
from scipy.stats import ecdf, norm, multivariate_normal, ks_2samp


#
def bootstrap_copula_perf_improvement_sampler(
    x_train,
    y_train,
    x_test,
    y_test,
    lasso_summary_distinct,
    performer,
    n_bootstrap_samples,
    bootstrap_sample_size_rate,
    signif_alpha,
):
    prev_perf_train = 0
    prev_perf_test = 0
    for j in range(lasso_summary_distinct.shape[0]):
        print('{0} / {1} jth'.format(j, lasso_summary_distinct.shape[0]))

        row = lasso_summary_distinct.iloc[j, :]
        row_ix = lasso_summary_distinct.index[j]

        jth_x_train = x_train[row['selected_features']].values
        jth_y_train = y_train.values

        jth_model = LinearRegression()
        jth_model.fit(X=jth_x_train, y=jth_y_train)
        jth_y_hat_train = jth_model.predict(X=jth_x_train)

        jth_train_perf = performer(x=jth_y_hat_train, y=jth_y_train)

        jth_distr_empirical = numpy.concatenate(
            (
                jth_x_train,
                jth_y_train.reshape(-1, 1),
            ),
            axis=1,
        )

        jth_cov_mx_empirical = pandas.DataFrame(jth_distr_empirical).cov().values
        joint_cov_zero = jth_cov_mx_empirical.copy()
        joint_cov_zero[-2, -1] = 0
        joint_cov_zero[-1, -2] = 0

        n = jth_x_train.shape[0]
        m = joint_cov_zero.shape[0]
        zero_means = numpy.zeros(shape=m)

        if (numpy.linalg.eigvals(joint_cov_zero) >= 0).all():
            generator_normal = multivariate_normal(
                mean=zero_means,
                cov=joint_cov_zero,
                allow_singular=True,
            )
        else:
            print("cannot adjust cov matrix to zero latest factor; applying total random for the case")

            white_cov = numpy.zeros(shape=joint_cov_zero.shape)
            empirical_covariances = numpy.diagonal(jth_cov_mx_empirical)

            for k in range(m):
                white_cov[k, k] = empirical_covariances[k]

            generator_normal = multivariate_normal(
                mean=zero_means,
                cov=white_cov,
                allow_singular=True,
            )

        perf_impr_boostrapped_sample = []
        for i in range(n_bootstrap_samples):

            sample_bootstrapped = {}
            normal_sample = generator_normal.rvs(size=int(n * bootstrap_sample_size_rate))

            for k in range(m):
                scale = joint_cov_zero[k, k]
                if scale != 0:
                    univariate_normal = norm(loc=0, scale=scale)
                    q_values = univariate_normal.cdf(x=normal_sample[:, k])
                    x_values = [
                        numpy.quantile(
                            a=jth_distr_empirical[:, k],
                            q=q,
                        )
                        for q in q_values
                    ]
                    x_values = numpy.array(x_values)
                else:
                    x_values = numpy.zeros(shape=(jth_distr_empirical.shape[0],))
                sample_bootstrapped[k] = x_values

            sample_bootstrapped = pandas.DataFrame(sample_bootstrapped).values

            sample_x_train_before = sample_bootstrapped[:, :-2]
            sample_x_train_after = sample_bootstrapped[:, :-1]

            sample_y_train = sample_bootstrapped[:, -1]

            sample_model_after = LinearRegression()
            sample_model_after.fit(X=sample_x_train_after, y=sample_y_train)
            sample_y_hat_after = sample_model_after.predict(X=sample_x_train_after)
            sample_perf_after = performer(x=sample_y_hat_after, y=sample_y_train)

            if sample_x_train_before.shape[1] == 0:
                sample_perf_before = 0
            else:
                sample_model_before = LinearRegression()
                sample_model_before.fit(X=sample_x_train_before, y=sample_y_train)
                sample_y_hat_before = sample_model_before.predict(X=sample_x_train_before)
                sample_perf_before = performer(x=sample_y_hat_before, y=sample_y_train)

            sample_perf_improvement = sample_perf_after - sample_perf_before

            perf_impr_boostrapped_sample.append(sample_perf_improvement)

        jth_perf_improvement = jth_train_perf - prev_perf_train
        perf_impr_pval = 1 - ecdf(perf_impr_boostrapped_sample).cdf.evaluate(jth_perf_improvement)

        jth_y_hat_test = jth_model.predict(X=x_test[row['selected_features']].values)

        jth_test_perf = performer(x=jth_y_hat_test, y=y_test.values)
        lasso_summary_distinct.loc[row_ix, 'perf_train'] = jth_train_perf
        lasso_summary_distinct.loc[row_ix, 'perf_test'] = jth_test_perf
        lasso_summary_distinct.loc[row_ix, 'perf_impr_train_value'] = jth_train_perf - prev_perf_train
        lasso_summary_distinct.loc[row_ix, 'perf_impr_test_value'] = jth_test_perf - prev_perf_test
        lasso_summary_distinct.loc[row_ix, 'perf_impr_train_significance'] = perf_impr_pval

        prev_perf_train = jth_train_perf
        prev_perf_test = jth_test_perf

    lasso_summary_distinct['_perf_signigicant_flag'] = (
            lasso_summary_distinct['perf_impr_train_significance'] <= signif_alpha).astype(dtype=int)
    lasso_summary_distinct['perf_improved_w3'] = lasso_summary_distinct['_perf_signigicant_flag'].rolling(
        window=3, min_periods=1).mean()
    lasso_summary_distinct['perf_improved_w3_mask'] = (lasso_summary_distinct['perf_improved_w3'] > 0.66).astype(
        dtype=int)
    lasso_summary_distinct['cover'] = lasso_summary_distinct['perf_improved_w3_mask'].cumprod()

    return lasso_summary_distinct


def bootstrap_copula_perf_difference_sampler(
    x_train,
    y_train,
    x_test,
    y_test,
    lasso_summary_distinct,
    performer,
    n_bootstrap_samples,
    bootstrap_sample_size_rate,
    signif_alpha,
):
    prev_perf_train = 0
    prev_perf_test = 0
    for j in range(lasso_summary_distinct.shape[0]):
        print('{0} / {1} jth'.format(j, lasso_summary_distinct.shape[0]))

        row = lasso_summary_distinct.iloc[j, :]
        row_ix = lasso_summary_distinct.index[j]

        jth_x_train = x_train[row['selected_features']].values
        jth_y_train = y_train.values

        jth_model = LinearRegression()
        jth_model.fit(X=jth_x_train, y=jth_y_train)
        jth_y_hat_train = jth_model.predict(X=jth_x_train)

        jth_train_perf = performer(x=jth_y_hat_train, y=jth_y_train)

        jth_distr_empirical = numpy.concatenate(
            (
                jth_x_train,
                jth_y_train.reshape(-1, 1),
            ),
            axis=1,
        )

        jth_cov_mx_empirical = pandas.DataFrame(jth_distr_empirical).cov().values
        joint_cov_zero = jth_cov_mx_empirical.copy()

        n = jth_x_train.shape[0]
        m = joint_cov_zero.shape[0]
        zero_means = numpy.zeros(shape=m)

        if (numpy.linalg.eigvals(joint_cov_zero) >= 0).all():
            generator_normal = multivariate_normal(
                mean=zero_means,
                cov=joint_cov_zero,
                allow_singular=True,
            )
        else:
            raise Exception("the cov matrix is misspecified for some reason; please take a look at what's going on here")

        perf_previous_sampled = []
        perf_jth_sampled = []
        for i in range(n_bootstrap_samples):

            sample_bootstrapped = {}
            normal_sample = generator_normal.rvs(size=int(n * bootstrap_sample_size_rate))

            for k in range(m):
                scale = joint_cov_zero[k, k]
                if scale != 0:
                    univariate_normal = norm(loc=0, scale=scale)
                    q_values = univariate_normal.cdf(x=normal_sample[:, k])
                    x_values = [
                        numpy.quantile(
                            a=jth_distr_empirical[:, k],
                            q=q,
                        )
                        for q in q_values
                    ]
                    x_values = numpy.array(x_values)
                else:
                    x_values = numpy.zeros(shape=(jth_distr_empirical.shape[0],))
                sample_bootstrapped[k] = x_values

            sample_bootstrapped = pandas.DataFrame(sample_bootstrapped).values

            sample_x_train_before = sample_bootstrapped[:, :-2]
            sample_x_train_after = sample_bootstrapped[:, :-1]

            sample_y_train = sample_bootstrapped[:, -1]

            sample_model_after = LinearRegression()
            sample_model_after.fit(X=sample_x_train_after, y=sample_y_train)
            sample_y_hat_after = sample_model_after.predict(X=sample_x_train_after)
            sample_perf_after = performer(x=sample_y_hat_after, y=sample_y_train)

            if sample_x_train_before.shape[1] == 0:
                sample_perf_before = 0
            else:
                sample_model_before = LinearRegression()
                sample_model_before.fit(X=sample_x_train_before, y=sample_y_train)
                sample_y_hat_before = sample_model_before.predict(X=sample_x_train_before)
                sample_perf_before = performer(x=sample_y_hat_before, y=sample_y_train)

            perf_previous_sampled.append(sample_perf_before)
            perf_jth_sampled.append(sample_perf_after)

        perf_impr_pval = ks_2samp(data1=perf_previous_sampled, data2=perf_jth_sampled, alternative='greater').pvalue

        jth_y_hat_test = jth_model.predict(X=x_test[row['selected_features']].values)

        jth_test_perf = performer(x=jth_y_hat_test, y=y_test.values)
        lasso_summary_distinct.loc[row_ix, 'perf_train'] = jth_train_perf
        lasso_summary_distinct.loc[row_ix, 'perf_test'] = jth_test_perf
        lasso_summary_distinct.loc[row_ix, 'perf_impr_train_value'] = jth_train_perf - prev_perf_train
        lasso_summary_distinct.loc[row_ix, 'perf_impr_test_value'] = jth_test_perf - prev_perf_test
        lasso_summary_distinct.loc[row_ix, 'perf_impr_train_significance'] = perf_impr_pval

        prev_perf_train = jth_train_perf
        prev_perf_test = jth_test_perf

    lasso_summary_distinct['_perf_signigicant_flag'] = (
            lasso_summary_distinct['perf_impr_train_significance'] <= signif_alpha).astype(dtype=int)
    lasso_summary_distinct['perf_improved_w3'] = lasso_summary_distinct['_perf_signigicant_flag'].rolling(
        window=3, min_periods=1).mean()
    # lasso_summary_distinct['perf_improved_w3'] =
    lasso_summary_distinct['perf_improved_w3_mask'] = (lasso_summary_distinct['perf_improved_w3'] > 0.66).astype(
        dtype=int)
    lasso_summary_distinct['cover'] = lasso_summary_distinct['perf_improved_w3_mask'].cumprod()

    return lasso_summary_distinct


def bootstrap_direct_perf_improvement_sampler(
    x_train,
    y_train,
    x_test,
    y_test,
    lasso_summary_distinct,
    performer,
    n_bootstrap_samples,
    bootstrap_sample_size_rate,
    signif_alpha,
):
    prev_perf_train = 0
    prev_perf_test = 0
    for j in range(lasso_summary_distinct.shape[0]):
        print('{0} / {1} jth'.format(j, lasso_summary_distinct.shape[0]))

        row = lasso_summary_distinct.iloc[j, :]
        row_ix = lasso_summary_distinct.index[j]

        jth_x_train = x_train[row['selected_features']].values
        jth_y_train = y_train.values

        jth_model = LinearRegression()
        jth_model.fit(X=jth_x_train, y=jth_y_train)
        jth_y_hat_train = jth_model.predict(X=jth_x_train)

        jth_train_perf = performer(x=jth_y_hat_train, y=jth_y_train)

        jth_distr_empirical = numpy.concatenate(
            (
                jth_x_train,
                jth_y_train.reshape(-1, 1),
            ),
            axis=1,
        )

        perf_impr_boostrapped_sample = []
        for i in range(n_bootstrap_samples):

            sample_bootstrapped = pandas.DataFrame(jth_distr_empirical).sample(
                n=int(jth_distr_empirical.shape[0] * bootstrap_sample_size_rate),
                replace=True,
            ).values

            sample_bootstrapped[:, -2] = numpy.random.normal(size=(sample_bootstrapped.shape[0]),)

            sample_x_train_before = sample_bootstrapped[:, :-2]
            sample_x_train_after = sample_bootstrapped[:, :-1]

            sample_y_train = sample_bootstrapped[:, -1]

            sample_model_after = LinearRegression()
            sample_model_after.fit(X=sample_x_train_after, y=sample_y_train)
            sample_y_hat_after = sample_model_after.predict(X=sample_x_train_after)
            sample_perf_after = performer(x=sample_y_hat_after, y=sample_y_train)

            if sample_x_train_before.shape[1] == 0:
                sample_perf_before = 0
            else:
                sample_model_before = LinearRegression()
                sample_model_before.fit(X=sample_x_train_before, y=sample_y_train)
                sample_y_hat_before = sample_model_before.predict(X=sample_x_train_before)
                sample_perf_before = performer(x=sample_y_hat_before, y=sample_y_train)

            sample_perf_improvement = sample_perf_after - sample_perf_before

            perf_impr_boostrapped_sample.append(sample_perf_improvement)

        jth_perf_improvement = jth_train_perf - prev_perf_train
        perf_impr_pval = 1 - ecdf(perf_impr_boostrapped_sample).cdf.evaluate(jth_perf_improvement)

        jth_y_hat_test = jth_model.predict(X=x_test[row['selected_features']].values)

        jth_test_perf = performer(x=jth_y_hat_test, y=y_test.values)
        lasso_summary_distinct.loc[row_ix, 'perf_train'] = jth_train_perf
        lasso_summary_distinct.loc[row_ix, 'perf_test'] = jth_test_perf
        lasso_summary_distinct.loc[row_ix, 'perf_impr_train_value'] = jth_train_perf - prev_perf_train
        lasso_summary_distinct.loc[row_ix, 'perf_impr_test_value'] = jth_test_perf - prev_perf_test
        lasso_summary_distinct.loc[row_ix, 'perf_impr_train_significance'] = perf_impr_pval

        prev_perf_train = jth_train_perf
        prev_perf_test = jth_test_perf

    lasso_summary_distinct['_perf_signigicant_flag'] = (
            lasso_summary_distinct['perf_impr_train_significance'] <= signif_alpha).astype(dtype=int)
    lasso_summary_distinct['perf_improved_w3'] = lasso_summary_distinct['_perf_signigicant_flag'].rolling(
        window=3, min_periods=1).mean()
    lasso_summary_distinct['perf_improved_w3_mask'] = (lasso_summary_distinct['perf_improved_w3'] > 0.66).astype(
        dtype=int)
    lasso_summary_distinct['cover'] = lasso_summary_distinct['perf_improved_w3_mask'].cumprod()

    return lasso_summary_distinct


def bootstrap_direct_perf_difference_sampler(
    x_train,
    y_train,
    x_test,
    y_test,
    lasso_summary_distinct,
    performer,
    n_bootstrap_samples,
    bootstrap_sample_size_rate,
    signif_alpha,
):
    prev_perf_train = 0
    prev_perf_test = 0
    for j in range(lasso_summary_distinct.shape[0]):
        print('{0} / {1} jth'.format(j, lasso_summary_distinct.shape[0]))

        row = lasso_summary_distinct.iloc[j, :]
        row_ix = lasso_summary_distinct.index[j]

        jth_x_train = x_train[row['selected_features']].values
        jth_y_train = y_train.values

        jth_model = LinearRegression()
        jth_model.fit(X=jth_x_train, y=jth_y_train)
        jth_y_hat_train = jth_model.predict(X=jth_x_train)

        jth_train_perf = performer(x=jth_y_hat_train, y=jth_y_train)

        jth_distr_empirical = numpy.concatenate(
            (
                jth_x_train,
                jth_y_train.reshape(-1, 1),
            ),
            axis=1,
        )

        perf_previous_sampled = []
        perf_jth_sampled = []
        for i in range(n_bootstrap_samples):

            sample_bootstrapped = pandas.DataFrame(jth_distr_empirical).sample(
                n=int(jth_distr_empirical.shape[0] * bootstrap_sample_size_rate),
                replace=True,
            ).values

            sample_x_train_before = sample_bootstrapped[:, :-2]
            sample_x_train_after = sample_bootstrapped[:, :-1]

            sample_y_train = sample_bootstrapped[:, -1]

            sample_model_after = LinearRegression()
            sample_model_after.fit(X=sample_x_train_after, y=sample_y_train)
            sample_y_hat_after = sample_model_after.predict(X=sample_x_train_after)
            sample_perf_after = performer(x=sample_y_hat_after, y=sample_y_train)

            if sample_x_train_before.shape[1] == 0:
                sample_perf_before = 0
            else:
                sample_model_before = LinearRegression()
                sample_model_before.fit(X=sample_x_train_before, y=sample_y_train)
                sample_y_hat_before = sample_model_before.predict(X=sample_x_train_before)
                sample_perf_before = performer(x=sample_y_hat_before, y=sample_y_train)

            perf_previous_sampled.append(sample_perf_before)
            perf_jth_sampled.append(sample_perf_after)

        perf_impr_pval = ks_2samp(data1=perf_previous_sampled, data2=perf_jth_sampled, alternative='greater').pvalue

        jth_y_hat_test = jth_model.predict(X=x_test[row['selected_features']].values)

        jth_test_perf = performer(x=jth_y_hat_test, y=y_test.values)
        lasso_summary_distinct.loc[row_ix, 'perf_train'] = jth_train_perf
        lasso_summary_distinct.loc[row_ix, 'perf_test'] = jth_test_perf
        lasso_summary_distinct.loc[row_ix, 'perf_impr_train_value'] = jth_train_perf - prev_perf_train
        lasso_summary_distinct.loc[row_ix, 'perf_impr_test_value'] = jth_test_perf - prev_perf_test
        lasso_summary_distinct.loc[row_ix, 'perf_impr_train_significance'] = perf_impr_pval

        prev_perf_train = jth_train_perf
        prev_perf_test = jth_test_perf

    lasso_summary_distinct['_perf_signigicant_flag'] = (
            lasso_summary_distinct['perf_impr_train_significance'] <= signif_alpha).astype(dtype=int)
    lasso_summary_distinct['perf_improved_w3'] = lasso_summary_distinct['_perf_signigicant_flag'].rolling(
        window=3, min_periods=1).mean()
    lasso_summary_distinct['perf_improved_w3_mask'] = (lasso_summary_distinct['perf_improved_w3'] > 0.66).astype(
        dtype=int)
    lasso_summary_distinct['cover'] = lasso_summary_distinct['perf_improved_w3_mask'].cumprod()

    return lasso_summary_distinct
