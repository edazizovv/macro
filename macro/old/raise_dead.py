#
import pandas

#
# data = pandas.read_csv('./result.csv')
data = pandas.read_csv('./result_model_medieval.csv')
data = data.set_index('jx')
# GDP GS3 A091RC1Q027SBEA TLT IVV
"""
y = data['IVV'].values
mstl = MSTL(y, periods=[3, 6, 12, 36])  # 3 = quarter, 6 = semiyear, 12 = year, 36 = 3 years
res = mstl.fit()
resulted = pandas.DataFrame(data={'date': data.index.values,
                                  'origin': y,
                                  'seasonal_3': res.seasonal[:, 0],
                                  'seasonal_6': res.seasonal[:, 1],
                                  'seasonal_12': res.seasonal[:, 2],
                                  'seasonal_36': res.seasonal[:, 3],
                                  'trend': res.trend,
                                  'residual': res.resid, })

ax = resulted.plot(x='date', y=['origin', 'trend', 'residual'])
resulted.plot(x='date', y=['seasonal_3', 'seasonal_6', 'seasonal_12', 'seasonal_36'], secondary_y=True, ax=ax)
"""

# https://www.statsmodels.org/dev/examples/notebooks/generated/tsa_filters.html#Christiano-Fitzgerald-approximate-band-pass-filter:-Inflation-and-Unemployment


# ['GS3', 'T10YIEM', 'USACPIALLMINMEI', 'KCFSI', 'MICH']    TLT_MEAN__pct
# ['TOTALSA', 'GDP', 'USACPIALLMINMEI', 'EMRATIO', 'KCFSI']
#
from macro.macro.functional import SomersD


def target_defined(y):
    if y < -0.015:
        return -1
    elif y < 0.015:
        return 0
    else:
        return 1


def score_defined(y):
    if y < -0.015:
        return -1
    elif y < 0.015:
        return 0
    else:
        return 1


data['target'] = data['TLT_MEAN__pct'].apply(func=target_defined)

"""
target = 'TLT_MEAN__pct'
pcs = []
sds = []
for j in range(data.columns.shape[0]):
    if j % 100 == 0:
        print('{0} / {1}'.format(j, data.columns.shape[0]))
    c = data.columns[j]
    pc = pearsonr(x=data[target].values[1:], y=data[c].values[:-1]).statistic
    # sd = somersd(x=data[target].values, y=data[c].values).statistic
    sd = SomersD(data[target].values[1:], data[c].values[:-1])
    pcs.append(pc)
    sds.append(sd)
"""

target = 'TLT_MEAN__pct'
x_factors = data.columns.values
time_axis = data.index
time_sub_rate = 0.50
time_sub_replace = False
nt = int(time_axis.shape[0] * time_sub_rate)
sds = []
for j in range(data.columns.shape[0]):
    import numpy
    # ix_time = numpy.random.choice(time_axis.values, size=(nt,), replace=time_sub_replace)
    ixes = numpy.random.choice(list(range(time_axis.shape[0] - 1)), size=(nt,), replace=time_sub_replace)
    x_ix_train = time_axis.values[:-1][ixes]
    y_ix_train = time_axis.values[1:][ixes]
    left_ixes = [x for x in list(range(time_axis.shape[0] - 1)) if x not in ixes]
    x_ix_test = time_axis.values[:-1][left_ixes]
    y_ix_test = time_axis.values[1:][left_ixes]

    x_train = data.loc[x_ix_train, x_factors].values
    y_train = data.loc[y_ix_train, target].values
    x_test = data.loc[x_ix_test, x_factors].values
    y_test = data.loc[y_ix_test, target].values
    biny_train = data.loc[y_ix_train, 'target'].values
    biny_test = data.loc[y_ix_test, 'target'].values

    xxx_train, xxx_test, yyy_train, yyy_test = x_train, x_test, y_train, y_test
    c = data.columns[j]
    sd = SomersD(data[target].values[1:], data[c].values[:-1])
    sds.append(sd)
# pandas.Series(data=pcs, index=data.columns.values).hist(bins=20)
# pandas.Series(data=sds, index=data.columns.values).hist(bins=20)
# pandas.Series(data=pcs, index=data.columns.values).dropna().sort_values().iloc[-10:]
# pandas.Series(data=sds, index=data.columns.values).dropna().sort_values().iloc[-10:]


# """
target = 'TLT_MEAN__pct'
# x_factors = pandas.Series(data=sds, index=data.columns.values).dropna().sort_values().iloc[-30:].index.values.tolist()
# x_factors = ['TLT__pct']
# '''
x_factors = ['PCU483111483111__full_seasonal_remove_mstl_get_seasonal_resid__rolling_3__None__<function relative_mean at 0x000001E0373865F0>',
 'PSAVERT__full_seasonal_remove_mstl_get_seasonal_resid__rolling_12__None__<function n_rate_conseq at 0x000001E037386290>',
 'TLT_MEAN__full_seasonal_remove_mstl_get_seasonal_resid__rolling_3__None__<function rel_to_max at 0x000001E0373863B0>',
 'CUUR0000SETA01__full_seasonal_remove_mstl_get_seasonal_resid__rolling_6__None__<function mean_relative at 0x000001E0373855A0>',
 'A824RL1Q225SBEA__rolling_12__None__<function ewm_6shock_relative_12 at 0x000001E037385B40>',
 'BAA__rolling_6__None__<function pct_std at 0x000001E0373864D0>',
 'PCU4883204883208__full_seasonal_remove_mstl_get_seasonal_resid__rolling_6__None__<function ewm_1shock_relative_3 at 0x000001E037385870>',
 'PCU483111483111__full_seasonal_remove_mstl_get_seasonal_resid__rolling_6__None__<function median_relative at 0x000001E037385630>',
 'CES3000000008__rolling_6__None__<function relative_positive_pct at 0x000001E037386680>',
 'A824RL1Q225SBEA__rolling_12__None__<function ewm_3shock_relative_12 at 0x000001E037385AB0>',
 'PCEDG__rolling_3__None__<function relative_q at 0x000001E0373857E0>',
 'A824RL1Q225SBEA__rolling_6__None__<function ewm_1shock_relative_12 at 0x000001E037385990>',
 'CUUR0000SEHA__full_seasonal_remove_mstl_get_seasonal_resid__rolling_3__None__<function relative_q at 0x000001E0373857E0>',
 'PCUOMFGOMFG__full_seasonal_remove_mstl_get_seasonal_resid__rolling_12__None__<function mean_relative at 0x000001E0373855A0>',
 'GS3M__full_seasonal_remove_mstl_get_seasonal_resid__rolling_6__None__<function rel_to_max at 0x000001E0373863B0>',
 'A824RL1Q225SBEA__rolling_12__None__<function linreg_relative at 0x000001E037385750>',
 'GS3__full_seasonal_remove_mstl_get_seasonal_resid__rolling_6__None__<function mean_relative at 0x000001E0373855A0>',
 'PCU483111483111__rolling_3__None__<function linear_r2err at 0x000001E037386200>',
 'CES3000000008__rolling_3__None__<function n_rate_full at 0x000001E037386320>',
 'EMVFINCRISES__full_seasonal_remove_mstl_get_seasonal_resid__rolling_6__None__<function ewm_1shock_relative_6 at 0x000001E037385900>',
 'INTDSRINM193N__full_seasonal_remove_mstl_get_seasonal_resid__rolling_6__None__<function relative_mean at 0x000001E0373865F0>',
 'GDP__full_seasonal_remove_mstl_get_seasonal_resid__rolling_12__None__<function linreg_relative at 0x000001E037385750>',
 'CPF1M__full_seasonal_remove_mstl_get_seasonal_resid__rolling_12__None__<function mean_relative at 0x000001E0373855A0>',
 'PCU484121484121__full_seasonal_remove_mstl_get_seasonal_resid__rolling_12__None__<function linreg_relative at 0x000001E037385750>',
 'SPASTT01BRM657N__full_seasonal_remove_mstl_get_seasonal_resid__rolling_3__None__<function relative_q at 0x000001E0373857E0>',
 'PCU4841214841212__full_seasonal_remove_mstl_get_seasonal_resid__rolling_12__None__<function mean_relative at 0x000001E0373855A0>',
 'PSAVERT__rolling_12__None__<function relative_q at 0x000001E0373857E0>',
 'CSUSHPINSA__full_seasonal_remove_mstl_get_seasonal_resid__rolling_6__None__<function pct_std at 0x000001E0373864D0>',
 'BAMLC0A0CMEY__rolling_3__None__<function relative_positive_pct at 0x000001E037386680>',
 'PCUOMFGOMFG__full_seasonal_remove_mstl_get_seasonal_resid__rolling_6__None__<function n_rate_conseq at 0x000001E037386290>',
 'TLT__pct__full_binners_20_perc',
 'PCUOMFGOMFG__rolling_3__None__<function linreg_relative at 0x000001E037385750>',
 'GS10__pct',
 'TLT_MEAN__full_seasonal_remove_mstl_get_seasonal_resid__rolling_3__None__<function relative_q at 0x000001E0373857E0>',
 'AWHAERT__rolling_3__None__<function relative_q at 0x000001E0373857E0>',
 'INTDSRTRM193N__full_seasonal_remove_mstl_get_seasonal_resid__rolling_6__None__<function n_rate_full at 0x000001E037386320>']


# '''
time_axis = data.index
time_sub_rate = 0.50
time_sub_replace = False
nt = int(time_axis.shape[0] * time_sub_rate)
n = 100
r2_trains, r2_tests = [], []
kt_trains, kt_tests = [], []
kt_trains_bin, kt_tests_bin = [], []
for j in range(n):
    import numpy
    # ix_time = numpy.random.choice(time_axis.values, size=(nt,), replace=time_sub_replace)
    ixes = numpy.random.choice(list(range(time_axis.shape[0] - 1)), size=(nt,), replace=time_sub_replace)
    x_ix_train = time_axis.values[:-1][ixes]
    y_ix_train = time_axis.values[1:][ixes]
    left_ixes = [x for x in list(range(time_axis.shape[0] - 1)) if x not in ixes]
    x_ix_test = time_axis.values[:-1][left_ixes]
    y_ix_test = time_axis.values[1:][left_ixes]

    x_train = data.loc[x_ix_train, x_factors].values
    y_train = data.loc[y_ix_train, target].values
    x_test = data.loc[x_ix_test, x_factors].values
    y_test = data.loc[y_ix_test, target].values
    biny_train = data.loc[y_ix_train, 'target'].values
    biny_test = data.loc[y_ix_test, 'target'].values

    xxx_train, xxx_test, yyy_train, yyy_test = x_train, x_test, y_train, y_test
    from sklearn.preprocessing import StandardScaler
    sk = StandardScaler()
    xxx_train_st = sk.fit_transform(X=xxx_train)
    xxx_test_st = sk.transform(X=xxx_test)
    from sklearn.linear_model import LinearRegression
    m = LinearRegression()
    m.fit(X=xxx_train_st, y=yyy_train)
    y_hat_train = m.predict(X=xxx_train_st)
    y_hat_test = m.predict(X=xxx_test_st)
    from sklearn.metrics import r2_score
    r2_train = r2_score(y_true=yyy_train, y_pred=y_hat_train)
    r2_test = r2_score(y_true=yyy_test, y_pred=y_hat_test)
    from scipy.stats import kendalltau
    kt_train = kendalltau(x=yyy_train, y=y_hat_train).statistic
    kt_test = kendalltau(x=yyy_test, y=y_hat_test).statistic
    y_hat_train_bin = pandas.Series(y_hat_train).apply(func=score_defined).values
    y_hat_test_bin = pandas.Series(y_hat_test).apply(func=score_defined).values
    kt_train_bin = kendalltau(x=biny_train, y=y_hat_train_bin).statistic
    kt_test_bin = kendalltau(x=biny_test, y=y_hat_test_bin).statistic
    # fig, ax = pyplot.subplots(1, 2)
    # pandas.DataFrame(data={'error': yyy_train - y_hat_train}).hist(bins=20, ax=ax[0])
    # pandas.DataFrame(data={'error': yyy_test - y_hat_test}).hist(bins=20, ax=ax[1])
    r2_trains.append(r2_train)
    r2_tests.append(r2_test)
    kt_trains.append(kt_train)
    kt_tests.append(kt_test)
    kt_trains_bin.append(kt_train_bin)
    kt_tests_bin.append(kt_test_bin)
# """
# zog = pandas.DataFrame(data={'y_true': yyy_test, 'y_hat': y_hat_test, 'y_bin': biny_test, 'y_hat_bin': y_hat_test_bin})
# from sklearn.metrics import confusion_matrix
# cm = confusion_matrix(y_true=biny_test, y_pred=y_hat_test_bin)
"""
# SELF
target = 'TLT_MEAN__pct'
x_factors = ['TLT__pct']
time_axis = data.index
time_sub_rate = 0.75
time_sub_replace = False
nt = int(time_axis.shape[0] * time_sub_rate)
n = 100
r2_trains, r2_tests = [], []
kt_trains, kt_tests = [], []
for j in range(n):
    import numpy
    # ix_time = numpy.random.choice(time_axis.values, size=(nt,), replace=time_sub_replace)
    ixes = numpy.random.choice(list(range(time_axis.shape[0] - 1)), size=(nt,), replace=time_sub_replace)
    x_ix_train = time_axis.values[:-1][ixes]
    y_ix_train = time_axis.values[1:][ixes]
    left_ixes = [x for x in list(range(time_axis.shape[0] - 1)) if x not in ixes]
    x_ix_test = time_axis.values[:-1][left_ixes]
    y_ix_test = time_axis.values[1:][left_ixes]

    x_train = data.loc[x_ix_train, x_factors].values
    y_train = data.loc[y_ix_train, target].values
    x_test = data.loc[x_ix_test, x_factors].values
    y_test = data.loc[y_ix_test, target].values

    from sklearn.model_selection import train_test_split
    xxx_train, xxx_test, yyy_train, yyy_test = x_train, x_test, y_train, y_test
    from sklearn.preprocessing import StandardScaler
    sk = StandardScaler()
    xxx_train_st = sk.fit_transform(X=xxx_train)
    xxx_test_st = sk.transform(X=xxx_test)
    from sklearn.linear_model import LinearRegression
    m = LinearRegression()
    m.fit(X=xxx_train_st, y=yyy_train)
    y_hat_train = m.predict(X=xxx_train_st)
    y_hat_test = m.predict(X=xxx_test_st)
    from sklearn.metrics import r2_score
    r2_train = r2_score(y_true=yyy_train, y_pred=y_hat_train)
    r2_test = r2_score(y_true=yyy_test, y_pred=y_hat_test)
    from scipy.stats import kendalltau
    kt_train = kendalltau(x=yyy_train, y=y_hat_train).statistic
    kt_test = kendalltau(x=yyy_test, y=y_hat_test).statistic
    from matplotlib import pyplot
    # fig, ax = pyplot.subplots(1, 2)
    # pandas.DataFrame(data={'error': yyy_train - y_hat_train}).hist(bins=20, ax=ax[0])
    # pandas.DataFrame(data={'error': yyy_test - y_hat_test}).hist(bins=20, ax=ax[1])
    r2_trains.append(r2_train)
    r2_tests.append(r2_test)
    kt_trains.append(kt_train)
    kt_tests.append(kt_test)
"""
"""
# LASSO
target = 'TLT_MEAN__pct'
x_factors = ['TLT__rolling_3__None__<function ewm_1shock_relative_12 at 0x000001E037385990>',
       'TLT__rolling_3__None__<function ewm_1shock_relative_3 at 0x000001E037385870>',
       'TLT__pct__full_binners_20_perc', 'TLT__pct',
       'TLT__rolling_3__None__<function ewm_1shock_relative_6 at 0x000001E037385900>',
       'TLT__rolling_3__None__<function mean_relative at 0x000001E0373855A0>',
       'IIPUSNETIQ__rolling_3__None__<function mean_relative at 0x000001E0373855A0>',
       'T20YIEM__rolling_6__None__<function linreg_relative at 0x000001E037385750>',
       'GEPUPPP__rolling_6__None__<function ewm_1shock_relative_3 at 0x000001E037385870>',
       'IIPUSLIAQ__rolling_12__None__<function ewm_1shock_relative_3 at 0x000001E037385870>',
       'CES3000000008__rolling_3__None__<function n_rate_full at 0x000001E037386320>',
       'GS1M__full_seasonal_remove_mstl_get_seasonal_resid__rolling_6__None__<function rel_to_max at 0x000001E0373863B0>',
       'PCES__rolling_12__None__<function relative_mean at 0x000001E0373865F0>',
       'A824RL1Q225SBEA__full_seasonal_remove_mstl_get_seasonal_resid__rolling_12__None__<function rel_to_max at 0x000001E0373863B0>',
       'AWHAERT__rolling_6__None__<function ewm_1shock_relative_12 at 0x000001E037385990>',
       'SPASTT01BRM657N__rolling_6__None__<function rel_to_max at 0x000001E0373863B0>',
       'AWHAERT__rolling_12__None__<function ewm_3shock_relative_6 at 0x000001E037385A20>',
       'IIPPORTAQ__rolling_6__None__<function ewm_1shock_relative_3 at 0x000001E037385870>',
       'DHUTRC1Q027SBEA__full_seasonal_remove_mstl_get_seasonal_resid__rolling_12__None__<function linear_slope at 0x000001E037386170>',
       'AWHAERT__rolling_12__None__<function rel_to_max at 0x000001E0373863B0>',
       'TLT__rolling_3__None__<function rel_to_max at 0x000001E0373863B0>',
       'TLT__rolling_3__None__<function linreg_relative at 0x000001E037385750>',
       'TLT__rolling_6__None__<function relative_q at 0x000001E0373857E0>',
       'AWHAERT__rolling_12__None__<function ewm_1shock_relative_3 at 0x000001E037385870>',
       'IIPPORTAQ__rolling_3__None__<function mean_relative at 0x000001E0373855A0>',
       'DMOTRC1Q027SBEA__full_seasonal_remove_mstl_get_seasonal_resid__rolling_12__None__<function rel_to_max at 0x000001E0373863B0>',
       'IIPUSNETIQ__rolling_3__None__<function rel_to_max at 0x000001E0373863B0>',
       'AWHAERT__rolling_3__None__<function mean_relative at 0x000001E0373855A0>',
       'TLT__rolling_6__None__<function n_rate_conseq at 0x000001E037386290>',
       'SPASTT01CNM657N__full_seasonal_remove_mstl_get_seasonal_resid__rolling_3__None__<function relative_q at 0x000001E0373857E0>']





time_axis = data.index
time_sub_rate = 0.75
time_sub_replace = False
nt = int(time_axis.shape[0] * time_sub_rate)
n = 100
r2_trains, r2_tests = [], []
kt_trains, kt_tests = [], []
for j in range(n):
    import numpy
    # ix_time = numpy.random.choice(time_axis.values, size=(nt,), replace=time_sub_replace)
    ixes = numpy.random.choice(list(range(time_axis.shape[0] - 1)), size=(nt,), replace=time_sub_replace)
    x_ix_train = time_axis.values[:-1][ixes]
    y_ix_train = time_axis.values[1:][ixes]
    left_ixes = [x for x in list(range(time_axis.shape[0] - 1)) if x not in ixes]
    x_ix_test = time_axis.values[:-1][left_ixes]
    y_ix_test = time_axis.values[1:][left_ixes]

    x_train = data.loc[x_ix_train, x_factors].values
    y_train = data.loc[y_ix_train, target].values
    x_test = data.loc[x_ix_test, x_factors].values
    y_test = data.loc[y_ix_test, target].values

    from sklearn.model_selection import train_test_split
    xxx_train, xxx_test, yyy_train, yyy_test = x_train, x_test, y_train, y_test
    from sklearn.preprocessing import StandardScaler
    sk = StandardScaler()
    xxx_train_st = sk.fit_transform(X=xxx_train)
    xxx_test_st = sk.transform(X=xxx_test)
    from sklearn.linear_model import LinearRegression
    m = LinearRegression()
    m.fit(X=xxx_train_st, y=yyy_train)
    y_hat_train = m.predict(X=xxx_train_st)
    y_hat_test = m.predict(X=xxx_test_st)
    from sklearn.metrics import r2_score
    r2_train = r2_score(y_true=yyy_train, y_pred=y_hat_train)
    r2_test = r2_score(y_true=yyy_test, y_pred=y_hat_test)
    from scipy.stats import kendalltau
    kt_train = kendalltau(x=yyy_train, y=y_hat_train).statistic
    kt_test = kendalltau(x=yyy_test, y=y_hat_test).statistic
    from matplotlib import pyplot
    # fig, ax = pyplot.subplots(1, 2)
    # pandas.DataFrame(data={'error': yyy_train - y_hat_train}).hist(bins=20, ax=ax[0])
    # pandas.DataFrame(data={'error': yyy_test - y_hat_test}).hist(bins=20, ax=ax[1])
    r2_trains.append(r2_train)
    r2_tests.append(r2_test)
    kt_trains.append(kt_train)
    kt_tests.append(kt_test)
"""

"""
# SHAP-BASE
target = 'TLT_MEAN__pct'
x_factors = ['PMSAVE__full_seasonal_remove_mstl_get_seasonal_resid__rolling_3__None__<function linear_r2err at 0x000001E037386200>',
 'DCAFRC1A027NBEA__full_seasonal_remove_mstl_get_seasonal_resid__rolling_6__None__<function ewm_1shock_relative_6 at 0x000001E037385900>',
 'SPASTT01USM657N__full_seasonal_remove_mstl_get_seasonal_resid__rolling_12__None__<function ewm_6shock_relative_12 at 0x000001E037385B40>',
 'RBUSBIS__full_seasonal_remove_mstl_get_seasonal_resid__rolling_12__None__<function ewm_3shock_relative_12 at 0x000001E037385AB0>',
 'DHUTRC1Q027SBEA__full_seasonal_remove_mstl_get_seasonal_resid__rolling_6__None__<function n_rate_full at 0x000001E037386320>',
 'EMVFINCRISES__full_seasonal_remove_mstl_get_seasonal_resid__rolling_12__None__<function ewm_1shock_relative_3 at 0x000001E037385870>',
 'MRTSSM44000USS__full_seasonal_remove_mstl_get_seasonal_resid__rolling_12__None__<function ewm_6shock_relative_12 at 0x000001E037385B40>',
 'PMSAVE__rolling_12__None__<function relative_q at 0x000001E0373857E0>',
 'AWHMAN__full_seasonal_remove_mstl_get_seasonal_resid__rolling_3__None__<function rel_to_max at 0x000001E0373863B0>',
 'GS20__rolling_3__None__<function relative_positive_pct at 0x000001E037386680>',
 'SPASTT01GBM657N__rolling_6__None__<function pct_std at 0x000001E0373864D0>',
 'CUUR0000SEHA__full_seasonal_remove_mstl_get_seasonal_resid__rolling_6__None__<function median_relative at 0x000001E037385630>',
 'GFDEBTN__full_seasonal_remove_mstl_get_seasonal_resid__rolling_12__None__<function relative_mean at 0x000001E0373865F0>',
 'DTBSPCKFM__full_seasonal_remove_mstl_get_seasonal_resid__rolling_3__None__<function linreg_relative at 0x000001E037385750>',
 'CUSR0000SETA02__full_seasonal_remove_mstl_get_seasonal_resid__rolling_3__None__<function linreg_relative at 0x000001E037385750>',
 'BAMLC0A0CMEY__rolling_3__None__<function linear_slope at 0x000001E037386170>',
 'SPASTT01MXM657N__rolling_12__None__<function relative_mean at 0x000001E0373865F0>',
 'MANEMP__full_seasonal_remove_mstl_get_seasonal_resid__rolling_12__None__<function rel_to_min at 0x000001E037386440>',
 'SPASTT01RUM657N__full_binners_20_perc',
 'RBUSBIS__pct__full_binners_20_smile',
 'IIPPORTAQ__full_binners_20_smile',
 'GS10__full_seasonal_remove_mstl_get_seasonal_resid__rolling_3__None__<function ewm_1shock_relative_3 at 0x000001E037385870>',
 'DHUTRC1Q027SBEA__rolling_12__None__<function linear_slope at 0x000001E037386170>',
 'AWHMAN__full_seasonal_remove_mstl_get_seasonal_resid__rolling_6__None__<function ewm_1shock_relative_3 at 0x000001E037385870>',
 'CPF1M__full_seasonal_remove_mstl_get_seasonal_resid__rolling_12__None__<function mean_relative at 0x000001E0373855A0>',
 'SPASTT01EZM657N__full_seasonal_remove_mstl_get_seasonal_resid__rolling_3__None__<function mean_relative at 0x000001E0373855A0>',
 'GS2__full_seasonal_remove_mstl_get_seasonal_resid__rolling_6__None__<function median_relative at 0x000001E037385630>',
 'DMOTRC1Q027SBEA__full_seasonal_remove_mstl_get_seasonal_resid__rolling_12__None__<function ewm_3shock_relative_12 at 0x000001E037385AB0>',
 'DFXARC1M027SBEA__full_seasonal_remove_mstl_get_seasonal_resid__rolling_6__None__<function linreg_relative at 0x000001E037385750>',
 'FPCPITOTLZGUSA__rolling_6__None__<function linear_r2err at 0x000001E037386200>',
 'TLT__pct',
 'GS10__pct',
 'GS3M__full_seasonal_remove_mstl_get_seasonal_resid__rolling_6__None__<function rel_to_max at 0x000001E0373863B0>',
 'INTDSRTRM193N__rolling_6__None__<function pct_std at 0x000001E0373864D0>',
 'RBUSBIS__full_seasonal_remove_mstl_get_seasonal_resid__rolling_3__None__<function ewm_1shock_relative_3 at 0x000001E037385870>',
 'CUUR0000SEHA__full_seasonal_remove_mstl_get_seasonal_resid__rolling_3__None__<function relative_q at 0x000001E0373857E0>',
 'SPASTT01CNM657N__full_seasonal_remove_mstl_get_seasonal_resid__rolling_3__None__<function relative_q at 0x000001E0373857E0>']



time_axis = data.index
time_sub_rate = 0.75
time_sub_replace = False
nt = int(time_axis.shape[0] * time_sub_rate)
n = 100
r2_trains, r2_tests = [], []
kt_trains, kt_tests = [], []
for j in range(n):
    import numpy
    # ix_time = numpy.random.choice(time_axis.values, size=(nt,), replace=time_sub_replace)
    ixes = numpy.random.choice(list(range(time_axis.shape[0] - 1)), size=(nt,), replace=time_sub_replace)
    x_ix_train = time_axis.values[:-1][ixes]
    y_ix_train = time_axis.values[1:][ixes]
    left_ixes = [x for x in list(range(time_axis.shape[0] - 1)) if x not in ixes]
    x_ix_test = time_axis.values[:-1][left_ixes]
    y_ix_test = time_axis.values[1:][left_ixes]

    x_train = data.loc[x_ix_train, x_factors].values
    y_train = data.loc[y_ix_train, target].values
    x_test = data.loc[x_ix_test, x_factors].values
    y_test = data.loc[y_ix_test, target].values

    from sklearn.model_selection import train_test_split
    xxx_train, xxx_test, yyy_train, yyy_test = x_train, x_test, y_train, y_test
    from sklearn.preprocessing import StandardScaler
    sk = StandardScaler()
    xxx_train_st = sk.fit_transform(X=xxx_train)
    xxx_test_st = sk.transform(X=xxx_test)
    from sklearn.linear_model import LinearRegression
    m = LinearRegression()
    m.fit(X=xxx_train_st, y=yyy_train)
    y_hat_train = m.predict(X=xxx_train_st)
    y_hat_test = m.predict(X=xxx_test_st)
    from sklearn.metrics import r2_score
    r2_train = r2_score(y_true=yyy_train, y_pred=y_hat_train)
    r2_test = r2_score(y_true=yyy_test, y_pred=y_hat_test)
    from scipy.stats import kendalltau
    kt_train = kendalltau(x=yyy_train, y=y_hat_train).statistic
    kt_test = kendalltau(x=yyy_test, y=y_hat_test).statistic
    from matplotlib import pyplot
    # fig, ax = pyplot.subplots(1, 2)
    # pandas.DataFrame(data={'error': yyy_train - y_hat_train}).hist(bins=20, ax=ax[0])
    # pandas.DataFrame(data={'error': yyy_test - y_hat_test}).hist(bins=20, ax=ax[1])
    r2_trains.append(r2_train)
    r2_tests.append(r2_test)
    kt_trains.append(kt_train)
    kt_tests.append(kt_test)
"""


"""
# SHAP-BASE  STD
target = 'TLT_MEAN__pct'
x_factors = ['A229RX0__full_seasonal_remove_mstl_get_seasonal_resid__rolling_12__None__<function relative_q at 0x000001E0373857E0>',
 'SPASTT01DEM657N__full_seasonal_remove_mstl_get_seasonal_resid__rolling_6__None__<function ewm_1shock_relative_6 at 0x000001E037385900>',
 'FRGSHPUSM649NCIS__full_seasonal_remove_mstl_get_seasonal_resid__rolling_6__None__<function relative_mean at 0x000001E0373865F0>',
 'SPASTT01BRM657N__full_seasonal_remove_mstl_get_seasonal_resid__rolling_3__None__<function linear_slope at 0x000001E037386170>',
 'DCAFRC1A027NBEA__full_seasonal_remove_mstl_get_seasonal_resid__rolling_6__None__<function ewm_1shock_relative_6 at 0x000001E037385900>',
 'BOPGSTB__full_seasonal_remove_mstl_get_seasonal_resid__rolling_3__None__<function rel_to_min at 0x000001E037386440>',
 'TLT_MEAN__full_seasonal_remove_mstl_get_seasonal_resid__rolling_6__None__<function relative_q at 0x000001E0373857E0>',
 'GS1__full_seasonal_remove_mstl_get_seasonal_resid__rolling_6__None__<function rel_to_max at 0x000001E0373863B0>',
 'SPASTT01BRM657N__full_seasonal_remove_mstl_get_seasonal_resid__rolling_3__None__<function relative_q at 0x000001E0373857E0>',
 'USEPUINDXM__rolling_12__None__<function linear_r2err at 0x000001E037386200>',
 'PCEND__full_seasonal_remove_mstl_get_seasonal_resid__rolling_12__None__<function mean_relative at 0x000001E0373855A0>',
 'UNRATE__full_seasonal_remove_mstl_get_seasonal_resid__rolling_3__None__<function relative_q at 0x000001E0373857E0>',
 'GS3M__full_seasonal_remove_mstl_get_seasonal_resid__rolling_6__None__<function rel_to_max at 0x000001E0373863B0>',
 'CES3000000008__rolling_6__None__<function relative_positive_pct at 0x000001E037386680>',
 'PCU484121484121__full_seasonal_remove_mstl_get_seasonal_resid__rolling_12__None__<function linreg_relative at 0x000001E037385750>',
 'CES3000000008__rolling_6__None__<function n_rate_full at 0x000001E037386320>',
 'DCAFRC1A027NBEA__full_seasonal_remove_mstl_get_seasonal_resid__rolling_3__None__<function ewm_1shock_relative_3 at 0x000001E037385870>',
 'DCAFRC1A027NBEA__full_seasonal_remove_mstl_get_seasonal_resid__rolling_6__None__<function median_relative at 0x000001E037385630>',
 'DCAFRC1A027NBEA__full_seasonal_remove_mstl_get_seasonal_resid__rolling_3__None__<function mean_relative at 0x000001E0373855A0>',
 'SPASTT01BRM657N__rolling_6__None__<function linear_slope at 0x000001E037386170>',
 'CES3000000008__rolling_12__None__<function linear_r2err at 0x000001E037386200>',
 'FDHBFRBN__full_seasonal_remove_mstl_get_seasonal_resid__rolling_6__None__<function linreg_relative at 0x000001E037385750>',
 'CES3000000008__full_seasonal_remove_mstl_get_seasonal_resid__rolling_6__None__<function relative_positive_pct at 0x000001E037386680>',
 'CES3000000008__rolling_3__None__<function linear_slope at 0x000001E037386170>',
 'INTDSRTRM193N__rolling_6__None__<function pct_std at 0x000001E0373864D0>',
 'CUUR0000SETA01__full_seasonal_remove_mstl_get_seasonal_resid__rolling_3__None__<function rel_to_max at 0x000001E0373863B0>',
 'EMVFINCRISES__full_seasonal_remove_mstl_get_seasonal_resid__rolling_6__None__<function ewm_1shock_relative_6 at 0x000001E037385900>',
 'BAA__full_seasonal_remove_mstl_get_seasonal_resid__rolling_3__None__<function rel_to_min at 0x000001E037386440>',
 'PCUOMFGOMFG__full_seasonal_remove_mstl_get_seasonal_resid__rolling_6__None__<function n_rate_conseq at 0x000001E037386290>',
 'INTDSRINM193N__full_seasonal_remove_mstl_get_seasonal_resid__rolling_6__None__<function relative_mean at 0x000001E0373865F0>',
 'TLT__rolling_6__None__<function linreg_relative at 0x000001E037385750>',
 'GS20__rolling_3__None__<function ewm_1shock_relative_6 at 0x000001E037385900>',
 'AWHAERT__rolling_12__None__<function rel_to_max at 0x000001E0373863B0>',
 'IIPUSASSQ__full_seasonal_remove_mstl_get_seasonal_resid__rolling_12__None__<function relative_mean at 0x000001E0373865F0>']



time_axis = data.index
time_sub_rate = 0.75
time_sub_replace = False
nt = int(time_axis.shape[0] * time_sub_rate)
n = 100
r2_trains, r2_tests = [], []
kt_trains, kt_tests = [], []
for j in range(n):
    import numpy
    # ix_time = numpy.random.choice(time_axis.values, size=(nt,), replace=time_sub_replace)
    ixes = numpy.random.choice(list(range(time_axis.shape[0] - 1)), size=(nt,), replace=time_sub_replace)
    x_ix_train = time_axis.values[:-1][ixes]
    y_ix_train = time_axis.values[1:][ixes]
    left_ixes = [x for x in list(range(time_axis.shape[0] - 1)) if x not in ixes]
    x_ix_test = time_axis.values[:-1][left_ixes]
    y_ix_test = time_axis.values[1:][left_ixes]

    x_train = data.loc[x_ix_train, x_factors].values
    y_train = data.loc[y_ix_train, target].values
    x_test = data.loc[x_ix_test, x_factors].values
    y_test = data.loc[y_ix_test, target].values

    from sklearn.model_selection import train_test_split
    xxx_train, xxx_test, yyy_train, yyy_test = x_train, x_test, y_train, y_test
    from sklearn.preprocessing import StandardScaler
    sk = StandardScaler()
    xxx_train_st = sk.fit_transform(X=xxx_train)
    xxx_test_st = sk.transform(X=xxx_test)
    from sklearn.linear_model import LinearRegression
    m = LinearRegression()
    m.fit(X=xxx_train_st, y=yyy_train)
    y_hat_train = m.predict(X=xxx_train_st)
    y_hat_test = m.predict(X=xxx_test_st)
    from sklearn.metrics import r2_score
    r2_train = r2_score(y_true=yyy_train, y_pred=y_hat_train)
    r2_test = r2_score(y_true=yyy_test, y_pred=y_hat_test)
    from scipy.stats import kendalltau
    kt_train = kendalltau(x=yyy_train, y=y_hat_train).statistic
    kt_test = kendalltau(x=yyy_test, y=y_hat_test).statistic
    from matplotlib import pyplot
    # fig, ax = pyplot.subplots(1, 2)
    # pandas.DataFrame(data={'error': yyy_train - y_hat_train}).hist(bins=20, ax=ax[0])
    # pandas.DataFrame(data={'error': yyy_test - y_hat_test}).hist(bins=20, ax=ax[1])
    r2_trains.append(r2_train)
    r2_tests.append(r2_test)
    kt_trains.append(kt_train)
    kt_tests.append(kt_test)
"""

"""
# COALITIONS
target = 'TLT_MEAN__pct'
x_factors = ['PCUOMFGOMFG__rolling_3__None__<function linreg_relative at 0x000001E037385750>',
 'MRTSSM44000USS__rolling_3__None__<function ewm_1shock_relative_12 at 0x000001E037385990>',
 'GS10__full_seasonal_remove_mstl_get_seasonal_resid__rolling_3__None__<function linear_slope at 0x000001E037386170>',
 'STICKCPIM157SFRBATL__full_seasonal_remove_mstl_get_seasonal_resid__rolling_12__None__<function ewm_3shock_relative_6 at 0x000001E037385A20>',
 'SPASTT01BRM657N__full_seasonal_remove_mstl_get_seasonal_resid__rolling_3__None__<function linear_slope at 0x000001E037386170>',
 'IVV__rolling_12__None__<function ewm_1shock_relative_3 at 0x000001E037385870>',
 'MRTSSM44000USS__pct',
 'CPF3M__full_seasonal_remove_mstl_get_seasonal_resid__rolling_12__None__<function ewm_1shock_relative_12 at 0x000001E037385990>',
 'GEPUPPP__rolling_12__None__<function ewm_1shock_relative_3 at 0x000001E037385870>',
 'MRTSSM44000USS__full_seasonal_remove_mstl_get_seasonal_resid__rolling_6__None__<function mean_relative at 0x000001E0373855A0>',
 'AWHAERT__rolling_3__None__<function mean_relative at 0x000001E0373855A0>',
 'RBUSBIS__rolling_3__None__<function ewm_1shock_relative_3 at 0x000001E037385870>',
 'INTDSRTRM193N__rolling_6__None__<function linear_r2err at 0x000001E037386200>',
 'EUEPUINDXM__rolling_6__None__<function linear_r2err at 0x000001E037386200>',
 'GS20__rolling_12__None__<function ewm_1shock_relative_3 at 0x000001E037385870>',
 'EMVFINCRISES__rolling_12__None__<function relative_positive_pct at 0x000001E037386680>',
 'RECPROUSM156N__full_seasonal_remove_mstl_get_seasonal_resid__rolling_12__None__<function rel_to_max at 0x000001E0373863B0>',
 'IIPUSNETIQ__rolling_3__None__<function ewm_1shock_relative_6 at 0x000001E037385900>',
 'CUUR0000SEHA__full_seasonal_remove_mstl_get_seasonal_resid__rolling_6__None__<function n_rate_full at 0x000001E037386320>',
 'GS20__rolling_6__None__<function mean_relative at 0x000001E0373855A0>',
 'PCETRIM12M159SFRBDAL__rolling_3__None__<function relative_mean at 0x000001E0373865F0>',
 'GS10__rolling_12__None__<function n_rate_conseq at 0x000001E037386290>',
 'CES3000000008__rolling_6__None__<function n_rate_full at 0x000001E037386320>',
 'T10YIEM__rolling_6__None__<function mean_relative at 0x000001E0373855A0>',
 'CES3000000008__rolling_12__None__<function ewm_1shock_relative_6 at 0x000001E037385900>',
 'GS20__full_seasonal_remove_mstl_get_seasonal_resid__rolling_3__None__<function linear_slope at 0x000001E037386170>',
 'GFDEBTN__full_seasonal_remove_mstl_get_seasonal_resid__rolling_6__None__<function relative_positive_pct at 0x000001E037386680>',
 'PCUOMFGOMFG__full_seasonal_remove_mstl_get_seasonal_resid__rolling_6__None__<function n_rate_conseq at 0x000001E037386290>',
 'PCEC96__full_seasonal_remove_mstl_get_seasonal_resid',
 'TLT__rolling_3__None__<function relative_positive_pct at 0x000001E037386680>',
 'IRLTLT01USM156N__full_seasonal_remove_mstl_get_seasonal_resid',
 'CUUR0000SEHA__full_seasonal_remove_mstl_get_seasonal_resid__rolling_12__None__<function ewm_1shock_relative_6 at 0x000001E037385900>',
 'GFDEBTN__rolling_12__None__<function linear_slope at 0x000001E037386170>',
 'GS3M__full_seasonal_remove_mstl_get_seasonal_resid__rolling_6__None__<function pct_std at 0x000001E0373864D0>',
 'A824RL1Q225SBEA__rolling_12__None__<function linreg_relative at 0x000001E037385750>',
 'PCUOMFGOMFG__full_seasonal_remove_mstl_get_seasonal_resid__rolling_12__None__<function mean_relative at 0x000001E0373855A0>',
 'SPASTT01MXM657N__rolling_12__None__<function relative_q at 0x000001E0373857E0>',
 'T10YIEM__rolling_3__None__<function relative_positive_pct at 0x000001E037386680>',
 'IIPUSNETIQ__rolling_6__None__<function median_relative at 0x000001E037385630>',
 'TLT_MEAN__rolling_3__None__<function ewm_1shock_relative_3 at 0x000001E037385870>',
 'SPASTT01RUM657N__rolling_12__None__<function relative_mean at 0x000001E0373865F0>',
 'DCAFRC1A027NBEA__full_seasonal_remove_mstl_get_seasonal_resid__rolling_6__None__<function ewm_1shock_relative_3 at 0x000001E037385870>',
 'GS10__rolling_12__None__<function pct_std at 0x000001E0373864D0>',
 'AIRRTMFMD11__full_seasonal_remove_mstl_get_seasonal_resid__rolling_6__None__<function linear_r2err at 0x000001E037386200>',
 'PCU484121484121__rolling_6__None__<function pct_std at 0x000001E0373864D0>',
 'CES3000000008__full_seasonal_remove_mstl_get_seasonal_resid__rolling_6__None__<function relative_positive_pct at 0x000001E037386680>',
 'PCEPI__rolling_12__None__<function n_rate_conseq at 0x000001E037386290>',
 'MRTSSM44000USS__rolling_3__None__<function linreg_relative at 0x000001E037385750>',
 'RBUSBIS__rolling_3__None__<function rel_to_max at 0x000001E0373863B0>',
 'EMVFINCRISES__rolling_12__None__<function ewm_6shock_relative_12 at 0x000001E037385B40>',
 'TLT__rolling_3__None__<function ewm_1shock_relative_12 at 0x000001E037385990>',
 'A824RL1Q225SBEA__rolling_12__None__<function mean_relative at 0x000001E0373855A0>',
 'DJRYRC1A027NBEA__full_seasonal_remove_mstl_get_seasonal_resid__rolling_12__None__<function linear_slope at 0x000001E037386170>',
 'DCAFRC1A027NBEA__full_seasonal_remove_mstl_get_seasonal_resid__rolling_3__None__<function rel_to_min at 0x000001E037386440>',
 'CPIAUCSL__pct__full_binners_20_perc',
 'SPASTT01BRM657N__full_seasonal_remove_mstl_get_seasonal_resid__rolling_3__None__<function rel_to_min at 0x000001E037386440>',
 'SPASTT01TRM657N__full_seasonal_remove_mstl_get_seasonal_resid__rolling_3__None__<function rel_to_min at 0x000001E037386440>',
 'GFDEBTN__full_seasonal_remove_mstl_get_seasonal_resid__rolling_12__None__<function linear_slope at 0x000001E037386170>',
 'DMOTRC1Q027SBEA__full_seasonal_remove_mstl_get_seasonal_resid__rolling_12__None__<function mean_relative at 0x000001E0373855A0>',
 'IC131__full_seasonal_remove_mstl_get_seasonal_resid__rolling_6__None__<function n_rate_full at 0x000001E037386320>',
 'GS10__rolling_3__None__<function mean_relative at 0x000001E0373855A0>',
 'DPHCRC1A027NBEA__full_seasonal_remove_mstl_get_seasonal_resid__rolling_3__None__<function mean_relative at 0x000001E0373855A0>',
 'CUUR0000SETA01__rolling_6__None__<function ewm_1shock_relative_6 at 0x000001E037385900>',
 'T20YIEM__rolling_3__None__<function ewm_1shock_relative_6 at 0x000001E037385900>',
 'TLT__pct__full_binners_20_smile',
 'PCEND__rolling_12__None__<function n_rate_conseq at 0x000001E037386290>',
 'IIPPORTAQ__rolling_12__None__<function linreg_relative at 0x000001E037385750>',
 'RSXFS__rolling_6__None__<function linreg_relative at 0x000001E037385750>',
 'TLT__full_seasonal_remove_mstl_get_seasonal_resid__rolling_6__None__<function rel_to_max at 0x000001E0373863B0>',
 'IRLTLT01USM156N__rolling_6__None__<function relative_q at 0x000001E0373857E0>',
 'GS10__rolling_3__None__<function ewm_1shock_relative_3 at 0x000001E037385870>',
 'GS10__rolling_3__None__<function linear_slope at 0x000001E037386170>',
 'TLT__rolling_6__None__<function median_relative at 0x000001E037385630>',
 'GS20__rolling_3__None__<function linear_slope at 0x000001E037386170>',
 'T20YIEM__rolling_3__None__<function rel_to_max at 0x000001E0373863B0>',
 'CPIAUCSL__pct__full_binners_20_smile',
 'TLT_MEAN__full_seasonal_remove_mstl_get_seasonal_resid__rolling_6__None__<function rel_to_min at 0x000001E037386440>',
 'SPASTT01INM657N__full_seasonal_remove_mstl_get_seasonal_resid__rolling_3__None__<function rel_to_min at 0x000001E037386440>',
 'IIPUSLIAQ__full_seasonal_remove_mstl_get_seasonal_resid__rolling_3__None__<function mean_relative at 0x000001E0373855A0>',
 'MSPUS__rolling_3__None__<function ewm_1shock_relative_3 at 0x000001E037385870>',
 'CUUR0000SETA01__rolling_6__None__<function mean_relative at 0x000001E0373855A0>',
 'SPASTT01KRM657N__rolling_12__None__<function rel_to_max at 0x000001E0373863B0>',
 'DTB3__full_seasonal_remove_mstl_get_seasonal_resid',
 'GPDIC1__rolling_6__None__<function linreg_relative at 0x000001E037385750>',
 'T10YIEM__rolling_3__None__<function linear_slope at 0x000001E037386170>',
 'PMSAVE__rolling_12__None__<function pct_std at 0x000001E0373864D0>',
 'T10YIEM__rolling_3__None__<function ewm_1shock_relative_6 at 0x000001E037385900>',
 'RBUSBIS__full_seasonal_remove_mstl_get_seasonal_resid__rolling_12__None__<function ewm_3shock_relative_12 at 0x000001E037385AB0>',
 'QCNR628BIS__full_seasonal_remove_mstl_get_seasonal_resid__rolling_6__None__<function pct_std at 0x000001E0373864D0>',
 'MRTSSM44000USS__rolling_6__None__<function n_rate_conseq at 0x000001E037386290>',
 'BOPGSTB__rolling_12__None__<function pct_std at 0x000001E0373864D0>',
 'CPIAUCSL__rolling_12__None__<function n_rate_conseq at 0x000001E037386290>',
 'PMSAVE__rolling_12__None__<function relative_q at 0x000001E0373857E0>',
 'JTU5300QUL__full_seasonal_remove_mstl_get_seasonal_resid__rolling_6__None__<function pct_std at 0x000001E0373864D0>',
 'SPASTT01CNM657N__full_seasonal_remove_mstl_get_seasonal_resid__rolling_3__None__<function rel_to_max at 0x000001E0373863B0>',
 'AAA__full_seasonal_remove_mstl_get_seasonal_resid__rolling_6__None__<function relative_mean at 0x000001E0373865F0>',
 'A229RX0__full_seasonal_remove_mstl_get_seasonal_resid__rolling_12__None__<function relative_q at 0x000001E0373857E0>',
 'IIPUSLIAQ__full_seasonal_remove_mstl_get_seasonal_resid__rolling_3__None__<function ewm_1shock_relative_12 at 0x000001E037385990>',
 'IVV_MEAN__full_seasonal_remove_mstl_get_seasonal_resid__rolling_12__None__<function ewm_3shock_relative_12 at 0x000001E037385AB0>',
 'AWHAERT__rolling_12__None__<function linreg_relative at 0x000001E037385750>',
 'CUUR0000SETA01__rolling_6__None__<function median_relative at 0x000001E037385630>',
 'IIPUSLIAQ__rolling_6__None__<function ewm_1shock_relative_3 at 0x000001E037385870>',
 'MRTSSM44112USN__rolling_12__None__<function n_rate_full at 0x000001E037386320>',
 'GS20__rolling_6__None__<function ewm_1shock_relative_6 at 0x000001E037385900>',
 'IIPUSASSQ__full_seasonal_remove_mstl_get_seasonal_resid__rolling_12__None__<function relative_mean at 0x000001E0373865F0>',
 'TLT__rolling_6__None__<function relative_q at 0x000001E0373857E0>',
 'GS10__full_seasonal_remove_mstl_get_seasonal_resid',
 'A091RC1Q027SBEA__full_seasonal_remove_mstl_get_seasonal_resid',
 'TLT_MEAN__rolling_3__None__<function linear_slope at 0x000001E037386170>',
 'RBUSBIS__full_seasonal_remove_mstl_get_seasonal_resid__rolling_12__None__<function ewm_6shock_relative_12 at 0x000001E037385B40>',
 'DNRGRC1M027SBEA__full_seasonal_remove_mstl_get_seasonal_resid__rolling_12__None__<function mean_relative at 0x000001E0373855A0>',
 'AIRRTMFMD11__full_seasonal_remove_mstl_get_seasonal_resid__rolling_6__None__<function pct_std at 0x000001E0373864D0>',
 'W006RC1Q027SBEA__full_binners_20_smile',
 'BAA__full_seasonal_remove_mstl_get_seasonal_resid__rolling_3__None__<function linear_slope at 0x000001E037386170>',
 'CUUR0000SETA01__rolling_3__None__<function ewm_1shock_relative_3 at 0x000001E037385870>',
 'AWHAERT__rolling_3__None__<function relative_q at 0x000001E0373857E0>',
 'SPASTT01RUM657N__rolling_6__None__<function rel_to_max at 0x000001E0373863B0>',
 'PSAVERT__full_seasonal_remove_mstl_get_seasonal_resid__rolling_6__None__<function n_rate_conseq at 0x000001E037386290>',
 'SPASTT01DEM657N__full_seasonal_remove_mstl_get_seasonal_resid__rolling_6__None__<function median_relative at 0x000001E037385630>',
 'DPHCRC1A027NBEA__full_seasonal_remove_mstl_get_seasonal_resid__rolling_12__None__<function ewm_3shock_relative_6 at 0x000001E037385A20>',
 'TLT__rolling_3__None__<function rel_to_max at 0x000001E0373863B0>',
 'DHLCRC1Q027SBEA__rolling_12__None__<function rel_to_min at 0x000001E037386440>',
 'CUUR0000SETA01__rolling_3__None__<function median_relative at 0x000001E037385630>',
 'CUSR0000SETA02__full_binners_20_smile',
 'DTBSPCKFM__full_seasonal_remove_mstl_get_seasonal_resid__rolling_3__None__<function relative_mean at 0x000001E0373865F0>',
 'IRLTLT01USM156N__rolling_6__None__<function rel_to_min at 0x000001E037386440>',
 'IVV__rolling_3__None__<function mean_relative at 0x000001E0373855A0>',
 'DJRYRC1A027NBEA__full_seasonal_remove_mstl_get_seasonal_resid__rolling_6__None__<function n_rate_full at 0x000001E037386320>',
 'W006RC1Q027SBEA__full_seasonal_remove_mstl_get_seasonal_resid__rolling_6__None__<function linear_r2err at 0x000001E037386200>',
 'SPASTT01RUM657N__rolling_6__None__<function relative_positive_pct at 0x000001E037386680>',
 'GS20__rolling_12__None__<function ewm_1shock_relative_6 at 0x000001E037385900>',
 'TLT_MEAN__rolling_12__None__<function pct_std at 0x000001E0373864D0>',
 'BAMLC0A0CMEY__rolling_3__None__<function linear_slope at 0x000001E037386170>',
 'T20YIEM__rolling_6__None__<function rel_to_min at 0x000001E037386440>',
 'CPF1M__full_seasonal_remove_mstl_get_seasonal_resid',
 'TLT__rolling_6__None__<function mean_relative at 0x000001E0373855A0>',
 'GDP__full_seasonal_remove_mstl_get_seasonal_resid__rolling_12__None__<function ewm_3shock_relative_6 at 0x000001E037385A20>',
 'SPASTT01KRM657N',
 'T20YIEM__rolling_6__None__<function n_rate_conseq at 0x000001E037386290>',
 'CUUR0000SETA01__full_seasonal_remove_mstl_get_seasonal_resid__rolling_6__None__<function mean_relative at 0x000001E0373855A0>',
 'AAA__pct',
 'GS20__rolling_6__None__<function pct_std at 0x000001E0373864D0>',
 'PCETRIM12M159SFRBDAL__full_seasonal_remove_mstl_get_seasonal_resid__rolling_12__None__<function n_rate_conseq at 0x000001E037386290>',
 'TLT__rolling_3__None__<function linreg_relative at 0x000001E037385750>',
 'PSAVERT__rolling_12__None__<function relative_q at 0x000001E0373857E0>',
 'INTDSRBRM193N__rolling_6__None__<function linear_r2err at 0x000001E037386200>',
 'DJRYRC1A027NBEA__full_seasonal_remove_mstl_get_seasonal_resid__rolling_3__None__<function median_relative at 0x000001E037385630>',
 'PCEC96__full_seasonal_remove_mstl_get_seasonal_resid__rolling_6__None__<function relative_q at 0x000001E0373857E0>',
 'PCEDG__rolling_6__None__<function linear_r2err at 0x000001E037386200>',
 'CUUR0000SETA01__full_seasonal_remove_mstl_get_seasonal_resid__rolling_12__None__<function ewm_3shock_relative_12 at 0x000001E037385AB0>',
 'BAA__rolling_12__None__<function pct_std at 0x000001E0373864D0>',
 'GS20__rolling_12__None__<function rel_to_min at 0x000001E037386440>',
 'MSPUS__full_seasonal_remove_mstl_get_seasonal_resid__rolling_3__None__<function linear_r2err at 0x000001E037386200>',
 'GS10__rolling_6__None__<function ewm_1shock_relative_6 at 0x000001E037385900>',
 'AAA__rolling_3__None__<function relative_positive_pct at 0x000001E037386680>',
 'MRTSSM44000USS__rolling_3__None__<function median_relative at 0x000001E037385630>',
 'CE16OV__full_seasonal_remove_mstl_get_seasonal_resid__rolling_12__None__<function n_rate_full at 0x000001E037386320>',
 'A091RC1Q027SBEA__full_seasonal_remove_mstl_get_seasonal_resid__rolling_12__None__<function median_relative at 0x000001E037385630>',
 'MRTSSM44000USS__rolling_3__None__<function ewm_1shock_relative_6 at 0x000001E037385900>',
 'CUUR0000SETA01__rolling_6__None__<function rel_to_max at 0x000001E0373863B0>',
 'GS20__rolling_6__None__<function ewm_1shock_relative_12 at 0x000001E037385990>',
 'DCAFRC1A027NBEA__full_seasonal_remove_mstl_get_seasonal_resid__rolling_6__None__<function ewm_1shock_relative_6 at 0x000001E037385900>',
 'MRTSSM44000USS__rolling_12__None__<function pct_std at 0x000001E0373864D0>',
 'SPASTT01GBM657N__full_seasonal_remove_mstl_get_seasonal_resid__rolling_12__None__<function ewm_1shock_relative_12 at 0x000001E037385990>',
 'INTDSRUSM193N__full_seasonal_remove_mstl_get_seasonal_resid',
 'IVV_MEAN__full_seasonal_remove_mstl_get_seasonal_resid__rolling_12__None__<function ewm_6shock_relative_12 at 0x000001E037385B40>',
 'DPHCRC1A027NBEA__rolling_6__None__<function linreg_relative at 0x000001E037385750>',
 'IIPUSLIAQ__full_seasonal_remove_mstl_get_seasonal_resid__rolling_3__None__<function rel_to_max at 0x000001E0373863B0>',
 'GS20__rolling_3__None__<function linreg_relative at 0x000001E037385750>',
 'CUUR0000SETA01__full_seasonal_remove_mstl_get_seasonal_resid__rolling_12__None__<function ewm_1shock_relative_12 at 0x000001E037385990>',
 'GEPUPPP__rolling_6__None__<function rel_to_min at 0x000001E037386440>',
 'TLT__rolling_12__None__<function n_rate_conseq at 0x000001E037386290>',
 'GEPUPPP__rolling_12__None__<function ewm_1shock_relative_12 at 0x000001E037385990>',
 'T10YIEM__rolling_6__None__<function linreg_relative at 0x000001E037385750>',
 'CUUR0000SETA01__full_seasonal_remove_mstl_get_seasonal_resid__rolling_3__None__<function rel_to_max at 0x000001E0373863B0>',
 'MRTSSM44000USS__pct__full_binners_20_smile',
 'GS10__rolling_3__None__<function linreg_relative at 0x000001E037385750>',
 'USEPUINDXM__rolling_12__None__<function linear_r2err at 0x000001E037386200>',
 'CUSR0000SETA02__full_seasonal_remove_mstl_get_seasonal_resid__rolling_3__None__<function linreg_relative at 0x000001E037385750>',
 'BAA__full_seasonal_remove_mstl_get_seasonal_resid__rolling_6__None__<function relative_q at 0x000001E0373857E0>',
 'CPF1M__full_seasonal_remove_mstl_get_seasonal_resid__rolling_12__None__<function ewm_3shock_relative_12 at 0x000001E037385AB0>',
 'GDP__rolling_12__None__<function rel_to_min at 0x000001E037386440>',
 'PSAVERT__rolling_12__None__<function pct_std at 0x000001E0373864D0>',
 'IIPUSLIAQ__rolling_6__None__<function rel_to_max at 0x000001E0373863B0>',
 'GEPUPPP__full_seasonal_remove_mstl_get_seasonal_resid__rolling_12__None__<function pct_std at 0x000001E0373864D0>',
 'GPDIC1__full_seasonal_remove_mstl_get_seasonal_resid__rolling_6__None__<function n_rate_full at 0x000001E037386320>',
 'SPASTT01ZAM657N__rolling_12__None__<function pct_std at 0x000001E0373864D0>',
 'DMOTRC1Q027SBEA__rolling_12__None__<function ewm_3shock_relative_12 at 0x000001E037385AB0>',
 'RBUSBIS__rolling_3__None__<function mean_relative at 0x000001E0373855A0>',
 'PCEND__rolling_3__None__<function rel_to_min at 0x000001E037386440>',
 'MRTSSM44000USS__rolling_3__None__<function mean_relative at 0x000001E0373855A0>',
 'PCEND__rolling_6__None__<function n_rate_conseq at 0x000001E037386290>',
 'UNRATE__full_seasonal_remove_mstl_get_seasonal_resid__rolling_3__None__<function linear_slope at 0x000001E037386170>',
 'T20YIEM__pct',
 'INTDSRBRM193N__full_seasonal_remove_mstl_get_seasonal_resid__rolling_6__None__<function rel_to_min at 0x000001E037386440>',
 'PCEPILFE__rolling_12__None__<function pct_std at 0x000001E0373864D0>',
 'SPASTT01AUM657N__full_seasonal_remove_mstl_get_seasonal_resid__rolling_12__None__<function ewm_6shock_relative_12 at 0x000001E037385B40>',
 'AAA__rolling_6__None__<function pct_std at 0x000001E0373864D0>',
 'CSUSHPINSA__rolling_12__None__<function mean_relative at 0x000001E0373855A0>',
 'GS20__rolling_3__None__<function n_rate_full at 0x000001E037386320>',
 'IIPUSNETIQ__rolling_6__None__<function relative_q at 0x000001E0373857E0>',
 'IIPUSNETIQ__rolling_6__None__<function rel_to_min at 0x000001E037386440>',
 'DCAFRC1A027NBEA__full_seasonal_remove_mstl_get_seasonal_resid__rolling_6__None__<function n_rate_full at 0x000001E037386320>',
 'DCAFRC1A027NBEA__full_seasonal_remove_mstl_get_seasonal_resid__rolling_3__None__<function ewm_1shock_relative_12 at 0x000001E037385990>',
 'USACPIALLMINMEI',
 'IRLTLT01USM156N__full_seasonal_remove_mstl_get_seasonal_resid__rolling_3__None__<function linear_slope at 0x000001E037386170>',
 'PCEPI__rolling_3__None__<function linreg_relative at 0x000001E037385750>',
 'SPASTT01KRM657N__rolling_3__None__<function linear_slope at 0x000001E037386170>',
 'SPASTT01RUM657N__rolling_12__None__<function n_rate_full at 0x000001E037386320>',
 'IRLTLT01USM156N__rolling_6__None__<function n_rate_conseq at 0x000001E037386290>',
 'PCEND__full_seasonal_remove_mstl_get_seasonal_resid__rolling_12__None__<function mean_relative at 0x000001E0373855A0>',
 'BAA__rolling_6__None__<function pct_std at 0x000001E0373864D0>',
 'IVV_MEAN__rolling_6__None__<function linreg_relative at 0x000001E037385750>',
 'IIPUSNETIQ__rolling_12__None__<function ewm_1shock_relative_12 at 0x000001E037385990>',
 'CUUR0000SETA01__pct__full_binners_20_perc',
 'SPASTT01KRM657N__rolling_6__None__<function linear_slope at 0x000001E037386170>',
 'TLT__full_seasonal_remove_mstl_get_seasonal_resid__rolling_3__None__<function linear_slope at 0x000001E037386170>',
 'SPASTT01INM657N__full_seasonal_remove_mstl_get_seasonal_resid__rolling_12__None__<function relative_positive_pct at 0x000001E037386680>',
 'IIPUSLIAQ__rolling_3__None__<function ewm_1shock_relative_6 at 0x000001E037385900>',
 'PCEND__rolling_6__None__<function pct_std at 0x000001E0373864D0>',
 'AIRRTMFMD11__rolling_6__None__<function n_rate_conseq at 0x000001E037386290>',
 'SPASTT01USM657N__full_seasonal_remove_mstl_get_seasonal_resid__rolling_12__None__<function median_relative at 0x000001E037385630>',
 'FEDFUNDS__rolling_12__None__<function linreg_relative at 0x000001E037385750>',
 'T10YIEM__pct',
 'QUSR628BIS__rolling_6__None__<function relative_mean at 0x000001E0373865F0>',
 'GS10__full_seasonal_remove_mstl_get_seasonal_resid__rolling_3__None__<function rel_to_max at 0x000001E0373863B0>',
 'MRTSSM44000USS__rolling_3__None__<function rel_to_min at 0x000001E037386440>',
 'RSXFS__full_seasonal_remove_mstl_get_seasonal_resid__rolling_12__None__<function ewm_3shock_relative_12 at 0x000001E037385AB0>',
 'PCEPI__full_seasonal_remove_mstl_get_seasonal_resid__rolling_12__None__<function ewm_6shock_relative_12 at 0x000001E037385B40>',
 'GEPUPPP__rolling_6__None__<function mean_relative at 0x000001E0373855A0>',
 'GS20__rolling_3__None__<function median_relative at 0x000001E037385630>',
 'FRGSHPUSM649NCIS__full_seasonal_remove_mstl_get_seasonal_resid__rolling_12__None__<function ewm_3shock_relative_12 at 0x000001E037385AB0>',
 'RBUSBIS__rolling_3__None__<function relative_q at 0x000001E0373857E0>',
 'IRLTLT01USM156N__rolling_3__None__<function median_relative at 0x000001E037385630>',
 'TLT_MEAN__rolling_6__None__<function ewm_1shock_relative_6 at 0x000001E037385900>',
 'BAA__full_seasonal_remove_mstl_get_seasonal_resid__rolling_3__None__<function rel_to_min at 0x000001E037386440>',
 'T20YIEM__rolling_3__None__<function rel_to_min at 0x000001E037386440>',
 'PCUOMFGOMFG__rolling_6__None__<function n_rate_conseq at 0x000001E037386290>',
 'PCEPI__full_seasonal_remove_mstl_get_seasonal_resid__rolling_6__None__<function relative_positive_pct at 0x000001E037386680>',
 'IIPPORTAQ__full_seasonal_remove_mstl_get_seasonal_resid__rolling_12__None__<function ewm_3shock_relative_6 at 0x000001E037385A20>',
 'SPASTT01ZAM657N__full_binners_20_smile',
 'IIPUSNETIQ__rolling_12__None__<function linreg_relative at 0x000001E037385750>',
 'T20YIEM__rolling_3__None__<function mean_relative at 0x000001E0373855A0>',
 'RBUSBIS__pct',
 'SPASTT01KRM657N__full_binners_20_perc',
 'CPF3M__full_seasonal_remove_mstl_get_seasonal_resid__rolling_12__None__<function ewm_6shock_relative_12 at 0x000001E037385B40>',
 'AAA__full_seasonal_remove_mstl_get_seasonal_resid__rolling_3__None__<function rel_to_max at 0x000001E0373863B0>',
 'RBUSBIS__pct__full_binners_20_perc',
 'TLT__pct',
 'CES3000000008__rolling_6__None__<function relative_positive_pct at 0x000001E037386680>',
 'TLT_MEAN__rolling_12__None__<function linreg_relative at 0x000001E037385750>',
 'AAA__rolling_6__None__<function rel_to_min at 0x000001E037386440>',
 'AWHAERT__rolling_3__None__<function linear_slope at 0x000001E037386170>',
 'USACPIALLMINMEI__full_binners_20_perc',
 'CES3000000008__rolling_12__None__<function ewm_3shock_relative_6 at 0x000001E037385A20>',
 'STICKCPIM157SFRBATL__full_seasonal_remove_mstl_get_seasonal_resid__rolling_12__None__<function ewm_3shock_relative_12 at 0x000001E037385AB0>',
 'IC131__full_seasonal_remove_mstl_get_seasonal_resid__rolling_12__None__<function linear_r2err at 0x000001E037386200>',
 'DMOTRC1Q027SBEA__full_seasonal_remove_mstl_get_seasonal_resid__rolling_12__None__<function ewm_3shock_relative_12 at 0x000001E037385AB0>',
 'PCEPI__pct',
 'IIPUSLIAQ__rolling_3__None__<function ewm_1shock_relative_12 at 0x000001E037385990>',
 'INTDSRUSM193N__full_binners_20_perc',
 'RBUSBIS__rolling_3__None__<function ewm_1shock_relative_12 at 0x000001E037385990>']


time_axis = data.index
time_sub_rate = 0.75
time_sub_replace = False
nt = int(time_axis.shape[0] * time_sub_rate)
n = 100
r2_trains, r2_tests = [], []
kt_trains, kt_tests = [], []
for j in range(n):
    import numpy
    # ix_time = numpy.random.choice(time_axis.values, size=(nt,), replace=time_sub_replace)
    ixes = numpy.random.choice(list(range(time_axis.shape[0] - 1)), size=(nt,), replace=time_sub_replace)
    x_ix_train = time_axis.values[:-1][ixes]
    y_ix_train = time_axis.values[1:][ixes]
    left_ixes = [x for x in list(range(time_axis.shape[0] - 1)) if x not in ixes]
    x_ix_test = time_axis.values[:-1][left_ixes]
    y_ix_test = time_axis.values[1:][left_ixes]

    x_train = data.loc[x_ix_train, x_factors].values
    y_train = data.loc[y_ix_train, target].values
    x_test = data.loc[x_ix_test, x_factors].values
    y_test = data.loc[y_ix_test, target].values

    from sklearn.model_selection import train_test_split
    xxx_train, xxx_test, yyy_train, yyy_test = x_train, x_test, y_train, y_test
    from sklearn.preprocessing import StandardScaler
    sk = StandardScaler()
    xxx_train_st = sk.fit_transform(X=xxx_train)
    xxx_test_st = sk.transform(X=xxx_test)
    from sklearn.linear_model import LinearRegression
    m = LinearRegression()
    m.fit(X=xxx_train_st, y=yyy_train)
    y_hat_train = m.predict(X=xxx_train_st)
    y_hat_test = m.predict(X=xxx_test_st)
    from sklearn.metrics import r2_score
    r2_train = r2_score(y_true=yyy_train, y_pred=y_hat_train)
    r2_test = r2_score(y_true=yyy_test, y_pred=y_hat_test)
    from scipy.stats import kendalltau
    kt_train = kendalltau(x=yyy_train, y=y_hat_train).statistic
    kt_test = kendalltau(x=yyy_test, y=y_hat_test).statistic
    from matplotlib import pyplot
    # fig, ax = pyplot.subplots(1, 2)
    # pandas.DataFrame(data={'error': yyy_train - y_hat_train}).hist(bins=20, ax=ax[0])
    # pandas.DataFrame(data={'error': yyy_test - y_hat_test}).hist(bins=20, ax=ax[1])
    r2_trains.append(r2_train)
    r2_tests.append(r2_test)
    kt_trains.append(kt_train)
    kt_tests.append(kt_test)
"""

"""
# DOWNCAST
target = 'TLT_MEAN__pct'
x_factors = ['AWHMAN__full_seasonal_remove_mstl_get_seasonal_resid__rolling_6__None__<function pct_std at 0x000001E0373864D0>',
       'MSPUS__rolling_3__None__<function linear_slope at 0x000001E037386170>',
       'GFDEBTN__rolling_12__None__<function linear_slope at 0x000001E037386170>',
       'IIPUSNETIQ__rolling_6__None__<function linear_slope at 0x000001E037386170>',
       'IIPUSLIAQ__full_seasonal_remove_mstl_get_seasonal_resid__rolling_3__None__<function linear_slope at 0x000001E037386170>']


time_axis = data.index
time_sub_rate = 0.75
time_sub_replace = False
nt = int(time_axis.shape[0] * time_sub_rate)
n = 100
r2_trains, r2_tests = [], []
kt_trains, kt_tests = [], []
for j in range(n):
    import numpy
    # ix_time = numpy.random.choice(time_axis.values, size=(nt,), replace=time_sub_replace)
    ixes = numpy.random.choice(list(range(time_axis.shape[0] - 1)), size=(nt,), replace=time_sub_replace)
    x_ix_train = time_axis.values[:-1][ixes]
    y_ix_train = time_axis.values[1:][ixes]
    left_ixes = [x for x in list(range(time_axis.shape[0] - 1)) if x not in ixes]
    x_ix_test = time_axis.values[:-1][left_ixes]
    y_ix_test = time_axis.values[1:][left_ixes]

    x_train = data.loc[x_ix_train, x_factors].values
    y_train = data.loc[y_ix_train, target].values
    x_test = data.loc[x_ix_test, x_factors].values
    y_test = data.loc[y_ix_test, target].values

    from sklearn.model_selection import train_test_split
    xxx_train, xxx_test, yyy_train, yyy_test = x_train, x_test, y_train, y_test
    from sklearn.preprocessing import StandardScaler
    sk = StandardScaler()
    xxx_train_st = sk.fit_transform(X=xxx_train)
    xxx_test_st = sk.transform(X=xxx_test)
    from sklearn.linear_model import LinearRegression
    m = LinearRegression()
    m.fit(X=xxx_train_st, y=yyy_train)
    y_hat_train = m.predict(X=xxx_train_st)
    y_hat_test = m.predict(X=xxx_test_st)
    from sklearn.metrics import r2_score
    r2_train = r2_score(y_true=yyy_train, y_pred=y_hat_train)
    r2_test = r2_score(y_true=yyy_test, y_pred=y_hat_test)
    from scipy.stats import kendalltau
    kt_train = kendalltau(x=yyy_train, y=y_hat_train).statistic
    kt_test = kendalltau(x=yyy_test, y=y_hat_test).statistic
    from matplotlib import pyplot
    # fig, ax = pyplot.subplots(1, 2)
    # pandas.DataFrame(data={'error': yyy_train - y_hat_train}).hist(bins=20, ax=ax[0])
    # pandas.DataFrame(data={'error': yyy_test - y_hat_test}).hist(bins=20, ax=ax[1])
    r2_trains.append(r2_train)
    r2_tests.append(r2_test)
    kt_trains.append(kt_train)
    kt_tests.append(kt_test)
"""


"""
# SHAP-BASE CUTB ASE
target = 'TLT_MEAN__pct'
x_factors = ['PMSAVE__full_seasonal_remove_mstl_get_seasonal_resid__rolling_3__None__<function linear_r2err at 0x000001E037386200>',
 'DCAFRC1A027NBEA__full_seasonal_remove_mstl_get_seasonal_resid__rolling_6__None__<function ewm_1shock_relative_6 at 0x000001E037385900>',
 'SPASTT01USM657N__full_seasonal_remove_mstl_get_seasonal_resid__rolling_12__None__<function ewm_6shock_relative_12 at 0x000001E037385B40>',
 'RBUSBIS__full_seasonal_remove_mstl_get_seasonal_resid__rolling_12__None__<function ewm_3shock_relative_12 at 0x000001E037385AB0>',
 'DHUTRC1Q027SBEA__full_seasonal_remove_mstl_get_seasonal_resid__rolling_6__None__<function n_rate_full at 0x000001E037386320>',
 'EMVFINCRISES__full_seasonal_remove_mstl_get_seasonal_resid__rolling_12__None__<function ewm_1shock_relative_3 at 0x000001E037385870>',
 'MRTSSM44000USS__full_seasonal_remove_mstl_get_seasonal_resid__rolling_12__None__<function ewm_6shock_relative_12 at 0x000001E037385B40>',
 'PMSAVE__rolling_12__None__<function relative_q at 0x000001E0373857E0>',
 'AWHMAN__full_seasonal_remove_mstl_get_seasonal_resid__rolling_3__None__<function rel_to_max at 0x000001E0373863B0>',
 'GS20__rolling_3__None__<function relative_positive_pct at 0x000001E037386680>',
 'SPASTT01GBM657N__rolling_6__None__<function pct_std at 0x000001E0373864D0>',
 'CUUR0000SEHA__full_seasonal_remove_mstl_get_seasonal_resid__rolling_6__None__<function median_relative at 0x000001E037385630>',
 'GFDEBTN__full_seasonal_remove_mstl_get_seasonal_resid__rolling_12__None__<function relative_mean at 0x000001E0373865F0>',
 'DTBSPCKFM__full_seasonal_remove_mstl_get_seasonal_resid__rolling_3__None__<function linreg_relative at 0x000001E037385750>',
 'CUSR0000SETA02__full_seasonal_remove_mstl_get_seasonal_resid__rolling_3__None__<function linreg_relative at 0x000001E037385750>',
 'BAMLC0A0CMEY__rolling_3__None__<function linear_slope at 0x000001E037386170>',
 'SPASTT01MXM657N__rolling_12__None__<function relative_mean at 0x000001E0373865F0>',
 'MANEMP__full_seasonal_remove_mstl_get_seasonal_resid__rolling_12__None__<function rel_to_min at 0x000001E037386440>',
 'SPASTT01RUM657N__full_binners_20_perc',
 'RBUSBIS__pct__full_binners_20_smile',
 'IIPPORTAQ__full_binners_20_smile',
 'GS10__full_seasonal_remove_mstl_get_seasonal_resid__rolling_3__None__<function ewm_1shock_relative_3 at 0x000001E037385870>',
 'DHUTRC1Q027SBEA__rolling_12__None__<function linear_slope at 0x000001E037386170>',
 'AWHMAN__full_seasonal_remove_mstl_get_seasonal_resid__rolling_6__None__<function ewm_1shock_relative_3 at 0x000001E037385870>',
 'CPF1M__full_seasonal_remove_mstl_get_seasonal_resid__rolling_12__None__<function mean_relative at 0x000001E0373855A0>',
 'SPASTT01EZM657N__full_seasonal_remove_mstl_get_seasonal_resid__rolling_3__None__<function mean_relative at 0x000001E0373855A0>',
 'GS2__full_seasonal_remove_mstl_get_seasonal_resid__rolling_6__None__<function median_relative at 0x000001E037385630>',
 'DMOTRC1Q027SBEA__full_seasonal_remove_mstl_get_seasonal_resid__rolling_12__None__<function ewm_3shock_relative_12 at 0x000001E037385AB0>',
 'DFXARC1M027SBEA__full_seasonal_remove_mstl_get_seasonal_resid__rolling_6__None__<function linreg_relative at 0x000001E037385750>',
 'FPCPITOTLZGUSA__rolling_6__None__<function linear_r2err at 0x000001E037386200>',
 'TLT__pct',
 'GS10__pct',
 'GS3M__full_seasonal_remove_mstl_get_seasonal_resid__rolling_6__None__<function rel_to_max at 0x000001E0373863B0>',
 'INTDSRTRM193N__rolling_6__None__<function pct_std at 0x000001E0373864D0>',
 'RBUSBIS__full_seasonal_remove_mstl_get_seasonal_resid__rolling_3__None__<function ewm_1shock_relative_3 at 0x000001E037385870>',
 'CUUR0000SEHA__full_seasonal_remove_mstl_get_seasonal_resid__rolling_3__None__<function relative_q at 0x000001E0373857E0>',
 'SPASTT01CNM657N__full_seasonal_remove_mstl_get_seasonal_resid__rolling_3__None__<function relative_q at 0x000001E0373857E0>']

'''
x_factors_ex = ['DTBSPCKFM__full_seasonal_remove_mstl_get_seasonal_resid__rolling_3__None__<function linreg_relative at 0x000001E037385750>',
       'AWHMAN__full_seasonal_remove_mstl_get_seasonal_resid__rolling_6__None__<function ewm_1shock_relative_3 at 0x000001E037385870>',
       'SPASTT01RUM657N__full_binners_20_perc',
       'CUSR0000SETA02__full_seasonal_remove_mstl_get_seasonal_resid__rolling_3__None__<function linreg_relative at 0x000001E037385750>',
       'DFXARC1M027SBEA__full_seasonal_remove_mstl_get_seasonal_resid__rolling_6__None__<function linreg_relative at 0x000001E037385750>',
       'EMVFINCRISES__full_seasonal_remove_mstl_get_seasonal_resid__rolling_12__None__<function ewm_1shock_relative_3 at 0x000001E037385870>']
'''

x_factors_ex = ['GS20__rolling_3__None__<function relative_positive_pct at 0x000001E037386680>',
       'DTBSPCKFM__full_seasonal_remove_mstl_get_seasonal_resid__rolling_3__None__<function linreg_relative at 0x000001E037385750>',
       'AWHMAN__full_seasonal_remove_mstl_get_seasonal_resid__rolling_6__None__<function ewm_1shock_relative_3 at 0x000001E037385870>',
       'SPASTT01RUM657N__full_binners_20_perc',
       'CUSR0000SETA02__full_seasonal_remove_mstl_get_seasonal_resid__rolling_3__None__<function linreg_relative at 0x000001E037385750>',
       'EMVFINCRISES__full_seasonal_remove_mstl_get_seasonal_resid__rolling_12__None__<function ewm_1shock_relative_3 at 0x000001E037385870>',
       'DFXARC1M027SBEA__full_seasonal_remove_mstl_get_seasonal_resid__rolling_6__None__<function linreg_relative at 0x000001E037385750>',
       'SPASTT01USM657N__full_seasonal_remove_mstl_get_seasonal_resid__rolling_12__None__<function ewm_6shock_relative_12 at 0x000001E037385B40>',
       'RBUSBIS__full_seasonal_remove_mstl_get_seasonal_resid__rolling_12__None__<function ewm_3shock_relative_12 at 0x000001E037385AB0>',
       'MRTSSM44000USS__full_seasonal_remove_mstl_get_seasonal_resid__rolling_12__None__<function ewm_6shock_relative_12 at 0x000001E037385B40>',
       'MANEMP__full_seasonal_remove_mstl_get_seasonal_resid__rolling_12__None__<function rel_to_min at 0x000001E037386440>',
       'DHUTRC1Q027SBEA__full_seasonal_remove_mstl_get_seasonal_resid__rolling_6__None__<function n_rate_full at 0x000001E037386320>',
       'CPF1M__full_seasonal_remove_mstl_get_seasonal_resid__rolling_12__None__<function mean_relative at 0x000001E0373855A0>',
       'DHUTRC1Q027SBEA__rolling_12__None__<function linear_slope at 0x000001E037386170>']    
x_factors = [x for x in x_factors if x not in x_factors_ex]


time_axis = data.index
time_sub_rate = 0.75
time_sub_replace = False
nt = int(time_axis.shape[0] * time_sub_rate)
n = 100
r2_trains, r2_tests = [], []
kt_trains, kt_tests = [], []



for j in range(n):
    import numpy
    # ix_time = numpy.random.choice(time_axis.values, size=(nt,), replace=time_sub_replace)
    ixes = numpy.random.choice(list(range(time_axis.shape[0] - 1)), size=(nt,), replace=time_sub_replace)
    x_ix_train = time_axis.values[:-1][ixes]
    y_ix_train = time_axis.values[1:][ixes]
    left_ixes = [x for x in list(range(time_axis.shape[0] - 1)) if x not in ixes]
    x_ix_test = time_axis.values[:-1][left_ixes]
    y_ix_test = time_axis.values[1:][left_ixes]

    x_train = data.loc[x_ix_train, x_factors].values
    y_train = data.loc[y_ix_train, target].values
    x_test = data.loc[x_ix_test, x_factors].values
    y_test = data.loc[y_ix_test, target].values

    from sklearn.model_selection import train_test_split
    xxx_train, xxx_test, yyy_train, yyy_test = x_train, x_test, y_train, y_test
    from sklearn.preprocessing import StandardScaler
    sk = StandardScaler()
    xxx_train_st = sk.fit_transform(X=xxx_train)
    xxx_test_st = sk.transform(X=xxx_test)
    from sklearn.linear_model import LinearRegression
    m = LinearRegression()
    m.fit(X=xxx_train_st, y=yyy_train)
    y_hat_train = m.predict(X=xxx_train_st)
    y_hat_test = m.predict(X=xxx_test_st)
    from sklearn.metrics import r2_score
    r2_train = r2_score(y_true=yyy_train, y_pred=y_hat_train)
    r2_test = r2_score(y_true=yyy_test, y_pred=y_hat_test)
    from scipy.stats import kendalltau
    kt_train = kendalltau(x=yyy_train, y=y_hat_train).statistic
    kt_test = kendalltau(x=yyy_test, y=y_hat_test).statistic
    from matplotlib import pyplot
    # fig, ax = pyplot.subplots(1, 2)
    # pandas.DataFrame(data={'error': yyy_train - y_hat_train}).hist(bins=20, ax=ax[0])
    # pandas.DataFrame(data={'error': yyy_test - y_hat_test}).hist(bins=20, ax=ax[1])
    r2_trains.append(r2_train)
    r2_tests.append(r2_test)
    kt_trains.append(kt_train)
    kt_tests.append(kt_test)
"""

"""
# SHAP-BASE  STD
target = 'TLT_MEAN__pct'
x_factors = ['A229RX0__full_seasonal_remove_mstl_get_seasonal_resid__rolling_12__None__<function relative_q at 0x000001E0373857E0>',
 'SPASTT01DEM657N__full_seasonal_remove_mstl_get_seasonal_resid__rolling_6__None__<function ewm_1shock_relative_6 at 0x000001E037385900>',
 'FRGSHPUSM649NCIS__full_seasonal_remove_mstl_get_seasonal_resid__rolling_6__None__<function relative_mean at 0x000001E0373865F0>',
 'SPASTT01BRM657N__full_seasonal_remove_mstl_get_seasonal_resid__rolling_3__None__<function linear_slope at 0x000001E037386170>',
 'DCAFRC1A027NBEA__full_seasonal_remove_mstl_get_seasonal_resid__rolling_6__None__<function ewm_1shock_relative_6 at 0x000001E037385900>',
 'BOPGSTB__full_seasonal_remove_mstl_get_seasonal_resid__rolling_3__None__<function rel_to_min at 0x000001E037386440>',
 'TLT_MEAN__full_seasonal_remove_mstl_get_seasonal_resid__rolling_6__None__<function relative_q at 0x000001E0373857E0>',
 'GS1__full_seasonal_remove_mstl_get_seasonal_resid__rolling_6__None__<function rel_to_max at 0x000001E0373863B0>',
 'SPASTT01BRM657N__full_seasonal_remove_mstl_get_seasonal_resid__rolling_3__None__<function relative_q at 0x000001E0373857E0>',
 'USEPUINDXM__rolling_12__None__<function linear_r2err at 0x000001E037386200>',
 'PCEND__full_seasonal_remove_mstl_get_seasonal_resid__rolling_12__None__<function mean_relative at 0x000001E0373855A0>',
 'UNRATE__full_seasonal_remove_mstl_get_seasonal_resid__rolling_3__None__<function relative_q at 0x000001E0373857E0>',
 'GS3M__full_seasonal_remove_mstl_get_seasonal_resid__rolling_6__None__<function rel_to_max at 0x000001E0373863B0>',
 'CES3000000008__rolling_6__None__<function relative_positive_pct at 0x000001E037386680>',
 'PCU484121484121__full_seasonal_remove_mstl_get_seasonal_resid__rolling_12__None__<function linreg_relative at 0x000001E037385750>',
 'CES3000000008__rolling_6__None__<function n_rate_full at 0x000001E037386320>',
 'DCAFRC1A027NBEA__full_seasonal_remove_mstl_get_seasonal_resid__rolling_3__None__<function ewm_1shock_relative_3 at 0x000001E037385870>',
 'DCAFRC1A027NBEA__full_seasonal_remove_mstl_get_seasonal_resid__rolling_6__None__<function median_relative at 0x000001E037385630>',
 'DCAFRC1A027NBEA__full_seasonal_remove_mstl_get_seasonal_resid__rolling_3__None__<function mean_relative at 0x000001E0373855A0>',
 'SPASTT01BRM657N__rolling_6__None__<function linear_slope at 0x000001E037386170>',
 'CES3000000008__rolling_12__None__<function linear_r2err at 0x000001E037386200>',
 'FDHBFRBN__full_seasonal_remove_mstl_get_seasonal_resid__rolling_6__None__<function linreg_relative at 0x000001E037385750>',
 'CES3000000008__full_seasonal_remove_mstl_get_seasonal_resid__rolling_6__None__<function relative_positive_pct at 0x000001E037386680>',
 'CES3000000008__rolling_3__None__<function linear_slope at 0x000001E037386170>',
 'INTDSRTRM193N__rolling_6__None__<function pct_std at 0x000001E0373864D0>',
 'CUUR0000SETA01__full_seasonal_remove_mstl_get_seasonal_resid__rolling_3__None__<function rel_to_max at 0x000001E0373863B0>',
 'EMVFINCRISES__full_seasonal_remove_mstl_get_seasonal_resid__rolling_6__None__<function ewm_1shock_relative_6 at 0x000001E037385900>',
 'BAA__full_seasonal_remove_mstl_get_seasonal_resid__rolling_3__None__<function rel_to_min at 0x000001E037386440>',
 'PCUOMFGOMFG__full_seasonal_remove_mstl_get_seasonal_resid__rolling_6__None__<function n_rate_conseq at 0x000001E037386290>',
 'INTDSRINM193N__full_seasonal_remove_mstl_get_seasonal_resid__rolling_6__None__<function relative_mean at 0x000001E0373865F0>',
 'TLT__rolling_6__None__<function linreg_relative at 0x000001E037385750>',
 'GS20__rolling_3__None__<function ewm_1shock_relative_6 at 0x000001E037385900>',
 'AWHAERT__rolling_12__None__<function rel_to_max at 0x000001E0373863B0>',
 'IIPUSASSQ__full_seasonal_remove_mstl_get_seasonal_resid__rolling_12__None__<function relative_mean at 0x000001E0373865F0>']

x_factors_ex = ['BOPGSTB__full_seasonal_remove_mstl_get_seasonal_resid__rolling_3__None__<function rel_to_min at 0x000001E037386440>',
       'CES3000000008__rolling_6__None__<function relative_positive_pct at 0x000001E037386680>',
       'DCAFRC1A027NBEA__full_seasonal_remove_mstl_get_seasonal_resid__rolling_6__None__<function ewm_1shock_relative_6 at 0x000001E037385900>',
       'SPASTT01BRM657N__rolling_6__None__<function linear_slope at 0x000001E037386170>',
       'PCEND__full_seasonal_remove_mstl_get_seasonal_resid__rolling_12__None__<function mean_relative at 0x000001E0373855A0>',
       'CES3000000008__rolling_3__None__<function linear_slope at 0x000001E037386170>',
       'SPASTT01BRM657N__full_seasonal_remove_mstl_get_seasonal_resid__rolling_3__None__<function linear_slope at 0x000001E037386170>',
       'CUUR0000SETA01__full_seasonal_remove_mstl_get_seasonal_resid__rolling_3__None__<function rel_to_max at 0x000001E0373863B0>']
x_factors = [x for x in x_factors if x not in x_factors_ex]




time_axis = data.index
time_sub_rate = 0.75
time_sub_replace = False
nt = int(time_axis.shape[0] * time_sub_rate)
n = 100
r2_trains, r2_tests = [], []
kt_trains, kt_tests = [], []
for j in range(n):
    import numpy
    # ix_time = numpy.random.choice(time_axis.values, size=(nt,), replace=time_sub_replace)
    ixes = numpy.random.choice(list(range(time_axis.shape[0] - 1)), size=(nt,), replace=time_sub_replace)
    x_ix_train = time_axis.values[:-1][ixes]
    y_ix_train = time_axis.values[1:][ixes]
    left_ixes = [x for x in list(range(time_axis.shape[0] - 1)) if x not in ixes]
    x_ix_test = time_axis.values[:-1][left_ixes]
    y_ix_test = time_axis.values[1:][left_ixes]

    x_train = data.loc[x_ix_train, x_factors].values
    y_train = data.loc[y_ix_train, target].values
    x_test = data.loc[x_ix_test, x_factors].values
    y_test = data.loc[y_ix_test, target].values

    from sklearn.model_selection import train_test_split
    xxx_train, xxx_test, yyy_train, yyy_test = x_train, x_test, y_train, y_test
    from sklearn.preprocessing import StandardScaler
    sk = StandardScaler()
    xxx_train_st = sk.fit_transform(X=xxx_train)
    xxx_test_st = sk.transform(X=xxx_test)
    from sklearn.linear_model import LinearRegression
    m = LinearRegression()
    m.fit(X=xxx_train_st, y=yyy_train)
    y_hat_train = m.predict(X=xxx_train_st)
    y_hat_test = m.predict(X=xxx_test_st)
    from sklearn.metrics import r2_score
    r2_train = r2_score(y_true=yyy_train, y_pred=y_hat_train)
    r2_test = r2_score(y_true=yyy_test, y_pred=y_hat_test)
    from scipy.stats import kendalltau
    kt_train = kendalltau(x=yyy_train, y=y_hat_train).statistic
    kt_test = kendalltau(x=yyy_test, y=y_hat_test).statistic
    from matplotlib import pyplot
    # fig, ax = pyplot.subplots(1, 2)
    # pandas.DataFrame(data={'error': yyy_train - y_hat_train}).hist(bins=20, ax=ax[0])
    # pandas.DataFrame(data={'error': yyy_test - y_hat_test}).hist(bins=20, ax=ax[1])
    r2_trains.append(r2_train)
    r2_tests.append(r2_test)
    kt_trains.append(kt_train)
    kt_tests.append(kt_test)

"""
