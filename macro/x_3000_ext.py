



class betaUGMARIMAClass:
    def __init__(self, model, model_kwargs, window, forward=1, log=True, pca=False):
        self._model = model
        self.model_kwargs = model_kwargs
        self.model = None
        self.window = window
        self.forward = forward
        self.log = log
        self.pca = pca
        self.value_type = ValueTypes.CONTINUOUS
        self.impute_max = numpy.nan
        self.impute_min = numpy.nan
        self.y_project_first_series = None
    @property
    def parametrization(self):
        dictated = tuple([hash('UGMARIMAClass'),
                          hash(self._model),
                          hash(self.model_kwargs.values()),
                          hash(str(self.window)),
                          hash(str(self.forward)),
                          hash(str(self.log)),
                          hash(str(self.pca)),
                          hash(self.value_type)])
        return dictated
    @property
    def parametrization_hash(self):
        hashed = hash(self.parametrization)
        return hashed
    def project_first(self, series_dict):
        run_time = time.time()
        assert len(list(series_dict.keys())) == 1
        result = series_dict[list(series_dict.keys())[0]].copy()
        name = result.name
        self.y_project_first_series = result.copy()
        if self.log:
            result = result.pct_change()
            result = (result + 1).apply(func=numpy.log)
        mask_positive_inf = result == numpy.inf
        mask_negative_inf = result == -numpy.inf
        self.impute_max = result[~mask_positive_inf].max()
        self.impute_min = result[~mask_negative_inf].min()
        result[mask_positive_inf] = self.impute_max
        result[mask_negative_inf] = self.impute_min
        y = result.iloc[1:].copy()
        self.model = self._model(window=self.window, **self.model_kwargs)
        run_time = time.time() - run_time
        print('prep', run_time)
        run_time = time.time()
        self.model.fit(y=y)
        run_time = time.time() - run_time
        print('fit', run_time)
        run_time = time.time()
        y_hat = self.model.predict(y=y)[1:]
        y_forecast = self.model.forecast(y=y)
        y_hat = numpy.concatenate((y_hat, numpy.array([y_forecast])))
        result = result.copy()
        result.iloc[1:] = y_hat.astype(dtype=result.dtype)
        run_time = time.time() - run_time
        print('cast', run_time)
        return result
    def project_second(self, series_dict):
        assert len(list(series_dict.keys())) == 1
        result = series_dict[list(series_dict.keys())[0]].copy()
        name = result.name
        forecasted = []
        for i in range(result.shape[0]):
            result_ = pandas.concat((self.y_project_first_series.iloc[i+1:],
                                     result.iloc[:i+1]), ignore_index=False)
            if self.log:
                result_ = result_.pct_change()
                result_ = (result_ + 1).apply(func=numpy.log)
            mask_positive_inf = result_ == numpy.inf
            mask_negative_inf = result_ == -numpy.inf
            result_[mask_positive_inf] = self.impute_max
            result_[mask_negative_inf] = self.impute_min
            y = result_.iloc[1:].copy()
            y_hat_i = self.model.forecast(y=y)
            forecasted.append(y_hat_i)
        result = result.copy()
        result = pandas.Series(data=forecasted, index=result.index)
        return result
    @property
    def lag(self):
        return self.window


class betaAutoArima:
    """
    AutoArima implementation following the algorithm outlined in https://otexts.com/fpp2/arima-r.html
    (with some minor adjustments)

    No seasonal component is considered

    """
    def __init__(self, window, max_d=2):
        self.max_window = window
        self.max_d = max_d
        self._arima = ARIMA
        self.arima = None
        self.fitted_p = None
        self.fitted_d = None
        self.fitted_q = None
        self.fitted_trend = None
    def fit(self, y):

        # identify d

        # codes: t_d0, c_d0, c_d1, n(c)_d2
        kpss_result = pandas.DataFrame(index=['t_d0', 'c_d0', 'c_d1', 'n_d2'],
                                       columns=['kpss_values_stat', 'kpss_values_p1', 'kpss_values_p10', 'kpss_trend', 'd'],
                                       data=[[numpy.nan, numpy.nan, numpy.nan, 'ct', 0],
                                             [numpy.nan, numpy.nan, numpy.nan, 'c', 0],
                                             [numpy.nan, numpy.nan, numpy.nan, 'c', 1],
                                             [numpy.nan, numpy.nan, numpy.nan, 'c', 2]])
        for ix in kpss_result.index:
            yy = y.copy()
            kpss_d = kpss_result.loc[ix, 'd']
            for j in range(kpss_d):
                yy = yy.diff()
            yy = yy.dropna().values
            kpss_trend = kpss_result.loc[ix, 'kpss_trend']
            try:
                kpss_output = kpss(x=yy, regression=kpss_trend, nlags='auto')
            except OverflowError:
                try:
                    kpss_output = kpss(x=yy, regression=kpss_trend, nlags='legacy')
                except Exception as e:
                    raise e
            except Exception as e:
                raise e
            # kpss_result.loc[ix, 'kpss_values'] = kpss_output[1]
            # due to undesired properties of the implementation of kpss test in the part how p-values are projected,
            # a decision was made to consider test statistic values themselves
            kpss_result.loc[ix, 'kpss_values_stat'] = kpss_output[0]
            kpss_result.loc[ix, 'kpss_values_p1'] = kpss_output[3]['1%']
            kpss_result.loc[ix, 'kpss_values_p10'] = kpss_output[3]['10%']
        lesser_mask = kpss_result['kpss_values_stat'] <= kpss_result['kpss_values_p10']
        if lesser_mask.sum() > 0:
            conditional_min = kpss_result.loc[lesser_mask, 'kpss_values_stat'].min()
            i = kpss_result['kpss_values_stat'].values.tolist().index(conditional_min)
        else:
            i = kpss_result['kpss_values_stat'].argmin()

        d, alternative_trend = kpss_result['d'].values[i], kpss_result['kpss_trend'].values[i]

        # first examination

        # codes: 0_d_0, 2_d_2, 1_d_0, 0_d_1, [0_d_0 -n]

        if alternative_trend == 'ct':
            arima_results = pandas.DataFrame(columns=['aic', 'p', 'q', 'trend'],
                                             data=[[numpy.nan, 0, 0, 'ct'],
                                                   [numpy.nan, 2, 2, 'ct'],
                                                   [numpy.nan, 1, 0, 'ct'],
                                                   [numpy.nan, 0, 1, 'ct'],
                                                   [numpy.nan, 0, 0, 'c'],
                                                   [numpy.nan, 2, 2, 'c'],
                                                   [numpy.nan, 1, 0, 'c'],
                                                   [numpy.nan, 0, 1, 'c'],
                                                   [numpy.nan, 0, 0, 'n']])
        elif d == 0:
            arima_results = pandas.DataFrame(columns=['aic', 'p', 'q', 'trend'],
                                             data=[[numpy.nan, 0, 0, 'c'],
                                                   [numpy.nan, 2, 2, 'c'],
                                                   [numpy.nan, 1, 0, 'c'],
                                                   [numpy.nan, 0, 1, 'c'],
                                                   [numpy.nan, 0, 0, 'n']])
        elif d == 1:
            arima_results = pandas.DataFrame(columns=['aic', 'p', 'q', 'trend'],
                                             data=[[numpy.nan, 0, 0, 't'],
                                                   [numpy.nan, 2, 2, 't'],
                                                   [numpy.nan, 1, 0, 't'],
                                                   [numpy.nan, 0, 1, 't'],
                                                   [numpy.nan, 0, 0, 'n']])
        else:
            arima_results = pandas.DataFrame(columns=['aic', 'p', 'q', 'trend'],
                                             data=[[numpy.nan, 0, 0, 't2'],
                                                   [numpy.nan, 2, 2, 't2'],
                                                   [numpy.nan, 1, 0, 't2'],
                                                   [numpy.nan, 0, 1, 't2']])

        for i in range(arima_results.shape[0]):
            p = arima_results['p'].values[i]
            q = arima_results['q'].values[i]
            trend = arima_results['trend'].values[i]
            arima = self._arima(endog=y.values, order=(p, d, q), trend=refactor_trend_code(trend), seasonal_order=(0, 0, 0, 0))
            arima_res = arima.fit()
            arima_results.loc[arima_results.index[i], 'aic'] = arima_res.aic
        i = arima_results['aic'].argmin()
        current_aic = arima_results['aic'].values[i]
        current_p = arima_results['p'].values[i]
        current_q = arima_results['q'].values[i]
        current_trend = arima_results['trend'].values[i]

        # loops

        finish = False
        while not finish:

            max_pq = self.max_window - max(d, current_p, current_q)

            # codes: (p+1, q), (p-1, q), (p, q+1), (p, q-1), all those without c
            data = []
            if current_p > 1:
                data_append = [[numpy.nan, current_p - 1, current_q, current_trend]]
                data += data_append
            if current_p < max_pq:
                data_append = [[numpy.nan, current_p + 1, current_q, current_trend]]
                data += data_append
            if current_q > 1:
                data_append = [[numpy.nan, current_p, current_q - 1, current_trend]]
                data += data_append
            if current_q < max_pq:
                data_append = [[numpy.nan, current_p, current_q + 1, current_trend]]
                data += data_append
            if current_trend != 'n':
                data_append_n = []
                for z in data:
                    zz = list(z)
                    zz[-1] = 'n'
                    data_append_n += [zz]
                data_append_c = []
                if current_trend == 'ct':
                    for z in data:
                        zz = list(z)
                        zz[-1] = 'c'
                        data_append_c += [zz]
                data += data_append_n
                data += data_append_c
            else:
                data_append_c = []
                for z in data:
                    zz = list(z)
                    zz[-1] = 'c' if d == 0 else 't' if d == 1 else 't2'
                    data_append_c += [zz]
                data += data_append_c
            arima_results = pandas.DataFrame(columns=['aic', 'p', 'q', 'trend'],
                                             data=data)

            for i in range(arima_results.shape[0]):
                p = arima_results['p'].values[i]
                q = arima_results['q'].values[i]
                trend = arima_results['trend'].values[i]
                arima = self._arima(endog=y.values, order=(p, d, q), trend=refactor_trend_code(trend), seasonal_order=(0, 0, 0, 0))
                arima_res = arima.fit()
                arima_results.loc[arima_results.index[i], 'aic'] = arima_res.aic
            i = arima_results['aic'].argmin()
            candidate_aic = arima_results['aic'].values[i]

            if candidate_aic <= current_aic:
                current_aic = candidate_aic
                current_p = arima_results['p'].values[i]
                current_q = arima_results['q'].values[i]
                current_trend = arima_results['trend'].values[i]
            else:
                finish = True

        self.fitted_p = current_p
        self.fitted_d = d
        self.fitted_q = current_q
        self.fitted_trend = current_trend
        self.arima = self._arima(endog=y.values, order=(current_p, d, current_q), trend=refactor_trend_code(current_trend),
                                 seasonal_order=(0, 0, 0, 0)).fit()

    def predict(self, y):

        self.arima = self._arima(endog=y.values, order=(self.fitted_p, self.fitted_d, self.fitted_q), trend=refactor_trend_code(self.fitted_trend),
                                 seasonal_order=(0, 0, 0, 0)).fit()

        prediction = self.arima.predict()
        return prediction

    def forecast(self, y):

        self.arima = self._arima(endog=y.values, order=(self.fitted_p, self.fitted_d, self.fitted_q), trend=refactor_trend_code(self.fitted_trend),
                                 seasonal_order=(0, 0, 0, 0)).fit()

        forecasted = self.arima.forecast()[-1]
        return forecasted


class betaDefiniteArima:
    def __init__(self, window, p, q, max_d=2):
        assert window >= max(p, q, max_d)
        self.p = p
        self.q = q
        self.max_d = max_d
        self.d = None
        self.trend = None
        self._arima = ARIMA
        self.arima = None
    def fit(self, y):

        # identify d

        # codes: t_d0, c_d0, c_d1, n(c)_d2
        kpss_result = pandas.DataFrame(index=['t_d0', 'c_d0', 'c_d1', 'n_d2'],
                                       columns=['kpss_values', 'kpss_trend', 'd'],
                                       data=[[numpy.nan, 'ct', 0],
                                             [numpy.nan, 'c', 0],
                                             [numpy.nan, 'c', 1],
                                             [numpy.nan, 'c', 2]])
        for ix in kpss_result.index:
            yy = y.copy()
            kpss_d = kpss_result.loc[ix, 'd']
            for j in range(kpss_d):
                yy = yy.diff()
            yy = yy.dropna().values
            kpss_trend = kpss_result.loc[ix, 'kpss_trend']
            try:
                kpss_output = kpss(x=yy, regression=kpss_trend, nlags='auto')
            except OverflowError:
                try:
                    kpss_output = kpss(x=yy, regression=kpss_trend, nlags='legacy')
                except Exception as e:
                    raise e
            except Exception as e:
                raise e
            kpss_result.loc[ix, 'kpss_values'] = kpss_output[1]
        i = kpss_result['kpss_values'].argmax()
        d, alternative_trend = kpss_result['d'].values[i], kpss_result['kpss_trend'].values[i]

        self.d = d
        if d == 0:
            self.trend = alternative_trend
        elif d == 1:
            self.trend = 'n' if alternative_trend == 'n' else 't'
        else:
            self.trend = 'n' if alternative_trend == 'n' else 't2'

        self.arima = self._arima(endog=y.values, order=(self.p, self.d, self.q), trend=refactor_trend_code(self.trend),
                                 seasonal_order=(0, 0, 0, 0)).fit()

    def predict(self, y):

        self.arima = self._arima(endog=y.values, order=(self.p, self.d, self.q), trend=refactor_trend_code(self.trend),
                                 seasonal_order=(0, 0, 0, 0)).fit()

        prediction = self.arima.predict()
        return prediction

    def forecast(self, y):

        self.arima = self._arima(endog=y.values, order=(self.p, self.d, self.q), trend=refactor_trend_code(self.trend),
                                 seasonal_order=(0, 0, 0, 0)).fit()

        forecasted = self.arima.forecast()[-1]
        return forecasted
