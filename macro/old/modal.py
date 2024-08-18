#
import numpy
import pandas

# data loaded
r = Frame()
target = 'TLT'
target_e = r[target]
time_axis = target_e.frame['DATE'].copy()
target_representation = [Representation()]
x_source_names = r.names.tolist()
x_representations = [Representation()] * len(x_source_names)
time_sub_rate = 0.1
time_sub_replace = True
nt = int(time_axis.shape[0] * time_sub_rate)
gb_func = 'mean'
x_sub_rate = 0.1
x_sub_replace = False
nx = int(r.n * x_sub_rate)
model_base = ...
model_base_kwg = ...

# additional reps

# change: x_source_names x_representations

# set
m_runs = 10
n_folds = 10
folds = [x for x in range(n_folds)]
folded = []
reported = []
for i in range(m_runs):
    ix_time = numpy.random.choice(time_axis.values, size=(nt,), replace=time_sub_replace)
    x_base = numpy.random.choice(r.names, size=(nx,), replace=x_sub_replace)

    data = []
    x_source_names = x_base.tolist()
    for j in range(len(x_source_names)):
        source_name = x_source_names[j]
        representation = x_representations[j]
        represented = r[source_name].represent(representation, origin=True, target_axis=time_axis,
                                               gb_func=gb_func, target_axis_subx=ix_time)
        data.append(represented)
    data = pandas.concat(data, axis=1, ignore_index=False)
    target_series = r[target].represent(target, origin=True, target_axis=time_axis,
                                        gb_func=gb_func, target_axis_subx=ix_time)

    cell_reported = []
    for k in folds:
        ix_folded = numpy.random.choice(ix_time,
                                        p=[1 / ix_time.shape[0]] * ix_time.shape[0],
                                        size=(int(ix_time.shape[0] / n_folds),),
                                        replace=False)
        ix_left = [x for x in ix_time if x not in ix_folded]
        x_train = data.loc[ix_folded, :]
        x_val = data.loc[ix_left, :]

        for c in x_train.columns:
            x_train[c] = x_train[c].fillna(x_train[c].mean())
            x_val[c] = x_val[c].fillna(x_val[c].mean())

        y_train = target_series[ix_folded]
        y_val = target_series[ix_left]

        y_train = y_train.fillna(y_train.mean())
        y_val = y_val.fillna(y_val.mean())

        model = model_base(**model_base_kwg)
        model.fit(x=x_train.values, y=y_train.values)

        y_hat_train = model.predict(x=x_train.values)
        y_hat_test = model.predict(x=x_val.values)

        fold_reported = model.report()
        fold_reported['fold'] = k
        cell_reported.append(fold_reported)
    cell_reported = pandas.concat(cell_reported, axis=1, ignore_index=True)
    cell_reported['cell'] = i
    reported.append(cell_reported)
reported = pandas.concat(reported, axis=1, ignore_index=True)

# estimates
reported_cell_agg = reported.groupby(by='fold')['improvement'].mean()
reported_agg = reported_cell_agg.groupby(by='cell')['improvement'].mean()
