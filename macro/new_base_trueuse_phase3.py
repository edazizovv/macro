#


#
import numpy
import pandas

from sklearn.tree import DecisionTreeRegressor

import torch
from torch import nn

from matplotlib import pyplot
from sklearn.tree import plot_tree

# from neusk.neura import WrappedNN
from neura import WrappedNN

from mayer.the_skeleton.diezel import DiStill
from mayer.the_skeleton.func import simple_search_report
from mayer.the_skeleton.losses import loss_202012var_party

#
from macro.new_base import Path, Projector, Item, FoldGenerator
from macro.new_data_check import control, controller_view
from macro.new_base_trueuse_pods import features, path_pseudo_edges, path_matrix, path_vertices, sources, name_list, param_list

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

targets = ['TLT_aggmean_pct', 'IVV_aggmean_pct']

savers = numpy.ones(shape=(len(path_vertices),)).astype(dtype=bool)
fg.init_path(path_vertices, path_matrix, path_pseudo_edges, savers)

timeaxis = sources[1].series['DATE'].values[sources[1].series['DATE'].values >= pandas.to_datetime('2007-01-01').isoformat()]

"""
Functions: to be moved to mayer
"""

def sharpe_ratio(yields_produced, yields_baseline):

    return (yields_produced - yields_baseline).mean() / yields_produced.std()

"""
Stage 3: Weight model
"""

class PtfLoss:
    def __init__(self):
        pass
    def __call__(self, y_pred, y_train):
        return loss_202012var_party(weights=y_pred, yields=y_train)

nn_model = WrappedNN
nn_kwargs = {
    'layers': [nn.Linear, nn.Linear],
    'layers_dimensions': [20, 2],
    'layers_kwargs': [{}, {}],
    'batchnorms': [None, None],  # nn.BatchNorm1d
    'activators': [nn.LeakyReLU, nn.Softmax],
    'interdrops': [0.0, 0.0],
    'optimiser': torch.optim.Adamax,  # Adamax / AdamW / SGD
    'optimiser_kwargs': {
        'lr': 0.001,
        'weight_decay': 0.001,
        # 'momentum': 0.9,
        # 'nesterov': True,
    },
    'scheduler': torch.optim.lr_scheduler.ConstantLR,
    'scheduler_kwargs': {
        'factor': 1,
        'total_iters': 2,
    },
    'loss_function': PtfLoss,
    'epochs': 2000,
}
distill_model = DecisionTreeRegressor
distill_kwargs = {'max_depth': 3}
commission_fee = 0.01

cumulatives = []
performances = []
for fold_n in fg.folds:
    print(fold_n)
    data_train, data_test = fg.fold(sources, features + targets, timeaxis, fold_n=fold_n)
    x_train, y_train = data_train[[x for x in data_train.columns if x not in targets]].iloc[:-1, :], data_train[targets].iloc[1:]
    x_test, y_test = data_test[[x for x in data_test.columns if x not in targets]].iloc[:-1, :], data_test[targets].iloc[1:]

    tt_train = x_train.index.values
    tt_test = x_test.index.values

    x_train = x_train.values
    x_test = x_test.values
    y_train = y_train.values
    y_test = y_test.values

    x_train = torch.tensor(x_train, dtype=torch.float)
    x_test = torch.tensor(x_test, dtype=torch.float)
    y_train = torch.tensor(y_train, dtype=torch.float)
    y_test = torch.tensor(y_test, dtype=torch.float)

    dynamics_benchmark_train = 0.01 * numpy.ones(shape=(x_train.shape[0],))
    dynamics_benchmark_test = 0.01 * numpy.ones(shape=(x_test.shape[0],))

    still = DiStill(
        nn_model=nn_model,
        nn_kwargs=nn_kwargs,
        distill_model=distill_model,
        distill_kwargs=distill_kwargs,
        commi=commission_fee,
    )

    still.still(X_train=x_train, Y_train=y_train, X_val=x_train, Y_val=y_train)
    signals_train = still.nn_signals(X=x_train)
    signals_test = still.nn_signals(X=x_test)
    dynamics_train = still.nn_portfolio(X=x_train, Y=y_train, cum=False)
    dynamics_test = still.nn_portfolio(X=x_test, Y=y_test, cum=False)
    cumulative_train = (1 + dynamics_train).cumprod()
    cumulative_test = (1 + dynamics_test).cumprod()

    dynamics_c0_train = 0.5 * y_train[:, 0].numpy() + 0.5 * y_train[:, 1].numpy()
    dynamics_c1_train = y_train[:, 0].numpy()
    dynamics_c2_train = y_train[:, 1].numpy()
    dynamics_synth_train = 0.8 * (0.5 * y_train[:, 0].numpy() + 0.5 * y_train[:, 1].numpy()) + 0.2 * dynamics_train.flatten()

    cumulative_c0_train = (1 + dynamics_c0_train).cumprod()
    cumulative_c1_train = (1 + dynamics_c1_train).cumprod()
    cumulative_c2_train = (1 + dynamics_c2_train).cumprod()
    cumulative_synth_train = (1 + dynamics_synth_train).cumprod()

    dynamics_c0_test = 0.5 * y_test[:, 0].numpy() + 0.5 * y_test[:, 1].numpy()
    dynamics_c1_test = y_test[:, 0].numpy()
    dynamics_c2_test = y_test[:, 1].numpy()
    dynamics_synth_test = 0.8 * (0.5 * y_test[:, 0].numpy() + 0.5 * y_test[:, 1].numpy()) + 0.2 * dynamics_test.flatten()

    cumulative_c0_test = (1 + dynamics_c0_test).cumprod()
    cumulative_c1_test = (1 + dynamics_c1_test).cumprod()
    cumulative_c2_test = (1 + dynamics_c2_test).cumprod()
    cumulative_synth_test = (1 + dynamics_synth_test).cumprod()

    still.plot(
        X_train=x_train,
        Y_train=y_train,
        tt_train=tt_train,
        bench_train=dynamics_benchmark_train,
        X_test=x_test,
        Y_test=y_test,
        tt_test=tt_test,
        bench_test=dynamics_benchmark_test,
        on='nn',
        report=simple_search_report,
        synth=0.8,
        do_plot=True,
    )

    reported_train = pandas.DataFrame(
        data={
            'c0': cumulative_c0_train,
            'c1': cumulative_c1_train,
            'c2': cumulative_c2_train,
            'synth': cumulative_synth_train,
            'hero': cumulative_train,
            'tt': tt_train,
        }
    )
    reported_test = pandas.DataFrame(
        data={
            'c0': cumulative_c0_test,
            'c1': cumulative_c1_test,
            'c2': cumulative_c2_test,
            'synth': cumulative_synth_test,
            'hero': cumulative_test,
            'tt': tt_test,
        }
    )

    reported_train['role'] = 'train'
    reported_test['role'] = 'test'

    reported = pandas.concat(
        (
            reported_train,
            reported_test,
        ),
        axis=0,
        ignore_index=True,
    )
    reported['fold'] = fold_n

    cumulatives.append(reported)

    sharpe_c0_train = sharpe_ratio(yields_produced=dynamics_c0_train, yields_baseline=dynamics_benchmark_train)
    sharpe_c1_train = sharpe_ratio(yields_produced=dynamics_c1_train, yields_baseline=dynamics_benchmark_train)
    sharpe_c2_train = sharpe_ratio(yields_produced=dynamics_c2_train, yields_baseline=dynamics_benchmark_train)
    sharpe_synth_train = sharpe_ratio(yields_produced=dynamics_synth_train, yields_baseline=dynamics_benchmark_train)
    sharpe_hero_train = sharpe_ratio(yields_produced=dynamics_train, yields_baseline=dynamics_benchmark_train)

    sharpe_c0_test = sharpe_ratio(yields_produced=dynamics_c0_test, yields_baseline=dynamics_benchmark_test)
    sharpe_c1_test = sharpe_ratio(yields_produced=dynamics_c1_test, yields_baseline=dynamics_benchmark_test)
    sharpe_c2_test = sharpe_ratio(yields_produced=dynamics_c2_test, yields_baseline=dynamics_benchmark_test)
    sharpe_synth_test = sharpe_ratio(yields_produced=dynamics_synth_test, yields_baseline=dynamics_benchmark_test)
    sharpe_hero_test = sharpe_ratio(yields_produced=dynamics_test, yields_baseline=dynamics_benchmark_test)

    aay_c0_train = cumulative_c0_train[-1] ** (1 / cumulative_c0_train.shape[0]) - 1
    aay_c1_train = cumulative_c1_train[-1] ** (1 / cumulative_c1_train.shape[0]) - 1
    aay_c2_train = cumulative_c2_train[-1] ** (1 / cumulative_c2_train.shape[0]) - 1
    aay_synth_train = cumulative_synth_train[-1] ** (1 / cumulative_synth_train.shape[0]) - 1
    aay_hero_train = cumulative_train[-1] ** (1 / cumulative_train.shape[0]) - 1

    aay_c0_test = cumulative_c0_test[-1] ** (1 / cumulative_c0_test.shape[0]) - 1
    aay_c1_test = cumulative_c1_test[-1] ** (1 / cumulative_c1_test.shape[0]) - 1
    aay_c2_test = cumulative_c2_test[-1] ** (1 / cumulative_c2_test.shape[0]) - 1
    aay_synth_test = cumulative_synth_test[-1] ** (1 / cumulative_synth_test.shape[0]) - 1
    aay_hero_test = cumulative_test[-1] ** (1 / cumulative_test.shape[0]) - 1

    date_start_train = tt_train[0]
    date_end_train = tt_train[-1]
    date_start_test = tt_test[0]
    date_end_test = tt_test[-1]

    performance_train = [
        aay_c0_train,
        sharpe_c0_train,
        aay_c1_train,
        sharpe_c1_train,
        aay_c2_train,
        sharpe_c2_train,
        aay_synth_train,
        sharpe_synth_train,
        aay_hero_train,
        sharpe_hero_train,
        date_start_train,
        date_end_train,
        'train',
        fold_n,
    ]

    performance_test = [
        aay_c0_test,
        sharpe_c0_test,
        aay_c1_test,
        sharpe_c1_test,
        aay_c2_test,
        sharpe_c2_test,
        aay_synth_test,
        sharpe_synth_test,
        aay_hero_test,
        sharpe_hero_test,
        date_start_test,
        date_end_test,
        'test',
        fold_n,
    ]

    performances.append(performance_train)
    performances.append(performance_test)

    fig, ax = pyplot.subplots()
    plot_tree(still.distill_model_fit, feature_names=features)
    fig.savefig(f'./tree_{fold_n}.svg')
    pyplot.close(fig)

reported = pandas.concat(
    cumulatives,
    axis=0,
    ignore_index=True,
)

reported_performance = pandas.DataFrame(
    data=performances,
    columns=[
        'aay_c0',
        'sharpe_c0',
        'aay_c1',
        'sharpe_c1',
        'aay_c2',
        'sharpe_c2',
        'aay_synth',
        'sharpe_synth',
        'aay_hero',
        'sharpe_synth',
        'date_start',
        'date_end',
        'role',
        'fold_n',
    ]
)
