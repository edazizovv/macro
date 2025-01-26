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
from macro.new_base_trueuse_phase3_garage import producer

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

reported, reported_performance = producer(
    fg=fg,
    sources=sources,
    features=features,
    targets=targets,
    timeaxis=timeaxis,
    nn_model=nn_model,
    nn_kwargs=nn_kwargs,
    distill_model=distill_model,
    distill_kwargs=distill_kwargs,
    commission_fee=commission_fee,
    simple_search_report=simple_search_report,
)
