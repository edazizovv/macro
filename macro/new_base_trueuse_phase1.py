#


#
import numpy
import pandas


#
from new_base import Path, Projector, Item, FoldGenerator
from new_base_test_projectors import WindowAppGenerator, VincentClassMobsterS
from new_data_check import control, controller_view
from new_base_truegarage import r2_metric, kendalltau_metric, somersd_metric, BasicLinearModel as MicroModel, BasicLassoSelectorModel as SelectorModel
from scipy import stats
from new_base_trueuse_pods import features, path_pseudo_edges, path_matrix, path_vertices, sources, name_list, param_list
from functional import SomersD, pv_metric

#
loader_source = '../data/data_meta/loader_pitch.xlsx'
controller_source = '../data/other/controller_pitch.xlsx'

controller = control(loader_source)
controller.to_excel(controller_source, index=False)

"""
This is three-phase build-up:
    1. Univariate thresh cut-off
    2. Finding LASSO-based skeleton
    3. Finding complement 
"""

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

fg.init_path(path_vertices, path_matrix, path_pseudo_edges)

timeaxis = sources[1].series['DATE'].values[sources[1].series['DATE'].values >= pandas.to_datetime('2007-01-01').isoformat()]

"""
Stage 0: Generate features
"""
print("Stage 0")

projector = Projector
# performer = SomersD
performer = pv_metric
target_source = 'TLT'
target_transform = path_pseudo_edges[path_vertices.index(target)]

vcs = VincentClassMobsterS(x_factors_in=features,
                           target=target, target_source=target_source, target_transform=target_transform,
                           name_list=name_list, param_list=param_list,
                           projector=projector, performer=performer)
vcs.pull(fg=fg, sources=sources, timeaxis=timeaxis)
collapsed, collapsed_stats = vcs.collapse()
# raise Exception("ghoul?")

"""
nex = ''
xxl, xxx = collapsed_stats[nex][0],  collapsed_stats[nex][1]
xxl = xxl.sort_values(by='test_result')
xxx = xxx.sort_values(by='transform')
"""
