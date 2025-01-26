#


#
import numpy
import pandas


#
from macro.new_base import Path, Projector, Item, FoldGenerator
from macro.new_base_test_projectors import WindowAppGenerator
from macro.new_base_trueuse_phase0_garage import VincentClassFeatureEngineeringDeck
from macro.new_data_check import control, controller_view
from macro.new_base_truegarage import r2_metric, kendalltau_metric, somersd_metric, BasicLinearModel as MicroModel, BasicLassoSelectorModel as SelectorModel
from scipy import stats
from macro.new_base_trueuse_pods import features, path_pseudo_edges, path_matrix, path_vertices, sources, name_list, param_list
from macro.functional import sd_metric, pv_metric

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

target = 'IVV_aggmean_pct'

save_features = numpy.ones(shape=(len(path_vertices),)).astype(dtype=bool)
fg.init_path(path_vertices, path_matrix, path_pseudo_edges, save_features)

timeaxis = sources[1].series['DATE'].values[sources[1].series['DATE'].values >= pandas.to_datetime('2007-01-01').isoformat()]

"""
Stage 0: Generate features
"""
print("Stage 0")

projector = Projector
# performer = sd_metric
performer = pv_metric
target_source = 'IVV'
base_transform = 'pct_shift1'
target_transform = path_pseudo_edges[path_vertices.index(target)]

'''
NOTE:
Performer MUST be a measure with "greater-better" design 
(can be symmetric or support both negative and positive association).
Absulte differences must also be meaningful for it. 
'''

vcs = VincentClassFeatureEngineeringDeck(x_factors_in=features,
                                         target=target, target_source=target_source, target_transform=target_transform,
                                         name_list=name_list, base_transform=base_transform, param_list=param_list,
                                         projector=projector, performer=performer)
vcs.pull(fg=fg, sources=sources, timeaxis=timeaxis)
collapsed, collapsed_stats = vcs.collapse()

"""
driver = 'AWHAERT'    # A229RX0 AAA AWHAERT AWHMAN BAA
collapsed_selection, collapsed_stats_selection = collapsed_stats[nex][0],  collapsed_stats[nex][1]
collapsed_selection = collapsed_selection.sort_values(by='test2_result')
collapsed_stats_selection = collapsed_stats_selection.sort_values(by='transform')
"""

"""
from macro.new_base_trueuse_pods import vxl
collapsed_selection = pandas.DataFrame(data={'new_names': collapsed})
collapsed_selection['sources'] = collapsed_selection['new_names'].apply(func=lambda x: x.split('__')[0])
collapsed_selection['transforms'] = collapsed_selection['new_names'].apply(func=lambda x: x.split('__')[1])
collapsed_selection = collapsed_selection.merge(right=vxl, left_on=['transforms'], right_on=['transform_name'], how='left')
collapsed_selection.to_excel('../data/reports/output_phase0.xlsx')
"""