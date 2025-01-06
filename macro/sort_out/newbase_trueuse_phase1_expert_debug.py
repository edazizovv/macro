

import scipy
import numpy
import pandas

from macro.new_base import Path, Projector, Item, FoldGenerator

from macro.new_base_trueuse_pods import features, path_pseudo_edges, path_matrix, path_vertices, sources, name_list, param_list
from macro.functional import SomersD, pv_metric

n_folds = 10
joint_lag = 12
val_rate = 0.5
overlap_rate = 0.15
fg = FoldGenerator(n_folds=n_folds, joint_lag=joint_lag, val_rate=val_rate, overlap_rate=overlap_rate)

target = 'TLT_aggmean_pct'

fg.init_path(path_vertices, path_matrix, path_pseudo_edges)

timeaxis = sources[1].series['DATE'].values[sources[1].series['DATE'].values >= pandas.to_datetime('2007-01-01').isoformat()]

projector = Projector
# performer = SomersD
performer = pv_metric
target_source = 'TLT'
target_transform = path_pseudo_edges[path_vertices.index(target)]


j = 0

x_factors_in = list(features)
name = x_factors_in[j]
print(name)

fg_local = fg.copy()

sub_features = ['{0}__{1}'.format(name, name_list[j]) for j in range(len(name_list))]

target_components = [x for x in fg.path.path_vertices if target_source in x]
target_components_mask = numpy.isin(fg.path.path_vertices, target_components)
target_components_ix = numpy.arange(fg.path.path_matrix.shape[0])[target_components_mask]
path_matrix_sub_target = fg.path.path_matrix[target_components_ix[:, numpy.newaxis], target_components_ix]
path_vertices_sub_target = numpy.array(fg.path.path_vertices)[target_components_ix].tolist()
path_pseudo_edges_sub_target = fg.path.path_pseudo_edges[target_components_ix].tolist()
n_targets = target_components_mask.sum()

local_path_vertices = path_vertices_sub_target + [name] + sub_features
local_path_matrix = numpy.zeros(shape=((len(name_list) + 1 + n_targets), (len(name_list) + 1 + n_targets)))

targets_mask = numpy.arange(n_targets)
local_path_matrix[targets_mask[:, numpy.newaxis], targets_mask] = path_matrix_sub_target

local_path_matrix[n_targets, 3:] = 1

local_path_pseudo_edges = path_pseudo_edges_sub_target + [None] + [projector(**param_list[j]) for j in range(len(param_list))]


fg_local.init_path(local_path_vertices, local_path_matrix, local_path_pseudo_edges)

global_resulted = None


fold_n = 1
print(fold_n)

local_sources = [x for x in sources if (x.name == target_source) or (x.name == name)]
data_train, data_test = fg_local.fold(local_sources, sub_features + [name] + [target], timeaxis, fold_n=fold_n)
x_train, y_train = data_train[[x for x in data_train.columns if x != target]].iloc[:-1, :], data_train[target].iloc[1:]
x_test, y_test = data_test[[x for x in data_test.columns if x != target]].iloc[:-1, :], data_test[target].iloc[1:]

local_resulted = []
for j in range(len(name_list)):

    performed = performer(x=x_test.iloc[:, j].values, y=y_test.values)
    local_resulted.append([fold_n, name_list[j], performed, numpy.nan])
local_resulted = pandas.DataFrame(data=local_resulted, columns=['fold_n', 'transform', 'performed', 'stabilized'])

base_performed = performer(x=x_test.iloc[:, len(name_list)].values, y=y_test.values)
local_resulted['base_performed'] = numpy.abs(base_performed)

# xxl = pandas.DataFrame(data={'x': x_test.iloc[:, len(self.name_list)].values, 'y': y_test.values})
