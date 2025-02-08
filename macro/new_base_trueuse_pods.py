#
import json


#
import numpy
import pandas


#
from macro.new_base import Path, Projector, Item, FoldGenerator
from macro.new_base_test_projectors import WindowRollImpulse, SimpleAggregator, SimpleCasterAggMonth
from macro.new_data_check import pod_loader, controller_view
from macro.new_base_test_projectors_translate import a_function_translator, a_function_kwg_translator


#
loader_source = '../data/data_meta/loader_pitch.xlsx'
controller_source = '../data/other/controller_pitch.xlsx'

source_xl = pandas.read_excel('../data/data_meta/vector.xlsx', sheet_name='sources')
path_xl = pandas.read_excel('../data/data_meta/vector.xlsx', sheet_name='prog')

sources = ([Item(name=x, loader_source=loader_source, controller_source=controller_source) for x in source_xl['source'].values if x in path_xl['vertices_in'].values])


vertices_out = path_xl['vertices_out'].values
vertices_in_conf = path_xl['vertices_in'].str.split(';').values
vertices_in = numpy.unique(vertices_in_conf.flatten())

assert vertices_out.shape[0] == numpy.unique(vertices_out).shape[0]
vertices_joint = numpy.unique([y for x in vertices_in for y in x] + vertices_out.tolist())

path_vertices = vertices_joint.tolist()
path_matrix = numpy.zeros(shape=(vertices_joint.shape[0], vertices_joint.shape[0]))
for k in range(path_xl.shape[0]):
    i = path_vertices.index(vertices_out[k])
    for x in vertices_in_conf[k]:
        j = path_vertices.index(x)
        if vertices_out[k] != x:
            path_matrix[j, i] = 1

sc = SimpleCasterAggMonth()


path_pseudo_edges = numpy.full(shape=(vertices_joint.shape[0],), fill_value=None)
for j in range(path_xl.shape[0]):
    if path_xl.loc[j, 'vertices_out'] == path_xl.loc[j, 'vertices_in']:
        projector = None
    else:
        projector_role = path_xl.loc[j, 'projector_role']
        a_function = a_function_translator[path_xl.loc[j, 'a_function']]
        a_function_kwg = a_function_kwg_translator(a=path_xl.loc[j, 'a_function_kwg'])
        if projector_role == 'downcast':
            projector = Projector(ts_creator=sc,
                                  role=projector_role,
                                  agg_function=a_function,
                                  agg_function_kwg=a_function_kwg)
        elif projector_role == 'recast':
            projector = Projector(ts_creator=sc,
                                  role=projector_role,
                                  app_function=a_function,
                                  app_function_kwg=a_function_kwg)
        else:
            raise Exception("'projector_code' should be valued as either 'downcast' or 'recast'")
    j = path_vertices.index(path_xl.loc[j, 'vertices_out'])
    path_pseudo_edges[j] = projector

features = path_xl.loc[path_xl['active'] == 'Y', 'vertices_out'].values.tolist()

vxl = pandas.read_excel('../data/data_meta/vincent.xlsx', sheet_name='mobsters')
name_list = vxl['transform_name'].values.tolist()
param_list = []

for j in range(vxl.shape[0]):

    a_function = a_function_translator[vxl.loc[j, 'a_function']]
    a_function_kwg = a_function_kwg_translator(a=vxl.loc[j, 'a_function_kwg'])

    pp = {'ts_creator': sc, 'role': 'recast', 'app_function': a_function, 'app_function_kwg': a_function_kwg}

    param_list.append(pp)
