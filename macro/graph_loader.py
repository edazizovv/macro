#


#
import numpy
import pandas


#
from macro.new_base import Projector, Item
from macro.new_base_test_projectors_translate import a_function_translator, a_function_kwg_translator
from macro.new_base_test_projectors import SimpleCasterAggMonth


#
class Freq:
    def __init__(self, pd_frequency, to_end_frequency):
        self._pd_frequency = pd_frequency
        self._to_end_frequency = to_end_frequency
        self.delta = None
    @property
    def freq(self):
        return self._pd_frequency
    @property
    def to_end(self):
        return self._to_end_frequency


class StandardMSFreq(Freq):
    def __init__(self):
        super().__init__(pd_frequency="MS", to_end_frequency="ME")


class ElementLoader:
    def __init__(self, vector_path):

        self.vector_path = vector_path

        self.timeaxis = None
        self.features = None
        self.sources = None

        self.path_matrix = None
        self.path_pseudo_edges = None

        self._source_xl = None
        self._path_xl = None
        self._unique_inputs = None
        self._unique_outputs = None

        self._read_n_load()

    def _check_all_sources_used(self):

        mask_isin = self._sources.isin(self._unique_inputs)
        if not mask_isin.all():
            print(self._sources[~mask_isin])
            raise Exception("STELLAR PATH: Not all sources are used; see the print above")

    def _check_all_inputs_in_sources(self):

        mask_isin = numpy.isin(self._unique_inputs, self._sources.values)
        if not mask_isin.all():
            print(pandas.Series(self._unique_inputs[~mask_isin]))
            raise Exception("STELLAR PATH: Not all inputs are in sources; see the print above")

    def _unpack_inputs(self):

        formal_inputs = self._path_xl['vertices_in'].str.split(';')

        self_inputs = self._path_xl['vertices_in'] == self._path_xl['vertices_out']
        def _seek(x):
            return any([el in self._path_xl.loc[~self_inputs, 'vertices_out'].values for el in x])
        chains = self._path_xl['vertices_in'].str.split(';').apply(func=_seek)

        self._inputs_lists = formal_inputs.values
        self._unique_inputs = numpy.unique([y for x in self._inputs_lists[(~self_inputs) & (~chains)] for y in x])

        self._check_all_inputs_in_sources()

    def _unpack_outputs(self):

        mask_out_duplicates = self._path_xl['vertices_out'].value_counts() > 1
        if mask_out_duplicates.any():
            print(self._path_xl.loc[mask_out_duplicates, 'vertices_out'])
            raise Exception("STELLAR PATH: Duplicates found in outputs; see the print above")

        self_inputs = self._path_xl['vertices_in'] == self._path_xl['vertices_out']

        self._unique_outputs = self._path_xl.loc[~self_inputs, 'vertices_out'].values

    @property
    def path_vertices(self):
        return self._unique_inputs.tolist() + self._unique_outputs.tolist()

    def _fill_path_matrix(self):

        nv = len(self.path_vertices)

        path_matrix = numpy.zeros(shape=(nv, nv))
        for k in range(self._path_xl.shape[0]):
            i = self.path_vertices.index(self._path_xl['vertices_out'].values[k])
            for x in self._inputs_lists[k]:
                j = self.path_vertices.index(x)
                if self._path_xl['vertices_out'].values[k] != x:
                    path_matrix[j, i] = 1

        self.path_matrix = path_matrix

    def _fill_pseudo_edges(self):

        sc = SimpleCasterAggMonth()

        path_pseudo_edges = numpy.full(shape=(len(self.path_vertices),), fill_value=None)
        for j in range(self._path_xl.shape[0]):
            if self._path_xl.loc[j, 'vertices_out'] == self._path_xl.loc[j, 'vertices_in']:
                projector = None
            else:
                projector_role = self._path_xl.loc[j, 'projector_role']
                a_function = a_function_translator[self._path_xl.loc[j, 'a_function']]
                a_function_kwg = a_function_kwg_translator(a=self._path_xl.loc[j, 'a_function_kwg'])
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
            j = self.path_vertices.index(self._path_xl.loc[j, 'vertices_out'])
            path_pseudo_edges[j] = projector

        self.path_pseudo_edges = path_pseudo_edges

    def _fill_features(self):

        self.features = self._path_xl.loc[self._path_xl['active'].isin(["Y", "Z"]), 'vertices_out'].values.tolist()

    def _read_n_load(self):

        self._source_xl = pandas.read_excel(self.vector_path, sheet_name='sources')
        self._path_xl = pandas.read_excel(self.vector_path, sheet_name='prog')

        self._sources = self._source_xl['source']
        self._unpack_inputs()
        self._unpack_outputs()
        self._check_all_sources_used()

        self.sources = ([Item(name=x) for x in self._sources.values])

        self._fill_path_matrix()
        self._fill_pseudo_edges()
        self._fill_features()
