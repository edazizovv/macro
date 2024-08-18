#


#
import numpy
import pandas


#


#
class Soil:
    def __init__(self, source, target_name, target_repr, time_axis, gb_func, model_base, model_base_kwg, time_sub_rate=0.1, time_sub_replace=True, x_sub_rate=0.1, x_sub_replace=True):
        self.source = source
        self.time_axis = time_axis
        self.gb_func = gb_func

        self.time_sub_rate = time_sub_rate
        self.time_sub_replace = time_sub_replace
        nt = int(self.time_axis.shape[0] * self.time_axis)
        self.ix_time = numpy.random.choice(self.time_axis.values, size=(nt,), replace=self.time_sub_replace)

        self.target_name = target_name
        self.target_repr = target_repr

        self.x_sub_rate = x_sub_rate
        self.x_sub_replace = x_sub_replace
        nx = int(self.source.n * self.x_sub_rate)
        self.x_base = numpy.random.choice(self.source.names, size=(nx,), replace=self.x_sub_replace)

        self.x_source_names = self.x_base.tolist()
        self.x_representations = [Representation()] * len(self.x_source_names)
        self.x_scores = [1 / len(self.x_source_names)] * len(self.x_source_names)

        self.model = None
        self.model_base = model_base
        self.model_base_kwg = model_base_kwg
        self.model_reported = None
        self.assessor = None
        self.grow_book = None

        self._water_garden()

        self.step_count = 1

    def step(self, assessor, grow_book):

        self._assess_scores(assessor=assessor)

        self._grow_elements(grow_book=grow_book)

        self._water_garden()

        self.step_count += 1

    def _assess_scores(self, assessor):
        self.assessor = assessor
        self.x_scores = assessor(self.model_reported)

    def _grow_elements(self, grow_book):
        self.grow_book = grow_book
        for j in range(len(self.x_source_names)):
            source_name = self.x_source_names[j]
            score = self.x_scores[j]
            r = numpy.random.choice([True, False], p=[score, 1 - score])
            if r:
                current_representation = self.x_representations[j]
                name = self.source[source_name].get_representation_name(current_representation)
                new_representation = self.grow_book.generate(name)
                self.x_source_names.append(source_name)
                self.x_representations.append(new_representation)

    def _water_garden(self):
        data = []
        for j in range(len(self.x_source_names)):
            source_name = self.x_source_names[j]
            representation = self.x_representations[j]
            represented = self.source[source_name].represent(representation, origin=True, target_axis=self.time_axis, gb_func=self.gb_func, target_axis_subx=self.ix_time)
            data.append(represented)
        data = pandas.concat(data, axis=1, ignore_index=False)
        target_data = self.source[self.target_name].represent(self.target_repr, origin=True, target_axis=self.time_axis, gb_func=self.gb_func, target_axis_subx=self.ix_time)

        self.model = self.model_base(**self.model_base_kwg)
        self.model_reported = self.model.fit(x=data.values, y=target_data.values)


class Representation:
    def __init__(self, repr_function=None):
        self.repr_function = repr_function
    def represent(self, frame, **kwargs):
        return self.repr_function(frame, **kwargs)


book_tree = {'None': ['seasonality_fix',
                      'fill_in_missing', 'fill_in_outliers', 'impute_spvs',
                      'dist_transform', 'binning', 'generators'],
             'dist_transform': ['dist_transform_advanced'],
             'dist_transform_advanced': ['dist_transform_tuning'],
             'binning': ['bin_focusing', 'bin_merging']}
book_tree_kwg = {'None': {},
                 'seasonality_fix': {},
                 'fill_in_missing': {},
                 'fill_in_outliers': {},
                 'impute_spvs': {},
                 'dist_transform': {},
                 'dist_transform_advanced': {},
                 'dist_transform_tuning': {},
                 'binning': {},
                 'bin_focusing': {},
                 'bin_merging': {},
                 'generators': {}}
book_tree_methods = {'None': {},
                 'seasonality_fix': {},
                 'fill_in_missing': {},
                 'fill_in_outliers': {},
                 'impute_spvs': {},
                 'dist_transform': {},
                 'dist_transform_advanced': {},
                 'dist_transform_tuning': {},
                 'binning': {},
                 'bin_focusing': {},
                 'bin_merging': {},
                 'generators': {}}
book_tree_default_probs = {'None': [0.2,
                                    0.075, 0.075, 0.075,
                                    0.175, 0.2, 0.2],
                           'dist_transform': [1.0],
                           'dist_transform_advanced': [1.0],
                           'binning': [0.5, 0.5]}

class GrowBook:
    def __init__(self):


# LEFT:
#   grow book
#   model
#   test trial


class Seed:
    ...
