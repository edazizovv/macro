#


#
import numpy
import pandas
from scipy import stats


#


#
class VincentClassFeatureEngineeringDeck:

    def __init__(self, x_factors_in, target, target_source, target_transform, name_list, base_transform, param_list, projector, performer=None, stabilizer=None):

        self.mobsters = {x_factors_in[j]: VincentClassFeatureEngineeringPod(name=x_factors_in[j], target=target, target_source=target_source, target_transform=target_transform, name_list=name_list, base_transform=base_transform, param_list=param_list, projector=projector, performer=performer, stabilizer=stabilizer) for j in range(len(x_factors_in))}

    def pull(self, fg, sources, timeaxis):

        for key in self.mobsters.keys():

            print('{0} / {1}'.format(list(self.mobsters.keys()).index(key), len(list(self.mobsters.keys()))))

            fg_local = fg.copy()

            local_path_vertices, local_path_matrix, local_path_pseudo_edges, savers = self.mobsters[key].supply(fg=fg)
            fg_local.init_path(local_path_vertices, local_path_matrix, local_path_pseudo_edges, savers)

            for fold_n in fg_local.folds:
                local_sources = [x for x in sources if (x.name == self.mobsters[key].target_source) or (x.name == self.mobsters[key].name)]
                data_train, data_test = fg_local.fold(local_sources, self.mobsters[key].features + [self.mobsters[key].name] + [self.mobsters[key].target], timeaxis, fold_n=fold_n)
                x_train, y_train = data_train[[x for x in data_train.columns if x != self.mobsters[key].target]].iloc[:-1, :], data_train[self.mobsters[key].target].iloc[1:]
                x_test, y_test = data_test[[x for x in data_test.columns if x != self.mobsters[key].target]].iloc[:-1, :], data_test[self.mobsters[key].target].iloc[1:]

                z_train = pandas.concat((x_train, y_train), axis=1)
                z_train.to_excel('../data/data_folds/data_train_ph0_{1}_{0}.xlsx'.format(fold_n, key), index=True)
                z_test = pandas.concat((x_test, y_test), axis=1)
                z_test.to_excel('../data/data_folds/data_test_ph0_{1}_{0}.xlsx'.format(fold_n, key), index=True)

                self.mobsters[key].pull(fold_n=fold_n, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)

    def collapse(self):

        collapsed = []
        collapsed_stats = {}

        for key in self.mobsters.keys():

            print('{0} / {1}'.format(list(self.mobsters.keys()).index(key), len(list(self.mobsters.keys()))))
            c, s = self.mobsters[key].collapse()
            collapsed += c
            collapsed_stats[key] = s

        collapsed = numpy.unique(collapsed).tolist()

        return collapsed, collapsed_stats


class VincentClassFeatureEngineeringPod:

    def __init__(self, name, target, target_source, target_transform, name_list, base_transform, param_list, projector, performer=None, stabilizer=None):

        self.name = name
        self.target = target
        self.target_source = target_source
        self.target_transform = target_transform
        self.name_list = name_list
        self.base_transform = base_transform
        self.param_list = param_list
        self._projector = projector
        self.performer = performer
        self.stabilizer = stabilizer

        self.features = ['{0}__{1}'.format(self.name, self.name_list[j]) for j in range(len(self.name_list))]
        self.local_resulted = None
        self.global_resulted = None
        self.global_resulted_agg = None

    def supply(self, fg):

        target_components = [x for x in fg.path.path_vertices if self.target_source in x]
        target_components_mask = numpy.isin(fg.path.path_vertices, target_components)
        target_components_ix = numpy.arange(fg.path.path_matrix.shape[0])[target_components_mask]
        path_matrix_sub_target = fg.path.path_matrix[target_components_ix[:, numpy.newaxis], target_components_ix]
        path_vertices_sub_target = numpy.array(fg.path.path_vertices)[target_components_ix].tolist()
        path_pseudo_edges_sub_target = fg.path.path_pseudo_edges[target_components_ix].tolist()
        n_targets = target_components_mask.sum()

        local_path_vertices = path_vertices_sub_target + [self.name] + self.features
        savers = [True] * len(path_vertices_sub_target) + [True] + [False] * len(self.features)
        savers = numpy.array(savers)
        local_path_matrix = numpy.zeros(shape=((len(self.name_list) + 1 + n_targets), (len(self.name_list) + 1 + n_targets)))

        targets_mask = numpy.arange(n_targets)
        local_path_matrix[targets_mask[:, numpy.newaxis], targets_mask] = path_matrix_sub_target

        local_path_matrix[n_targets, 3:] = 1

        local_path_pseudo_edges = path_pseudo_edges_sub_target + [None] + [self._projector(**self.param_list[j]) for j in range(len(self.param_list))]

        return local_path_vertices, local_path_matrix, local_path_pseudo_edges, savers


    def pull(self, fold_n, x_train, y_train, x_test, y_test):

        self.local_resulted = []
        for j in range(len(self.name_list)):

            if self.performer is not None:
                performed = self.performer(x=x_test.iloc[:, j].values, y=y_test.values)
                if pandas.isna(performed):
                    if x_test.iloc[:, j].unique().shape[0] == 1:
                        performed = 0
                    else:
                        raise Exception("Unhandled issue causing NaN performance; check please")
                performed = numpy.abs(performed)
            else:
                performed = numpy.nan
            if self.stabilizer is not None:
                stabilized = self.stabilizer(x=x_test.iloc[:, j].values, y=y_test.values)
            else:
                stabilized = numpy.nan
            self.local_resulted.append([fold_n, self.name_list[j], performed, stabilized])
        self.local_resulted = pandas.DataFrame(data=self.local_resulted, columns=['fold_n', 'transform', 'performed', 'stabilized'])

        base_performed = self.performer(x=x_test.iloc[:, self.name_list.index(self.base_transform)].values, y=y_test.values)
        self.local_resulted['base_performed'] = numpy.abs(base_performed)

        if self.global_resulted is None:
            self.global_resulted = self.local_resulted.copy()
        else:
            self.global_resulted = pandas.concat((self.global_resulted, self.local_resulted), axis=0, ignore_index=False)

    def collapse(self):

        # https://www.statisticssolutions.com/free-resources/directory-of-statistical-analyses/paired-sample-t-test/
        self.global_resulted['perf_diff'] = self.global_resulted['performed'] - self.global_resulted['base_performed']
        global_resulted_agg_part = self.global_resulted.groupby(by='transform')
        global_resulted_agg_mean = global_resulted_agg_part[['perf_diff', 'performed', 'base_performed']].mean().rename(columns={'perf_diff': 'perf_diff_mean', 'performed': 'performed_mean', 'base_performed': 'base_performed_mean'})
        global_resulted_agg_std = global_resulted_agg_part[['perf_diff', 'performed', 'base_performed']].std().rename(columns={'perf_diff': 'perf_diff_std', 'performed': 'performed_std', 'base_performed': 'base_performed_std'})
        global_resulted_agg_count = global_resulted_agg_part[['perf_diff']].count().rename(columns={'perf_diff': 'n'})
        global_resulted_agg = global_resulted_agg_mean.merge(right=global_resulted_agg_std, left_index=True, right_index=True, how='outer')
        global_resulted_agg = global_resulted_agg.merge(right=global_resulted_agg_count, left_index=True, right_index=True, how='outer')

        def tester(x, col):
            arg = x[f'{col}_mean'] / (x[f'{col}_std'] / (x['n'] ** 0.5))
            pv = 1 - stats.t.cdf(x=arg, df=x['n'] - 1)
            return pv

        # test#1 on zero mean for base (alt: greater)
        global_resulted_agg['test1_base_result'] = global_resulted_agg.apply(func=tester, axis=1, args=('base_performed',))
        # test#1 on zero mean for candidate (alt: greater)
        global_resulted_agg['test1_candidate_result'] = global_resulted_agg.apply(func=tester, axis=1, args=('performed',))
        # test#2 on mean(candidate) == mean(base) (alt: candidate greater)
        global_resulted_agg['test2_result'] = global_resulted_agg.apply(func=tester, axis=1, args=('perf_diff',))

        global_resulted_agg = global_resulted_agg.sort_values(by='test2_result')
        self.global_resulted_agg = global_resulted_agg.copy()

        cmp_alpha_thresh = 0.05

        the_chosen = []

        global_resulted_filtered = global_resulted_agg[(global_resulted_agg['test2_result'] <= cmp_alpha_thresh) & (global_resulted_agg['test1_candidate_result'] <= cmp_alpha_thresh)].copy()
        if global_resulted_filtered.shape[0] > 0:
            the_chosen += ['{0}__{1}'.format(self.name, global_resulted_filtered.index.values[0])]
        else:
            global_resulted_filtered = global_resulted_agg[global_resulted_agg['test1_base_result'] <= cmp_alpha_thresh].copy()
            if global_resulted_filtered.shape[0] > 0:
                the_chosen += ['{0}__{1}'.format(self.name, self.base_transform)]
            else:
                the_chosen += []

        return the_chosen, (self.global_resulted_agg, self.global_resulted)
