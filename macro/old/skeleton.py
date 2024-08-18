#


#
import numpy
import pandas
import seaborn
from graphviz import Digraph


#


#
freqs_ordered = {'AS-JAN': 1, 'QS-OCT': 2, 'MS': 3}


def time_order(comparator, name):
    for x in comparator.keys():
        if x in name:
            return comparator[x]
    raise KeyError()


def freq_greater(source, than, equal=False):
    if equal:
        return freqs_ordered[source] >= freqs_ordered[than]
    else:
        return freqs_ordered[source] > freqs_ordered[than]


def add_entity(graph, data, reprs=None):
    data['date'] = pandas.to_datetime(data['date'])
    data = data.set_index('date')
    assert data.shape[1] == 1
    name = data.columns[0]
    data = data[name]
    data = data.fillna(data.rolling(window=6, min_periods=1).mean())
    data = data.fillna(method='ffill')
    entity = Entity(source_series=data, name=name)
    if reprs is not None:
        for key, value in reprs.items():
            entity.estimate_repr(repr_estimator=value, repr_name=key)
        entity.tight_dates()
    graph.entities.append(entity)


def product(array):
    result = numpy.array([[(array[i], array[j]) for j in range(len(array))] for i in range(len(array))])
    return result


class Entity:
    def __init__(self, source_series, name):
        self.source_series = source_series
        self.series = [source_series]
        self.names = ['{0}_source'.format(name)]
        self.name = name
        self.min_date = min(source_series.index)
        self.max_date = max(source_series.index)
    @property
    def freq(self):
        return pandas.infer_freq(self.source_series.index)
    def estimate_repr(self, repr_estimator, repr_name):
        estimated = repr_estimator(self.source_series.copy())
        # estimated = pandas.Series(data=estimated, index=self.source_series.index)
        self.series.append(estimated)
        self.names.append('{0}_{1}'.format(self.name, repr_name))
        self.tight_dates()
    def tight_dates(self):
        min_dates = []
        max_dates = []
        for se in self.series:
            mask = ~pandas.isna(se)
            dates = se.index[mask]
            min_dates.append(dates.min())
            max_dates.append(dates.max())
        self.min_date = max(min_dates)
        self.max_date = min(max_dates)
        self.set_date_bounds(self.min_date, self.max_date)
    def set_date_bounds(self, min_date, max_date):
        self.min_date = min_date
        self.max_date = max_date
        for j in range(len(self.series)):
            self.series[j] = self.series[j][(self.series[j].index >= self.min_date) *
                                            (self.series[j].index <= self.max_date)]
    @property
    def repr_names(self):
        return self.names
    @property
    def repr(self):
        return self.series
    @property
    def repr_name(self):
        return self.name
    @property
    def date_bounds(self):
        return self.min_date, self.max_date


class Graph:
    def __init__(self):
        self.entities = []
        self.connections = None
        self.connections_thresh_w = None
        self.connections_measure = None
        self.connections_names_mx = None
    def print_all_freqs(self):
        for e in self.entities:
            print(e.name, e.freq)
    def summarize_freqs(self):
        freqs = numpy.unique([e.freq for e in self.entities])
        return freqs
    def upcast_freqs(self, freqs, upcasters, upcasters_names):
        for i in range(len(upcasters)):
            freq = freqs[i]
            splitter = upcasters[i]
            splitter_name = upcasters_names[i]
            for j in range(len(self.entities)):
                if not freq_greater(source=self.entities[j].freq, than=freq, equal=True):
                    self.entities[j].estimate_repr(repr_estimator=splitter, repr_name=splitter_name)
    def downcast_freqs(self, freqs, downcasters, downcasters_names):
        for i in range(len(downcasters)):
            freq = freqs[i]
            filler = downcasters[i]
            filler_name = downcasters_names[i]
            for j in range(len(self.entities)):
                if freq_greater(source=self.entities[j].freq, than=freq, equal=False):
                    self.entities[j].estimate_repr(repr_estimator=filler, repr_name=filler_name)
    def time_cutoff_all_entities(self):
        min_dates, max_dates = [], []
        for entity in self.entities:
            local_min, local_max = entity.date_bounds
            min_dates.append(local_min)
            max_dates.append(local_max)
        global_min = max(min_dates)
        global_max = min(max_dates)
        for j in range(len(self.entities)):
            self.entities[j].set_date_bounds(global_min, global_max)
    def add_representations(self, representations, representations_names):
        for i in range(len(representations)):
            representation = representations[i]
            representation_name = representations_names[i]
            for j in range(len(self.entities)):
                self.entities[j].estimate_repr(repr_estimator=representation, repr_name=representation_name)
        self.time_cutoff_all_entities()
    def estimate_connections(self, measure, thresh=0.7):
        self.connections_measure = measure
        reprs = [x for r in self.entities for x in r.repr]
        reprs_names = numpy.array([x for r in self.entities for x in r.repr_names])
        self.connections_names_mx = product(reprs_names)
        corr = numpy.full(shape=(len(reprs_names), len(reprs_names)), fill_value=numpy.nan)
        for i in range(len(reprs_names)):
            for j in range(len(reprs_names)):
                if i != j:
                    corr[i, j] = measure(reprs[i], reprs[j])
        corr = pandas.DataFrame(data=corr, index=reprs_names, columns=reprs_names)
        self.connections = corr
        self.connections_thresh_w = corr.abs() > thresh
    def plot_connections(self):
        ...
    def plot_connections_thresh(self, time_order_comparator, hide_sources=False, time_order_strict=False):
        gc = Digraph(comment='pydge')
        gc.attr(sep="200")

        palette = seaborn.color_palette("flare", self.connections.shape[0]).as_hex()
        n = -1
        for j in range(len(self.entities)):
            group_name = self.entities[j].repr_name
            names = self.entities[j].repr_names
            # with gc.subgraph(name=group_name) as sgc:
            n += 1
            #     sgc.attr(label=group_name)
            for i in range(len(names)):
                name = names[i]
                gc.node(name=name, label=name,
                        color='#000000',
                        fillcolor=palette[n],
                        penwidth='1', style='filled')

        for i in range(self.connections_names_mx[self.connections_thresh_w, :].shape[0]):
            if hide_sources:
                pass_criteria = '_source' not in self.connections_names_mx[self.connections_thresh_w, :][i][0]
            else:
                pass_criteria = True
            if time_order_strict:
                time_order_criteria = time_order(comparator=time_order_comparator,
                                                 name=self.connections_names_mx[self.connections_thresh_w, :][i][
                                                     0]) < time_order(comparator=time_order_comparator, name=
                self.connections_names_mx[self.connections_thresh_w, :][i][1])
            else:
                time_order_criteria = time_order(comparator=time_order_comparator,
                                                 name=self.connections_names_mx[self.connections_thresh_w, :][i][
                                                     0]) <= time_order(comparator=time_order_comparator, name=
                self.connections_names_mx[self.connections_thresh_w, :][i][1])

            if pass_criteria and time_order_criteria:
                value = self.connections.loc[self.connections_names_mx[self.connections_thresh_w, :][i][0], self.connections_names_mx[self.connections_thresh_w, :][i][1]]
                gc.edge(tail_name=self.connections_names_mx[self.connections_thresh_w, :][i][0],
                        head_name=self.connections_names_mx[self.connections_thresh_w, :][i][1],
                        label='{0:.4f}'.format(value), penwidt='1',  # penwidth='0:.4f'.format(value),
                        style='solid')

        return gc
    def plot_connections_thresh_at(self, at, time_order_comparator, hide_sources=False, time_order_strict=False):
        gc = Digraph(comment='pydge')
        # gc.attr(sep="200")

        palette = seaborn.color_palette("flare", self.connections.shape[0]).as_hex()
        n = -1
        for j in range(len(self.entities)):
            group_name = self.entities[j].repr_name
            names = self.entities[j].repr_names
            # with gc.subgraph(name=group_name) as sgc:
            n += 1
            #     sgc.attr(label=group_name)
            for i in range(len(names)):
                name = names[i]
                gc.node(name=name, label=name,
                        color='#000000',
                        fillcolor=palette[n],
                        penwidth='1', style='filled')

        for i in range(self.connections_names_mx[self.connections_thresh_w, :].shape[0]):
            if hide_sources:
                pass_criteria = ('_source' not in self.connections_names_mx[self.connections_thresh_w, :][i][0]) and \
                                (at == self.connections_names_mx[self.connections_thresh_w, :][i][1])
            else:
                pass_criteria = at == self.connections_names_mx[self.connections_thresh_w, :][i][1]
            if time_order_strict:
                time_order_criteria = time_order(comparator=time_order_comparator,
                                                 name=self.connections_names_mx[self.connections_thresh_w, :][i][
                                                     0]) < time_order(comparator=time_order_comparator, name=
                self.connections_names_mx[self.connections_thresh_w, :][i][1])
            else:
                time_order_criteria = time_order(comparator=time_order_comparator,
                                                 name=self.connections_names_mx[self.connections_thresh_w, :][i][
                                                     0]) <= time_order(comparator=time_order_comparator, name=
                self.connections_names_mx[self.connections_thresh_w, :][i][1])

            if pass_criteria and time_order_criteria:
                value = self.connections.loc[self.connections_names_mx[self.connections_thresh_w, :][i][0], self.connections_names_mx[self.connections_thresh_w, :][i][1]]
                gc.edge(tail_name=self.connections_names_mx[self.connections_thresh_w, :][i][0],
                        head_name=self.connections_names_mx[self.connections_thresh_w, :][i][1],
                        label='{0:.4f}'.format(value), penwidt='1',  # penwidth='0:.4f'.format(value),
                        style='solid')

        return gc