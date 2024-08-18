#
import os


#
import pandas


#
from skeleton import add_entity, Graph
from dead_niggers_storage import repr_lag_generator, repr_mean_window_generator, upcast_generator, downcast_generator, measure_pearsonr, measure_kendalltau, trans_diff, trans_pct, repr_will_double, repr_n_for_double, repr_dot_alfa_generator, repr_dot_beta_generator, repr_geometric_mean_base_pct_generator

#
d = './data/'
g = Graph()

repr_lag_1 = repr_lag_generator(n_lag=1)
repr_lag_3 = repr_lag_generator(n_lag=3)
repr_lag_6 = repr_lag_generator(n_lag=6)
repr_lag_12 = repr_lag_generator(n_lag=12)
repr_lag_24 = repr_lag_generator(n_lag=24)
repr_mean_3 = repr_mean_window_generator(w=3)
repr_mean_6 = repr_mean_window_generator(w=6)
repr_mean_12 = repr_mean_window_generator(w=12)
repr_mean_24 = repr_mean_window_generator(w=24)
reprs = {'lag_1': repr_lag_1, 'lag_3': repr_lag_3, 'lag_6': repr_lag_6, 'lag_12': repr_lag_12, 'lag_24': repr_lag_24,
         'mean_1': repr_lag_1, 'mean_3': repr_lag_3, 'mean_6': repr_lag_6, 'mean_12': repr_lag_12, 'mean_24': repr_lag_24,
         'diff': trans_diff, 'pct': trans_pct, }


repr_alfa_4 = repr_dot_alfa_generator(w=4)
repr_alfa_6 = repr_dot_alfa_generator(w=6)
repr_alfa_12 = repr_dot_alfa_generator(w=12)
repr_alfa_24 = repr_dot_alfa_generator(w=24)
repr_beta_4 = repr_dot_beta_generator(w=4)
repr_beta_6 = repr_dot_beta_generator(w=6)
repr_beta_12 = repr_dot_beta_generator(w=12)
repr_beta_24 = repr_dot_beta_generator(w=24)
repr_gmean_3 = repr_geometric_mean_base_pct_generator(w=3)
repr_gmean_6 = repr_geometric_mean_base_pct_generator(w=6)
repr_gmean_12 = repr_geometric_mean_base_pct_generator(w=12)
repr_gmean_24 = repr_geometric_mean_base_pct_generator(w=24)
pct_repr = {'double': repr_will_double, 'n_double': repr_n_for_double,
            'alfa_4': repr_alfa_4, 'alfa_6': repr_alfa_6, 'alfa_12': repr_alfa_12, 'alfa_24': repr_alfa_24,
            'beta_4': repr_beta_4, 'beta_6': repr_beta_6, 'beta_12': repr_beta_12, 'beta_24': repr_beta_24,
            'gmean_3': repr_gmean_3, 'gmean_6': repr_gmean_6, 'gmean_12': repr_gmean_12, 'gmean_24': repr_gmean_24, }


for f in os.listdir(d):
    sliced = pandas.read_csv('{0}{1}'.format(d, f), na_values='.')
    sliced = sliced.rename(columns={'DATE': 'date'})
    add_entity(graph=g, data=sliced, reprs=None)

# g.summarize_freqs()

# upcast_qs = upcast_generator(freq='QS-OCT')
# upcast_ms = upcast_generator(freq='MS')

# downcast_qs = downcast_generator(freq='QS-OCT')
# downcast_as = downcast_generator(freq='AS-JAN')

# g.upcast_freqs(freqs=['QS-OCT', 'MS'], upcasters=[upcast_qs, upcast_ms], upcasters_names=['Q', 'M'])
# g.downcast_freqs(freqs=['QS-OCT', 'AS-JAN'], downcasters=[downcast_qs, downcast_as], downcasters_names=['Q', 'Y'])

g.add_representations(representations=list(reprs.values()), representations_names=list(reprs.keys()))

for j in range(len(g.entities)):
    if g.entities[j].name not in ['GDP', 'IVV', 'TLT']:
        for i in range(len(list(pct_repr.keys()))):
            representation = list(pct_repr.values())[i]
            representation_name = list(pct_repr.keys())[i]
            g.entities[j].estimate_repr(repr_estimator=representation, repr_name=representation_name)
g.time_cutoff_all_entities()

time_order_comparator = {
    'source': 0,
    'lag_1': -1,
    'lag_3': -3,
    'lag_6': -6,
    'lag_12': -12,
    'lag_24': -24,
    'mean_1': -1,
    'mean_3': -1,
    'mean_6': -1,
    'mean_12': -1,
    'mean_24': -1,
    'diff': -1,
    'pct': -1,
    'double': -1,
    'n_double': -1,
    'alfa_4': -1,
    'alfa_6': -1,
    'alfa_12': -1,
    'alfa_24': -1,
    'beta_4': -1,
    'beta_6': -1,
    'beta_12': -1,
    'beta_24': -1,
    'gmean_3': -1,
    'gmean_6': -1,
    'gmean_12': -1,
    'gmean_24': -1,

}
g.estimate_connections(measure=measure_kendalltau, thresh=0.5)
# gc = g.plot_connections_thresh(time_order_comparator=time_order_comparator, hide_sources=True, time_order_strict=True)
gc = g.plot_connections_thresh_at(at='TLT_source', time_order_comparator=time_order_comparator, hide_sources=False, time_order_strict=True)
gc.render(directory='./', view=True)

# + rsi
