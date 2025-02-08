#


#


#
from macro.new_base import FoldGenerator
from macro.new_data_check import pod_loader

#
"""
Stage -1: Fold generator set-up & data quality report
"""

pod_loader()

n_folds = 10
# n_folds = 2
joint_lag = 12
val_rate = 0.5
overlap_rate = 0.15
fg = FoldGenerator(n_folds=n_folds, joint_lag=joint_lag, val_rate=val_rate, overlap_rate=overlap_rate)

start_date = '2007-01-01'
end_date = '2025-01-01'
