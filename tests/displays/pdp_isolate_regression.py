"""
# Test pdpbox for regression problem
### dataset: Kaggle Rossmann store dataset
"""

import pandas as pd
import numpy as np
import sys
sys.path.insert(0, '../../')

from pdpbox import pdp, get_dataset
"""
## get dataset
"""

test_ross = get_dataset.ross()
ross_data = test_ross['data']
ross_features = test_ross['features']
ross_model = test_ross['rf_model']
ross_target = test_ross['target']
"""
## Binary feature: SchoolHoliday
"""

pdp_SchoolHoliday = pdp.pdp_isolate(
    model=ross_model,
    dataset=ross_data,
    model_features=ross_features,
    feature='SchoolHoliday'
)
"""
"""

fig, axes = pdp.pdp_plot(
    pdp_isolate_out=pdp_SchoolHoliday,
    feature_name='SchoolHoliday',
    plot_lines=True,
    frac_to_plot=100
)
_ = axes['pdp_ax'].set_xticklabels(['Not SchoolHoliday', 'SchoolHoliday'])
"""
"""

fig, axes = pdp.pdp_plot(
    pdp_isolate_out=pdp_SchoolHoliday,
    feature_name='SchoolHoliday',
    plot_pts_dist=True,
    plot_lines=True,
    frac_to_plot=100
)
_ = axes['pdp_ax']['_count_ax'].set_xticklabels(
    ['Not SchoolHoliday', 'SchoolHoliday']
)
"""
"""
"""
## one-hot encoding feature: StoreType
"""

pdp_StoreType = pdp.pdp_isolate(
    model=ross_model,
    dataset=ross_data,
    model_features=ross_features,
    feature=['StoreType_a', 'StoreType_b', 'StoreType_c', 'StoreType_d']
)
"""
"""

fig, axes = pdp.pdp_plot(
    pdp_isolate_out=pdp_StoreType,
    feature_name='StoreType',
    plot_lines=True,
    frac_to_plot=100
)
"""
"""

fig, axes = pdp.pdp_plot(
    pdp_isolate_out=pdp_StoreType,
    feature_name='StoreType',
    plot_pts_dist=True,
    plot_lines=True,
    frac_to_plot=100
)
"""
"""
"""
## numeric feature: weekofyear
"""

pdp_weekofyear = pdp.pdp_isolate(
    model=ross_model,
    dataset=ross_data,
    model_features=ross_features,
    feature='weekofyear'
)
"""
"""

fig, axes = pdp.pdp_plot(
    pdp_isolate_out=pdp_weekofyear,
    feature_name='weekofyear',
    plot_lines=True,
    frac_to_plot=100
)
"""
"""

fig, axes = pdp.pdp_plot(
    pdp_isolate_out=pdp_weekofyear,
    feature_name='weekofyear',
    plot_pts_dist=True,
    plot_lines=True,
    frac_to_plot=100
)
"""
"""

fig, axes = pdp.pdp_plot(
    pdp_isolate_out=pdp_weekofyear,
    feature_name='weekofyear',
    plot_pts_dist=True,
    plot_lines=True,
    frac_to_plot=100,
    x_quantile=True,
    show_percentile=True
)
"""
"""
