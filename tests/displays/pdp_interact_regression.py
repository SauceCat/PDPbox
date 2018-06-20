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
## Numeric and Binary feature: weekofyear and SchoolHoliday
"""

pdp_weekofyear_SchoolHoliday = pdp.pdp_interact(
    model=ross_model,
    dataset=ross_data,
    model_features=ross_features,
    features=['weekofyear', 'SchoolHoliday']
)
"""
"""

fig, axes = pdp.pdp_interact_plot(
    pdp_interact_out=pdp_weekofyear_SchoolHoliday,
    feature_names=['weekofyear', 'SchoolHoliday'],
    plot_type='contour',
    x_quantile=True,
    plot_pdp=False
)
_ = axes['pdp_inter_ax'].set_yticklabels(['Not SchoolHoliday', 'SchoolHoliday'])
"""
"""

fig, axes = pdp.pdp_interact_plot(
    pdp_interact_out=pdp_weekofyear_SchoolHoliday,
    feature_names=['weekofyear', 'SchoolHoliday'],
    plot_type='contour',
    x_quantile=True,
    plot_pdp=True
)
_ = axes['pdp_inter_ax']['_pdp_inter_ax'].set_yticklabels(
    ['Not SchoolHoliday', 'SchoolHoliday']
)
"""
"""

fig, axes = pdp.pdp_interact_plot(
    pdp_interact_out=pdp_weekofyear_SchoolHoliday,
    feature_names=['weekofyear', 'SchoolHoliday'],
    plot_type='grid',
    plot_pdp=False
)
"""
"""

fig, axes = pdp.pdp_interact_plot(
    pdp_interact_out=pdp_weekofyear_SchoolHoliday,
    feature_names=['weekofyear', 'SchoolHoliday'],
    plot_type='grid',
    plot_pdp=True
)
"""
"""
"""
## Onehot encoding and Binary feature: StoreType and SchoolHoliday
"""

pdp_StoreType_SchoolHoliday = pdp.pdp_interact(
    model=ross_model,
    dataset=ross_data,
    model_features=ross_features,
    features=[
        ['StoreType_a', 'StoreType_b', 'StoreType_c', 'StoreType_d'],
        'SchoolHoliday'
    ]
)
"""
"""

fig, axes = pdp.pdp_interact_plot(
    pdp_interact_out=pdp_StoreType_SchoolHoliday,
    feature_names=['StoreType', 'SchoolHoliday'],
    plot_type='grid',
    plot_pdp=False
)
"""
"""

fig, axes = pdp.pdp_interact_plot(
    pdp_interact_out=pdp_StoreType_SchoolHoliday,
    feature_names=['StoreType', 'SchoolHoliday'],
    plot_type='grid',
    plot_pdp=True
)
"""
"""
