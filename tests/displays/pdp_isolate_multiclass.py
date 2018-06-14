"""
# Test pdpbox for multi-class classification problem
### dataset: Kaggle Otto dataset
"""

import pandas as pd
import numpy as np
import sys
sys.path.insert(0, '../../')

from pdpbox import pdp, get_dataset
"""
## get dataset
"""

test_otto = get_dataset.otto()
otto_data = test_otto['data']
otto_features = test_otto['features']
otto_model = test_otto['rf_model']
otto_target = test_otto['target']
"""
## numeric feature: feat_67
"""

pdp_feat_67_rf = pdp.pdp_isolate(
    model=otto_model,
    dataset=otto_data,
    model_features=otto_features,
    feature='feat_67'
)
"""
"""

fig, axes = pdp.pdp_plot(
    pdp_isolate_out=pdp_feat_67_rf,
    feature_name='feat_67',
    center=True,
    x_quantile=True,
    ncols=3
)
"""
"""

fig, axes = pdp.pdp_plot(
    pdp_isolate_out=pdp_feat_67_rf,
    feature_name='feat_67',
    center=True,
    x_quantile=True,
    ncols=2,
    plot_lines=True,
    frac_to_plot=100,
    which_classes=[0, 3, 7]
)
"""
"""

fig, axes = pdp.pdp_plot(
    pdp_isolate_out=pdp_feat_67_rf,
    feature_name='feat_67',
    center=True,
    x_quantile=True,
    ncols=2,
    plot_lines=True,
    frac_to_plot=100,
    which_classes=[0, 3, 7],
    plot_pts_dist=True
)
"""
"""

fig, axes = pdp.pdp_plot(
    pdp_isolate_out=pdp_feat_67_rf,
    feature_name='feat_67',
    center=True,
    x_quantile=True,
    ncols=2,
    plot_lines=True,
    frac_to_plot=100,
    which_classes=[5],
    plot_pts_dist=True
)
"""
"""
