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
## Numeric and Numeric feature: feat_67 and feat_32
"""

pdp_67_24_rf = pdp.pdp_interact(
    model=otto_model,
    dataset=otto_data,
    model_features=otto_features,
    features=['feat_67', 'feat_24'],
    n_jobs=4
)
"""
"""

fig, axes = pdp.pdp_interact_plot(
    pdp_interact_out=pdp_67_24_rf,
    feature_names=['feat_67', 'feat_24'],
    plot_type='contour',
    x_quantile=False,
    plot_pdp=False,
    ncols=3
)
"""
"""

fig, axes = pdp.pdp_interact_plot(
    pdp_interact_out=pdp_67_24_rf,
    feature_names=['feat_67', 'feat_24'],
    plot_type='contour',
    x_quantile=True,
    plot_pdp=False,
    ncols=3
)
"""
"""

fig, axes = pdp.pdp_interact_plot(
    pdp_interact_out=pdp_67_24_rf,
    feature_names=['feat_67', 'feat_24'],
    plot_type='contour',
    x_quantile=True,
    plot_pdp=True,
    ncols=3
)
"""
"""

fig, axes = pdp.pdp_interact_plot(
    pdp_interact_out=pdp_67_24_rf,
    feature_names=['feat_67', 'feat_24'],
    plot_type='contour',
    x_quantile=True,
    plot_pdp=True,
    which_classes=[0, 3, 7]
)
"""
"""

fig, axes = pdp.pdp_interact_plot(
    pdp_interact_out=pdp_67_24_rf,
    feature_names=['feat_67', 'feat_24'],
    plot_type='contour',
    x_quantile=True,
    plot_pdp=True,
    which_classes=[5]
)
"""
"""

fig, axes = pdp.pdp_interact_plot(
    pdp_interact_out=pdp_67_24_rf,
    feature_names=['feat_67', 'feat_24'],
    plot_type='grid',
    plot_pdp=False,
    ncols=3
)
"""
"""

fig, axes = pdp.pdp_interact_plot(
    pdp_interact_out=pdp_67_24_rf,
    feature_names=['feat_67', 'feat_24'],
    plot_type='grid',
    plot_pdp=True,
    ncols=3
)
"""
"""
