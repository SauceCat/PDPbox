"""
# Test pdpbox for binary classification problem
## dataset: Kaggle Titanic dataset
"""

import pandas as pd
import numpy as np
import sys
sys.path.insert(0, '../../')

from pdpbox import pdp, get_dataset
"""
## get dataset
"""

test_titanic = get_dataset.titanic()
titanic_data = test_titanic['data']
titanic_features = test_titanic['features']
titanic_model = test_titanic['xgb_model']
titanic_target = test_titanic['target']
"""
## Numeric and Binary feature: Fare and Sex
"""

pdp_fare_sex = pdp.pdp_interact(
    model=titanic_model,
    dataset=titanic_data,
    model_features=titanic_features,
    features=['Fare', 'Sex']
)
"""
"""

fig, axes = pdp.pdp_interact_plot(
    pdp_interact_out=pdp_fare_sex, feature_names=['Fare', 'Sex']
)
"""
"""

fig, axes = pdp.pdp_interact_plot(
    pdp_interact_out=pdp_fare_sex,
    feature_names=['Fare', 'Sex'],
    plot_type='contour',
    x_quantile=True,
    plot_pdp=False
)
"""
"""

fig, axes = pdp.pdp_interact_plot(
    pdp_interact_out=pdp_fare_sex,
    feature_names=['Fare', 'Sex'],
    plot_type='contour',
    x_quantile=True,
    plot_pdp=True
)
"""
"""

fig, axes = pdp.pdp_interact_plot(
    pdp_interact_out=pdp_fare_sex,
    feature_names=['Fare', 'Sex'],
    plot_type='grid',
    plot_pdp=False
)
"""
"""

fig, axes = pdp.pdp_interact_plot(
    pdp_interact_out=pdp_fare_sex,
    feature_names=['Fare', 'Sex'],
    plot_type='grid',
    plot_pdp=True
)
"""
"""

fig, axes = pdp.pdp_interact_plot(
    pdp_interact_out=pdp_fare_sex,
    feature_names=['Fare', 'Sex'],
    plot_type='grid',
    plot_pdp=True,
    figsize=(10, 10)
)
"""
"""

pdp_fare_sex = pdp.pdp_interact(
    model=titanic_model,
    dataset=titanic_data,
    model_features=titanic_features,
    features=['Fare', 'Sex'],
    num_grid_points=[15, None]
)
"""
"""

fig, axes = pdp.pdp_interact_plot(
    pdp_interact_out=pdp_fare_sex,
    feature_names=['Fare', 'Sex'],
    plot_type='contour',
    x_quantile=True,
    plot_pdp=True
)
"""
"""

plot_params = {
    'title': 'PDP interact for "fare" and "gender"',
    'subtitle': 'Number of unique grid points: (fare: 10, gender: 2)',
    'contour_color': 'white',
    'font_family': 'Helvetica',
    'cmap': 'summer',
    'inter_fill_alpha': 0.6,
    'inter_fontsize': 10,
}

pdp_fare_sex = pdp.pdp_interact(
    model=titanic_model,
    dataset=titanic_data,
    model_features=titanic_features,
    features=['Fare', 'Sex']
)
"""
"""

fig, axes = pdp.pdp_interact_plot(
    pdp_interact_out=pdp_fare_sex,
    feature_names=['Fare', 'Sex'],
    plot_type='grid',
    plot_pdp=False,
    plot_params=plot_params
)
"""
"""
"""
## Onehot encoding and Binary feature: Embarked and Sex
"""
pdp_embark_sex = pdp.pdp_interact(
    model=titanic_model,
    dataset=titanic_data,
    model_features=titanic_features,
    features=[['Embarked_C', 'Embarked_S', 'Embarked_Q'], 'Sex']
)
"""
"""

fig, axes = pdp.pdp_interact_plot(
    pdp_interact_out=pdp_embark_sex,
    feature_names=['Embark', 'Sex'],
    plot_type='grid',
    plot_pdp=False
)
"""
"""

fig, axes = pdp.pdp_interact_plot(
    pdp_interact_out=pdp_embark_sex,
    feature_names=['Embark', 'Sex'],
    plot_type='grid',
    plot_pdp=True
)
"""
"""
