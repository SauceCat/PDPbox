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
## Binary feature: Sex
"""

pdp_sex = pdp.pdp_isolate(
    model=titanic_model,
    dataset=titanic_data,
    model_features=titanic_features,
    feature='Sex'
)
"""
"""

fig, axes = pdp.pdp_plot(pdp_isolate_out=pdp_sex, feature_name='Sex')
_ = axes['pdp_ax'].set_xticklabels(['Female', 'Male'])
"""
"""

fig, axes = pdp.pdp_plot(
    pdp_isolate_out=pdp_sex, feature_name='Sex', plot_pts_dist=True
)
_ = axes['pdp_ax']['_count_ax'].set_xticklabels(['Female', 'Male'])
"""
"""

fig, axes = pdp.pdp_plot(
    pdp_isolate_out=pdp_sex,
    feature_name='Sex',
    plot_pts_dist=True,
    figsize=(12, 8)
)
"""
"""

fig, axes = pdp.pdp_plot(
    pdp_isolate_out=pdp_sex,
    feature_name='Sex',
    plot_pts_dist=True,
    plot_lines=True,
    frac_to_plot=0.1
)
"""
"""

fig, axes = pdp.pdp_plot(
    pdp_isolate_out=pdp_sex,
    feature_name='Sex',
    plot_pts_dist=True,
    plot_lines=True,
    frac_to_plot=0.1,
    center=False
)
"""
"""

fig, axes = pdp.pdp_plot(
    pdp_isolate_out=pdp_sex,
    feature_name='Sex',
    plot_pts_dist=True,
    cluster=True,
    n_cluster_centers=10,
    cluster_method='approx'
)
"""
"""
"""
## one-hot encoding feature: Embarked
"""
pdp_embark = pdp.pdp_isolate(
    model=titanic_model,
    dataset=titanic_data,
    model_features=titanic_features,
    feature=['Embarked_C', 'Embarked_S', 'Embarked_Q']
)
"""
"""

fig, axes = pdp.pdp_plot(pdp_isolate_out=pdp_embark, feature_name='Embark')
"""
"""

fig, axes = pdp.pdp_plot(
    pdp_isolate_out=pdp_embark, feature_name='Embark', plot_pts_dist=True
)
"""
"""

fig, axes = pdp.pdp_plot(
    pdp_isolate_out=pdp_embark,
    feature_name='Embark',
    plot_pts_dist=True,
    plot_lines=True,
    frac_to_plot=0.1
)
"""
"""

fig, axes = pdp.pdp_plot(
    pdp_isolate_out=pdp_embark,
    feature_name='Embark',
    plot_pts_dist=True,
    plot_lines=True,
    frac_to_plot=0.1,
    center=False
)
"""
"""

fig, axes = pdp.pdp_plot(
    pdp_isolate_out=pdp_embark,
    feature_name='Embark',
    plot_pts_dist=True,
    cluster=True,
    n_cluster_centers=10,
    cluster_method='approx'
)
"""
"""
"""
## numeric feature: Fare
"""

pdp_fare = pdp.pdp_isolate(
    model=titanic_model,
    dataset=titanic_data,
    model_features=titanic_features,
    feature='Fare'
)
"""
"""

fig, axes = pdp.pdp_plot(pdp_isolate_out=pdp_fare, feature_name='Fare')
"""
"""

fig, axes = pdp.pdp_plot(
    pdp_isolate_out=pdp_fare, feature_name='Fare', plot_pts_dist=True
)
"""
"""

fig, axes = pdp.pdp_plot(
    pdp_isolate_out=pdp_fare,
    feature_name='Fare',
    plot_pts_dist=True,
    plot_lines=True,
    frac_to_plot=0.1
)
"""
"""

fig, axes = pdp.pdp_plot(
    pdp_isolate_out=pdp_fare,
    feature_name='Fare',
    plot_pts_dist=True,
    plot_lines=True,
    frac_to_plot=0.1,
    center=False
)
"""
"""

fig, axes = pdp.pdp_plot(
    pdp_isolate_out=pdp_fare,
    feature_name='Fare',
    plot_pts_dist=True,
    plot_lines=True,
    frac_to_plot=0.1,
    x_quantile=True
)
"""
"""

fig, axes = pdp.pdp_plot(
    pdp_isolate_out=pdp_fare,
    feature_name='Fare',
    plot_pts_dist=True,
    plot_lines=True,
    frac_to_plot=0.1,
    x_quantile=True,
    show_percentile=True
)
"""
"""

pdp_fare = pdp.pdp_isolate(
    model=titanic_model,
    dataset=titanic_data,
    model_features=titanic_features,
    feature='Fare',
    num_grid_points=15
)
"""
"""

fig, axes = pdp.pdp_plot(
    pdp_isolate_out=pdp_fare,
    feature_name='Fare',
    plot_pts_dist=True,
    plot_lines=True,
    frac_to_plot=0.1,
    x_quantile=True,
    show_percentile=True
)
"""
"""

pdp_fare = pdp.pdp_isolate(
    model=titanic_model,
    dataset=titanic_data,
    model_features=titanic_features,
    feature='Fare',
    grid_type='equal'
)
"""
"""

fig, axes = pdp.pdp_plot(
    pdp_isolate_out=pdp_fare,
    feature_name='Fare',
    plot_pts_dist=True,
    plot_lines=True,
    frac_to_plot=0.1,
    x_quantile=True
)
"""
"""

pdp_fare = pdp.pdp_isolate(
    model=titanic_model,
    dataset=titanic_data,
    model_features=titanic_features,
    feature='Fare',
    percentile_range=(5, 95)
)
"""
"""

fig, axes = pdp.pdp_plot(
    pdp_isolate_out=pdp_fare,
    feature_name='Fare',
    plot_pts_dist=True,
    plot_lines=True,
    frac_to_plot=0.1,
    x_quantile=True,
    show_percentile=True
)
"""
"""

pdp_fare = pdp.pdp_isolate(
    model=titanic_model,
    dataset=titanic_data,
    model_features=titanic_features,
    feature='Fare',
    grid_type='equal',
    grid_range=(5, 100)
)
"""
"""

fig, axes = pdp.pdp_plot(
    pdp_isolate_out=pdp_fare,
    feature_name='Fare',
    plot_pts_dist=True,
    plot_lines=True,
    frac_to_plot=0.1,
    x_quantile=True
)
"""
"""

pdp_fare = pdp.pdp_isolate(
    model=titanic_model,
    dataset=titanic_data,
    model_features=titanic_features,
    feature='Fare',
    cust_grid_points=range(5, 50, 5)
)
"""
"""

fig, axes = pdp.pdp_plot(
    pdp_isolate_out=pdp_fare,
    feature_name='Fare',
    plot_pts_dist=True,
    plot_lines=True,
    frac_to_plot=0.1,
    x_quantile=True
)
"""
"""

plot_params = {
    'title': 'PDP for feature "gender"',
    'subtitle': "Number of unique grid points: 2",
    'font_family': 'Helvetica',
    'line_cmap': 'Greens',
    'xticks_rotation': 0,
    'pdp_color': '#3288bd',
    'pdp_hl_color': '#fee08b',
    'pdp_linewidth': 2,
    'zero_color': '#d53e4f',
    'zero_linewidth': 1.5,
    'fill_color': '#abdda4',
    'fill_alpha': 0.3,
    'markersize': 4.5,
}

pdp_fare = pdp.pdp_isolate(
    model=titanic_model,
    dataset=titanic_data,
    model_features=titanic_features,
    feature='Fare'
)
"""
"""

fig, axes = pdp.pdp_plot(
    pdp_isolate_out=pdp_fare,
    feature_name='Fare',
    plot_pts_dist=True,
    plot_lines=True,
    frac_to_plot=0.1,
    x_quantile=True,
    plot_params=plot_params
)
"""
"""
