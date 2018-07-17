import os
from yapf.yapflib.yapf_api import FormatFile
import pytest


IMPORT_STR = """
import pandas as pd
import numpy as np
import sys
sys.path.insert(0, '../../')

from pdpbox import pdp, get_dataset\n
"""

GET_DATASET_STR = '''
"""
## get dataset
"""\n
'''

CODE_GAP_STR = '\n\n"""\n"""\n\n'
PDP_PLOT_STR = 'fig, axes = pdp.pdp_plot(%s)'

PLOT_PARAMS_STR = """
plot_params = {
    'title': 'PDP for feature "%s"',
    'subtitle': "Number of unique grid points: %d",
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
}\n
"""

# TITANIC STRINGS
TITANIC_TITLE_STR = '''
"""
# Test pdpbox for binary classification problem
## dataset: Kaggle Titanic dataset
"""\n
'''

TITANIC_BINARY_FEATURE_STR = '''
"""
## Binary feature: Sex
"""\n
'''

TITANIC_ONEHOT_FEATURE_STR = '''
"""
## one-hot encoding feature: Embarked
"""
'''

TITANIC_NUMERIC_STR = '''
"""
## numeric feature: Fare
"""\n
'''

TITANIC_DATASET_STR = """
test_titanic = get_dataset.titanic()
titanic_data = test_titanic['data']
titanic_features = test_titanic['features']
titanic_model = test_titanic['xgb_model']
titanic_target = test_titanic['target']\n
"""

# ROSS STRINGS
ROSS_TITLE_STR = '''
"""
# Test pdpbox for regression problem
### dataset: Kaggle Rossmann store dataset
"""\n
'''

ROSS_BINARY_FEATURE_STR = '''
"""
## Binary feature: SchoolHoliday
"""\n
'''

ROSS_ONEHOT_FEATURE_STR = '''
"""
## one-hot encoding feature: StoreType
"""\n
'''

ROSS_NUMERIC_FEATURE_STR = '''
"""
## numeric feature: weekofyear
"""\n
'''

ROSS_DATASET_STR = """
test_ross = get_dataset.ross()
ross_data = test_ross['data']
ross_features = test_ross['features']
ross_model = test_ross['rf_model']
ross_target = test_ross['target']\n
"""

# OTTO STRINGS
OTTO_TITLE_STR = '''
"""
# Test pdpbox for multi-class classification problem
### dataset: Kaggle Otto dataset
"""\n
'''

OTTO_NUMERIC_STR = '''
"""
## numeric feature: feat_67
"""\n
'''

OTTO_DATASET_STR = """
test_otto = get_dataset.otto()
otto_data = test_otto['data']
otto_features = test_otto['features']
otto_model = test_otto['rf_model']
otto_target = test_otto['target']\n
"""


def generate_notebook(output_path):
    formated_script = FormatFile(output_path, style_config='facebook')

    os.remove(output_path)
    with open(output_path, "a") as outbook:
        outbook.write(formated_script[0].replace('\r', ''))
    outbook.close()

    os.system("python -m py2nb %s %s" % (output_path, output_path.replace('.py', '.ipynb')))
    os.system("ipython nbconvert --to=notebook --execute %s" % output_path.replace('.py', '.ipynb'))
    os.remove(output_path.replace('.py', '.ipynb'))
    os.system("mv %s %s" % (output_path.replace('.py', '.nbconvert.ipynb'), output_path.replace('.py', '.ipynb')))


@pytest.mark.display
def test_display_binary():
    output_path = 'tests/displays/pdp_isolate_binary.py'

    # delete potential generated script and notebook
    if os.path.isfile(output_path):
        os.remove(output_path)

    with open(output_path, "a") as outbook:
        outbook.write(TITANIC_TITLE_STR)
        outbook.write(IMPORT_STR)
        outbook.write(GET_DATASET_STR)
        outbook.write(TITANIC_DATASET_STR)

        # binary feature
        outbook.write(TITANIC_BINARY_FEATURE_STR)
        titanic_binary_pdp_str = "pdp_sex = pdp.pdp_isolate(model=titanic_model, dataset=titanic_data, " \
                                 "model_features=titanic_features, feature='Sex')"
        outbook.write(titanic_binary_pdp_str)
        outbook.write(CODE_GAP_STR)

        # binary feature plot config
        titanic_binary_default_config = "pdp_isolate_out=pdp_sex, feature_name='Sex'"
        titanic_binary_add_configs = ["plot_pts_dist=True",
                                      "plot_pts_dist=True, figsize=(12, 8)",
                                      "plot_pts_dist=True, plot_lines=True, frac_to_plot=0.1",
                                      "plot_pts_dist=True, plot_lines=True, frac_to_plot=0.1, center=False",
                                      "plot_pts_dist=True, cluster=True, n_cluster_centers=10, cluster_method='approx'"]
        outbook.write(PDP_PLOT_STR % titanic_binary_default_config)
        outbook.write("\n_ = axes['pdp_ax'].set_xticklabels(['Female', 'Male'])")
        outbook.write(CODE_GAP_STR)
        for config_idx, config in enumerate(titanic_binary_add_configs):
            outbook.write(PDP_PLOT_STR % (titanic_binary_default_config + ', ' + config))
            if config_idx == 0:
                outbook.write("\n_ = axes['pdp_ax']['_count_ax'].set_xticklabels(['Female', 'Male'])")
            outbook.write(CODE_GAP_STR)

        # onehot feature
        outbook.write(TITANIC_ONEHOT_FEATURE_STR)
        titanic_onehot_pdp_str = "pdp_embark = pdp.pdp_isolate(model=titanic_model, dataset=titanic_data, " \
                                 "model_features=titanic_features, feature=['Embarked_C', 'Embarked_S', 'Embarked_Q'])"
        outbook.write(titanic_onehot_pdp_str)
        outbook.write(CODE_GAP_STR)

        # onehot feature plot config
        titanic_onehot_default_config = "pdp_isolate_out=pdp_embark, feature_name='Embark'"
        titanic_onehot_add_configs = ["plot_pts_dist=True",
                                      "plot_pts_dist=True, plot_lines=True, frac_to_plot=0.1",
                                      "plot_pts_dist=True, plot_lines=True, frac_to_plot=0.1, center=False",
                                      "plot_pts_dist=True, cluster=True, n_cluster_centers=10, cluster_method='approx'"]
        outbook.write(PDP_PLOT_STR % titanic_onehot_default_config)
        outbook.write(CODE_GAP_STR)
        for config in titanic_onehot_add_configs:
            outbook.write(PDP_PLOT_STR % (titanic_onehot_default_config + ', ' + config))
            outbook.write(CODE_GAP_STR)

        # numeric feature
        # 1: default, all kinds of plots
        outbook.write(TITANIC_NUMERIC_STR)
        titanic_numeric_pdp_str = "pdp_fare = pdp.pdp_isolate(model=titanic_model, dataset=titanic_data, " \
                                  "model_features=titanic_features, feature='Fare')"
        outbook.write(titanic_numeric_pdp_str)
        outbook.write(CODE_GAP_STR)

        # numeric feature plot config
        titanic_numeric_default_config = "pdp_isolate_out=pdp_fare, feature_name='Fare'"
        titanic_numeric_add_configs = ["plot_pts_dist=True",
                                       "plot_pts_dist=True, plot_lines=True, frac_to_plot=0.1",
                                       "plot_pts_dist=True, plot_lines=True, frac_to_plot=0.1, center=False",
                                       "plot_pts_dist=True, plot_lines=True, frac_to_plot=0.1, x_quantile=True",
                                       "plot_pts_dist=True, plot_lines=True, frac_to_plot=0.1, x_quantile=True, "
                                       "show_percentile=True"]
        outbook.write(PDP_PLOT_STR % titanic_numeric_default_config)
        outbook.write(CODE_GAP_STR)
        for config in titanic_numeric_add_configs:
            outbook.write(PDP_PLOT_STR % (titanic_numeric_default_config + ', ' + config))
            outbook.write(CODE_GAP_STR)

        # 2: num_grid_points=15
        titanic_numeric_pdp_str = "pdp_fare = pdp.pdp_isolate(model=titanic_model, dataset=titanic_data, " \
                                  "model_features=titanic_features, feature='Fare', num_grid_points=15)"
        outbook.write(titanic_numeric_pdp_str)
        outbook.write(CODE_GAP_STR)

        # numeric feature plot config
        titanic_numeric_add_configs = [
            "plot_pts_dist=True, plot_lines=True, frac_to_plot=0.1, x_quantile=True, show_percentile=True"]
        for config in titanic_numeric_add_configs:
            outbook.write(PDP_PLOT_STR % (titanic_numeric_default_config + ', ' + config))
            outbook.write(CODE_GAP_STR)

        # 3: grid_type='equal'
        titanic_numeric_pdp_str = "pdp_fare = pdp.pdp_isolate(model=titanic_model, dataset=titanic_data, " \
                                  "model_features=titanic_features, feature='Fare', grid_type='equal')"
        outbook.write(titanic_numeric_pdp_str)
        outbook.write(CODE_GAP_STR)

        # numeric feature plot config
        titanic_numeric_add_configs = ["plot_pts_dist=True, plot_lines=True, frac_to_plot=0.1, x_quantile=True"]
        for config in titanic_numeric_add_configs:
            outbook.write(PDP_PLOT_STR % (titanic_numeric_default_config + ', ' + config))
            outbook.write(CODE_GAP_STR)

        # 4: percentile_range=(5, 95)
        titanic_numeric_pdp_str = "pdp_fare = pdp.pdp_isolate(model=titanic_model, dataset=titanic_data, " \
                                  "model_features=titanic_features, feature='Fare', percentile_range=(5, 95))"
        outbook.write(titanic_numeric_pdp_str)
        outbook.write(CODE_GAP_STR)

        # numeric feature plot config
        titanic_numeric_add_configs = [
            "plot_pts_dist=True, plot_lines=True, frac_to_plot=0.1, x_quantile=True, show_percentile=True"]
        for config in titanic_numeric_add_configs:
            outbook.write(PDP_PLOT_STR % (titanic_numeric_default_config + ', ' + config))
            outbook.write(CODE_GAP_STR)

        # 5: grid_type='equal', grid_range=(5, 100)
        titanic_numeric_pdp_str = "pdp_fare = pdp.pdp_isolate(model=titanic_model, dataset=titanic_data, " \
                                  "model_features=titanic_features, feature='Fare', grid_type='equal', " \
                                  "grid_range=(5, 100))"
        outbook.write(titanic_numeric_pdp_str)
        outbook.write(CODE_GAP_STR)

        # numeric feature plot config
        titanic_numeric_add_configs = ["plot_pts_dist=True, plot_lines=True, frac_to_plot=0.1, x_quantile=True"]
        for config in titanic_numeric_add_configs:
            outbook.write(PDP_PLOT_STR % (titanic_numeric_default_config + ', ' + config))
            outbook.write(CODE_GAP_STR)

        # 6: cust_grid_points=range(5, 50, 5)
        titanic_numeric_pdp_str = "pdp_fare = pdp.pdp_isolate(model=titanic_model, dataset=titanic_data, " \
                                  "model_features=titanic_features, feature='Fare', cust_grid_points=range(5, 50, 5))"
        outbook.write(titanic_numeric_pdp_str)
        outbook.write(CODE_GAP_STR)

        # numeric feature plot config
        titanic_numeric_add_configs = ["plot_pts_dist=True, plot_lines=True, frac_to_plot=0.1, x_quantile=True"]
        for config in titanic_numeric_add_configs:
            outbook.write(PDP_PLOT_STR % (titanic_numeric_default_config + ', ' + config))
            outbook.write(CODE_GAP_STR)

        # plot params
        outbook.write(PLOT_PARAMS_STR % ('gender', 2))
        titanic_numeric_pdp_str = "pdp_fare = pdp.pdp_isolate(model=titanic_model, dataset=titanic_data, " \
                                  "model_features=titanic_features, feature='Fare')"
        outbook.write(titanic_numeric_pdp_str)
        outbook.write(CODE_GAP_STR)

        # numeric feature plot config
        titanic_numeric_add_configs = ["plot_pts_dist=True, plot_lines=True, frac_to_plot=0.1, "
                                       "x_quantile=True, plot_params=plot_params"]
        for config in titanic_numeric_add_configs:
            outbook.write(PDP_PLOT_STR % (titanic_numeric_default_config + ', ' + config))
            outbook.write(CODE_GAP_STR)

    outbook.close()
    generate_notebook(output_path)


@pytest.mark.display
def test_display_regression():
    output_path = 'tests/displays/pdp_isolate_regression.py'

    # delete potential generated script and notebook
    if os.path.isfile(output_path):
        os.remove(output_path)

    with open(output_path, "a") as outbook:
        outbook.write(ROSS_TITLE_STR)
        outbook.write(IMPORT_STR)
        outbook.write(GET_DATASET_STR)
        outbook.write(ROSS_DATASET_STR)

        # binary feature
        outbook.write(ROSS_BINARY_FEATURE_STR)
        ross_binary_pdp_str = "pdp_SchoolHoliday = pdp.pdp_isolate(model=ross_model, dataset=ross_data, " \
                              "model_features=ross_features, feature='SchoolHoliday')"
        outbook.write(ross_binary_pdp_str)
        outbook.write(CODE_GAP_STR)

        # binary feature plot config
        ross_binary_default_config = "pdp_isolate_out=pdp_SchoolHoliday, feature_name='SchoolHoliday'"
        ross_binary_add_configs = ["plot_lines=True, frac_to_plot=100",
                                   "plot_pts_dist=True, plot_lines=True, frac_to_plot=100"]

        for config_idx, config in enumerate(ross_binary_add_configs):
            outbook.write(PDP_PLOT_STR % (ross_binary_default_config + ', ' + config))
            if config_idx == 0:
                outbook.write("\n_ = axes['pdp_ax'].set_xticklabels(['Not SchoolHoliday', 'SchoolHoliday'])")
            if config_idx == 1:
                outbook.write("\n_ = axes['pdp_ax']['_count_ax'].set_xticklabels(["
                              "'Not SchoolHoliday', 'SchoolHoliday'])")
            outbook.write(CODE_GAP_STR)

        # onehot feature
        outbook.write(ROSS_ONEHOT_FEATURE_STR)
        ross_onehot_pdp_str = "pdp_StoreType = pdp.pdp_isolate(model=ross_model, dataset=ross_data, " \
                              "model_features=ross_features, " \
                              "feature=['StoreType_a', 'StoreType_b', 'StoreType_c', 'StoreType_d'])"
        outbook.write(ross_onehot_pdp_str)
        outbook.write(CODE_GAP_STR)

        # onehot feature plot config
        ross_onehot_default_config = "pdp_isolate_out=pdp_StoreType, feature_name='StoreType'"
        ross_onehot_add_configs = ["plot_lines=True, frac_to_plot=100",
                                   "plot_pts_dist=True, plot_lines=True, frac_to_plot=100"]
        for config in ross_onehot_add_configs:
            outbook.write(PDP_PLOT_STR % (ross_onehot_default_config + ', ' + config))
            outbook.write(CODE_GAP_STR)

        # numeric feature
        # 1
        outbook.write(ROSS_NUMERIC_FEATURE_STR)
        ross_numeric_pdp_str = "pdp_weekofyear = pdp.pdp_isolate(model=ross_model, dataset=ross_data, " \
                               "model_features=ross_features, feature='weekofyear')"
        outbook.write(ross_numeric_pdp_str)
        outbook.write(CODE_GAP_STR)

        # numeric feature plot config
        ross_numeric_default_config = "pdp_isolate_out=pdp_weekofyear, feature_name='weekofyear'"
        ross_numeric_add_configs = ["plot_lines=True, frac_to_plot=100",
                                    "plot_pts_dist=True, plot_lines=True, frac_to_plot=100",
                                    "plot_pts_dist=True, plot_lines=True, frac_to_plot=100, x_quantile=True, "
                                    "show_percentile=True"]
        for config in ross_numeric_add_configs:
            outbook.write(PDP_PLOT_STR % (ross_numeric_default_config + ', ' + config))
            outbook.write(CODE_GAP_STR)

    outbook.close()
    generate_notebook(output_path=output_path)


@pytest.mark.display
def test_display_multiclass():
    output_path = 'tests/displays/pdp_isolate_multiclass.py'

    # delete potential generated script and notebook
    if os.path.isfile(output_path):
        os.remove(output_path)

    with open(output_path, "a") as outbook:
        outbook.write(OTTO_TITLE_STR)
        outbook.write(IMPORT_STR)
        outbook.write(GET_DATASET_STR)
        outbook.write(OTTO_DATASET_STR)

        # numeric feature
        outbook.write(OTTO_NUMERIC_STR)
        otto_numeric_pdp_str = "pdp_feat_67_rf = pdp.pdp_isolate(model=otto_model, dataset=otto_data, " \
                               "model_features=otto_features, feature='feat_67')"
        outbook.write(otto_numeric_pdp_str)
        outbook.write(CODE_GAP_STR)

        # numeric feature plot config
        otto_numeric_default_config = "pdp_isolate_out=pdp_feat_67_rf, feature_name='feat_67'"
        otto_numeric_add_configs = ["center=True, x_quantile=True, ncols=3",
                                    "center=True, x_quantile=True, ncols=2, plot_lines=True, "
                                    "frac_to_plot=100, which_classes=[0, 3, 7]",
                                    "center=True, x_quantile=True, ncols=2, plot_lines=True, "
                                    "frac_to_plot=100, which_classes=[0, 3, 7], plot_pts_dist=True",
                                    "center=True, x_quantile=True, ncols=2, plot_lines=True, frac_to_plot=100, "
                                    "which_classes=[5], plot_pts_dist=True"]
        for config in otto_numeric_add_configs:
            outbook.write(PDP_PLOT_STR % (otto_numeric_default_config + ', ' + config))
            outbook.write(CODE_GAP_STR)

    outbook.close()
    generate_notebook(output_path=output_path)
