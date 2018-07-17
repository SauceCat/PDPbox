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
PDP_INTERACT_PLOT_STR = 'fig, axes = pdp.pdp_interact_plot(%s)'

PLOT_PARAMS_STR = """
plot_params = {
    'title': 'PDP interact for "%s" and "%s"',
    'subtitle': 'Number of unique grid points: (%s: %d, %s: %d)',
    'contour_color':  'white',
    'font_family': 'Helvetica',
    'cmap': 'summer',
    'inter_fill_alpha': 0.6,
    'inter_fontsize': 10,
}\n
"""

# TITANIC STRINGS
TITANIC_TITLE_STR = '''
"""
# Test pdpbox for binary classification problem
## dataset: Kaggle Titanic dataset
"""\n
'''

TITANIC_NUMERIC_BINARY_FEATURE_STR = '''
"""
## Numeric and Binary feature: Fare and Sex
"""\n
'''

TITANIC_ONEHOT_BINARY_FEATURE_STR = '''
"""
## Onehot encoding and Binary feature: Embarked and Sex
"""
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

ROSS_NUMERIC_BINARY_FEATURE_STR = '''
"""
## Numeric and Binary feature: weekofyear and SchoolHoliday
"""\n
'''

ROSS_ONEHOT_BINARY_FEATURE_STR = '''
"""
## Onehot encoding and Binary feature: StoreType and SchoolHoliday
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

OTTO_NUMERIC_NUMERIC_STR = '''
"""
## Numeric and Numeric feature: feat_67 and feat_32
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
    os.system("ipython nbconvert --to=notebook --execute %s --ExecutePreprocessor.timeout=300"
              % output_path.replace('.py', '.ipynb'))
    os.remove(output_path.replace('.py', '.ipynb'))
    os.system("mv %s %s" % (output_path.replace('.py', '.nbconvert.ipynb'), output_path.replace('.py', '.ipynb')))


@pytest.mark.display
def test_pdp_interact_display_binary():
    output_path = 'tests/displays/pdp_interact_binary.py'

    # delete potential generated script and notebook
    if os.path.isfile(output_path):
        os.remove(output_path)

    with open(output_path, "a") as outbook:
        outbook.write(TITANIC_TITLE_STR)
        outbook.write(IMPORT_STR)
        outbook.write(GET_DATASET_STR)
        outbook.write(TITANIC_DATASET_STR)

        # numeric and binary feature
        # 1: default, all kinds of plots
        outbook.write(TITANIC_NUMERIC_BINARY_FEATURE_STR)
        titanic_numeric_binary_pdp_str = "pdp_fare_sex = pdp.pdp_interact(model=titanic_model, dataset=titanic_data, " \
                                         "model_features=titanic_features, features=['Fare', 'Sex'])"
        outbook.write(titanic_numeric_binary_pdp_str)
        outbook.write(CODE_GAP_STR)

        # numeric and binary feature plot config
        titanic_numeric_binary_default_config = "pdp_interact_out=pdp_fare_sex, feature_names=['Fare', 'Sex']"
        titanic_numeric_binary_add_configs = ["plot_type='contour', x_quantile=True, plot_pdp=False",
                                              "plot_type='contour', x_quantile=True, plot_pdp=True",
                                              "plot_type='grid', plot_pdp=False",
                                              "plot_type='grid', plot_pdp=True",
                                              "plot_type='grid', plot_pdp=True, figsize=(10, 10)"]
        outbook.write(PDP_INTERACT_PLOT_STR % titanic_numeric_binary_default_config)
        outbook.write(CODE_GAP_STR)
        for config in titanic_numeric_binary_add_configs:
            outbook.write(PDP_INTERACT_PLOT_STR % (titanic_numeric_binary_default_config + ', ' + config))
            outbook.write(CODE_GAP_STR)

        # 2: num_grid_points=15
        titanic_numeric_binary_pdp_str = "pdp_fare_sex = pdp.pdp_interact(model=titanic_model, dataset=titanic_data, " \
                                         "model_features=titanic_features, features=['Fare', 'Sex'], " \
                                         "num_grid_points=[15, None])"
        outbook.write(titanic_numeric_binary_pdp_str)
        outbook.write(CODE_GAP_STR)

        # numeric and binary feature plot config
        titanic_numeric_binary_add_configs = ["plot_type='contour', x_quantile=True, plot_pdp=True"]
        for config in titanic_numeric_binary_add_configs:
            outbook.write(PDP_INTERACT_PLOT_STR % (titanic_numeric_binary_default_config + ', ' + config))
            outbook.write(CODE_GAP_STR)

        # 3: plot params
        outbook.write(PLOT_PARAMS_STR % ('fare', 'gender', 'fare', 10, 'gender', 2))
        titanic_numeric_binary_pdp_str = "pdp_fare_sex = pdp.pdp_interact(model=titanic_model, dataset=titanic_data, " \
                                         "model_features=titanic_features, features=['Fare', 'Sex'])"
        outbook.write(titanic_numeric_binary_pdp_str)
        outbook.write(CODE_GAP_STR)

        # numeric and binary feature plot config
        titanic_numeric_binary_add_configs = ["plot_type='grid', plot_pdp=False, plot_params=plot_params"]
        for config in titanic_numeric_binary_add_configs:
            outbook.write(PDP_INTERACT_PLOT_STR % (titanic_numeric_binary_default_config + ', ' + config))
            outbook.write(CODE_GAP_STR)

        # onehot and binary feature
        outbook.write(TITANIC_ONEHOT_BINARY_FEATURE_STR)
        titanic_onehot_binary_pdp_str = "pdp_embark_sex = pdp.pdp_interact(model=titanic_model, dataset=titanic_data, " \
                                        "model_features=titanic_features, " \
                                        "features=[['Embarked_C', 'Embarked_S', 'Embarked_Q'], 'Sex'])"
        outbook.write(titanic_onehot_binary_pdp_str)
        outbook.write(CODE_GAP_STR)

        # onehot and binary feature plot config
        titanic_onehot_binary_default_config = "pdp_interact_out=pdp_embark_sex, feature_names=['Embark', 'Sex']"
        titanic_onehot_binary_add_configs = ["plot_type='grid', plot_pdp=False",
                                             "plot_type='grid', plot_pdp=True"]
        for config in titanic_onehot_binary_add_configs:
            outbook.write(PDP_INTERACT_PLOT_STR % (titanic_onehot_binary_default_config + ', ' + config))
            outbook.write(CODE_GAP_STR)

    outbook.close()
    generate_notebook(output_path)


@pytest.mark.display
def test_pdp_interact_display_regression():
    output_path = 'tests/displays/pdp_interact_regression.py'

    # delete potential generated script and notebook
    if os.path.isfile(output_path):
        os.remove(output_path)

    with open(output_path, "a") as outbook:
        outbook.write(ROSS_TITLE_STR)
        outbook.write(IMPORT_STR)
        outbook.write(GET_DATASET_STR)
        outbook.write(ROSS_DATASET_STR)

        # numeric and binary feature
        outbook.write(ROSS_NUMERIC_BINARY_FEATURE_STR)
        ross_numeric_binary_pdp_str = "pdp_weekofyear_SchoolHoliday = pdp.pdp_interact(model=ross_model, " \
                                      "dataset=ross_data, model_features=ross_features, " \
                                      "features=['weekofyear', 'SchoolHoliday'])"
        outbook.write(ross_numeric_binary_pdp_str)
        outbook.write(CODE_GAP_STR)

        # numeric and binary feature plot config
        ross_numeric_binary_default_config = "pdp_interact_out=pdp_weekofyear_SchoolHoliday, " \
                                             "feature_names=['weekofyear', 'SchoolHoliday']"
        ross_numeric_binary_add_configs = ["plot_type='contour', x_quantile=True, plot_pdp=False",
                                           "plot_type='contour', x_quantile=True, plot_pdp=True",
                                           "plot_type='grid', plot_pdp=False",
                                           "plot_type='grid', plot_pdp=True"]

        for config_idx, config in enumerate(ross_numeric_binary_add_configs):
            outbook.write(PDP_INTERACT_PLOT_STR % (ross_numeric_binary_default_config + ', ' + config))
            if config_idx == 0:
                outbook.write("\n_ = axes['pdp_inter_ax'].set_yticklabels(['Not SchoolHoliday', 'SchoolHoliday'])")
            if config_idx == 1:
                outbook.write("\n_ = axes['pdp_inter_ax']['_pdp_inter_ax'].set_yticklabels(["
                              "'Not SchoolHoliday', 'SchoolHoliday'])")
            outbook.write(CODE_GAP_STR)

        # onehot and binary feature
        outbook.write(ROSS_ONEHOT_BINARY_FEATURE_STR)
        ross_onehot_binary_pdp_str = "pdp_StoreType_SchoolHoliday = pdp.pdp_interact(model=ross_model, " \
                                     "dataset=ross_data, model_features=ross_features, " \
                                     "features=[['StoreType_a', 'StoreType_b', 'StoreType_c', 'StoreType_d'], 'SchoolHoliday'])"
        outbook.write(ross_onehot_binary_pdp_str)
        outbook.write(CODE_GAP_STR)

        # onehot and binary feature plot config
        ross_onehot_binary_default_config = "pdp_interact_out=pdp_StoreType_SchoolHoliday, " \
                                            "feature_names=['StoreType', 'SchoolHoliday']"
        ross_onehot_binary_add_configs = ["plot_type='grid', plot_pdp=False",
                                          "plot_type='grid', plot_pdp=True"]
        for config in ross_onehot_binary_add_configs:
            outbook.write(PDP_INTERACT_PLOT_STR % (ross_onehot_binary_default_config + ', ' + config))
            outbook.write(CODE_GAP_STR)

    outbook.close()
    generate_notebook(output_path=output_path)


@pytest.mark.display
def test_pdp_interact_display_multiclass():
    output_path = 'tests/displays/pdp_interact_multiclass.py'

    # delete potential generated script and notebook
    if os.path.isfile(output_path):
        os.remove(output_path)

    with open(output_path, "a") as outbook:
        outbook.write(OTTO_TITLE_STR)
        outbook.write(IMPORT_STR)
        outbook.write(GET_DATASET_STR)
        outbook.write(OTTO_DATASET_STR)

        # numeric feature
        outbook.write(OTTO_NUMERIC_NUMERIC_STR)
        otto_numeric_numeric_pdp_str = "pdp_67_24_rf = pdp.pdp_interact(model=otto_model, dataset=otto_data, " \
                                       "model_features=otto_features, features=['feat_67', 'feat_24'], n_jobs=4)"
        outbook.write(otto_numeric_numeric_pdp_str)
        outbook.write(CODE_GAP_STR)

        # numeric and numeric feature plot config
        otto_numeric_numeric_default_config = "pdp_interact_out=pdp_67_24_rf, feature_names=['feat_67', 'feat_24']"
        otto_numeric_numeric_add_configs = ["plot_type='contour', x_quantile=False, plot_pdp=False, ncols=3",
                                            "plot_type='contour', x_quantile=True, plot_pdp=False, ncols=3",
                                            "plot_type='contour', x_quantile=True, plot_pdp=True, ncols=3",
                                            "plot_type='contour', x_quantile=True, plot_pdp=True, "
                                            "which_classes=[0, 3, 7]",
                                            "plot_type='contour', x_quantile=True, plot_pdp=True, which_classes=[5]",
                                            "plot_type='grid', plot_pdp=False, ncols=3",
                                            "plot_type='grid', plot_pdp=True, ncols=3"]
        for config in otto_numeric_numeric_add_configs:
            outbook.write(PDP_INTERACT_PLOT_STR % (otto_numeric_numeric_default_config + ', ' + config))
            outbook.write(CODE_GAP_STR)

    outbook.close()
    generate_notebook(output_path=output_path)
