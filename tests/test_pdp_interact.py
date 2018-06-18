
import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
from pandas.testing import assert_frame_equal
import pandas as pd
import matplotlib

from pdpbox.pdp import pdp_interact, pdp_interact_plot


@pytest.mark.skip(reason="slow")
def test_pdp_interact_binary(titanic_model, titanic_data, titanic_features):
    # binary, numeric
    pdp_interact_out = pdp_interact(model=titanic_model, dataset=titanic_data,
                                    model_features=titanic_features, features=['Fare', 'Sex'])
    assert pdp_interact_out._type == 'PDPInteract_instance'
    assert pdp_interact_out.n_classes == 2
    assert pdp_interact_out.which_class is None
    assert pdp_interact_out.features == ['Fare', 'Sex']
    assert pdp_interact_out.feature_types == ['numeric', 'binary']
    assert len(pdp_interact_out.feature_grids) == 2
    assert_array_almost_equal(pdp_interact_out.feature_grids[0],
                              np.array([0., 7.73284444, 7.8958, 8.6625, 13., 16.7, 26., 35.11111111, 73.5, 512.3292]),
                              decimal=8)
    assert_array_equal(pdp_interact_out.feature_grids[1], np.array([0, 1]))
    assert len(pdp_interact_out.pdp_isolate_outs) == 2
    expected = pd.DataFrame({'Fare': {0: 0.0, 6: 8.6625, 12: 26.0, 18: 512.3292},
                             'Sex': {0: 0.0, 6: 0.0, 12: 0.0, 18: 0.0},
                             'preds': {0: 0.6237624883651733, 6: 0.6005081534385681,
                                       12: 0.6391391158103943, 18: 0.7784096002578735}})
    assert_frame_equal(pdp_interact_out.pdp.iloc[[0, 6, 12, 18]], expected, check_like=True, check_dtype=False)

    # binary, onehot
    pdp_interact_out = pdp_interact(model=titanic_model, dataset=titanic_data, model_features=titanic_features,
                                    features=['Sex', ['Embarked_C', 'Embarked_S', 'Embarked_Q']])
    assert pdp_interact_out.features == ['Sex', ['Embarked_C', 'Embarked_S', 'Embarked_Q']]
    assert pdp_interact_out.feature_types == ['binary', 'onehot']
    assert_array_equal(pdp_interact_out.feature_grids[1], np.array(['Embarked_C', 'Embarked_S', 'Embarked_Q']))
    expected = pd.DataFrame({'Embarked_C': {0: 0, 3: 0, 5: 1},
                             'Embarked_Q': {0: 1, 3: 1, 5: 0},
                             'Embarked_S': {0: 0, 3: 0, 5: 0},
                             'Sex': {0: 0, 3: 1, 5: 1},
                             'preds': {0: 0.7331125140190125, 3: 0.21476328372955322, 5: 0.2710586190223694}})
    assert_frame_equal(pdp_interact_out.pdp.iloc[[0, 3, 5]], expected, check_like=True, check_dtype=False)


@pytest.mark.skip(reason="slow")
def test_pdp_interact_regression(ross_model, ross_data, ross_features):
    # binary, numeric
    pdp_interact_out = pdp_interact(model=ross_model, dataset=ross_data, model_features=ross_features,
                                    features=['SchoolHoliday', 'weekofyear'])
    assert pdp_interact_out.n_classes == 0
    assert pdp_interact_out.which_class is None
    assert pdp_interact_out.features == ['SchoolHoliday', 'weekofyear']
    assert pdp_interact_out.feature_types == ['binary', 'numeric']
    assert len(pdp_interact_out.feature_grids) == 2
    assert_array_equal(pdp_interact_out.feature_grids[0], np.array([0, 1]))
    assert_array_equal(pdp_interact_out.feature_grids[1], np.array([ 1.,  5., 10., 15., 20., 25., 30., 37., 45., 52.]))
    expected = pd.DataFrame({'SchoolHoliday': {0: 0.0, 6: 0.0, 12: 1.0, 18: 1.0},
                             'preds': {0: 6369.878633951306, 6: 5831.552135812868,
                                       12: 7311.965564610852, 18: 7129.481794228513},
                             'weekofyear': {0: 1.0, 6: 30.0, 12: 10.0, 18: 45.0}})
    assert_frame_equal(pdp_interact_out.pdp.iloc[[0, 6, 12, 18]], expected, check_like=True, check_dtype=False)

    # binary, onehot
    pdp_interact_out = pdp_interact(model=ross_model, dataset=ross_data, model_features=ross_features,
                                    features=['SchoolHoliday', ['StoreType_a', 'StoreType_b', 'StoreType_c', 'StoreType_d']])
    assert pdp_interact_out.features == ['SchoolHoliday', ['StoreType_a', 'StoreType_b', 'StoreType_c', 'StoreType_d']]
    assert pdp_interact_out.feature_types == ['binary', 'onehot']
    assert_array_equal(pdp_interact_out.feature_grids[1],
                       np.array(['StoreType_a', 'StoreType_b', 'StoreType_c', 'StoreType_d']))
    expected = pd.DataFrame({'SchoolHoliday': {0: 0, 3: 0, 5: 1, 7: 1}, 'StoreType_a': {0: 0, 3: 1, 5: 0, 7: 1},
                             'StoreType_b': {0: 0, 3: 0, 5: 0, 7: 0}, 'StoreType_c': {0: 0, 3: 0, 5: 1, 7: 0},
                             'StoreType_d': {0: 1, 3: 0, 5: 0, 7: 0},
                             'preds': {0: 6704.289795452508, 3: 7027.828665842933,
                                       5: 6929.464657747087, 7: 7077.063554254726}})
    assert_frame_equal(pdp_interact_out.pdp.iloc[[0, 3, 5, 7]], expected, check_like=True, check_dtype=False)


def test_pdp_isolate_multiclass(otto_model, otto_data, otto_features):
    pdp_interact_out = pdp_interact(model=otto_model, dataset=otto_data, model_features=otto_features,
                                    features=['feat_67', 'feat_24'], n_jobs=4)
    assert len(pdp_interact_out) == 9
    assert pdp_interact_out[0]._type == 'PDPInteract_instance'
    assert pdp_interact_out[0].n_classes == 9

    for i in range(9):
        assert pdp_interact_out[i].which_class == i

    assert pdp_interact_out[0].features == ['feat_67', 'feat_24']
    assert pdp_interact_out[0].feature_types == ['numeric', 'numeric']
    assert len(pdp_interact_out[0].feature_grids) == 2
    assert_array_equal(pdp_interact_out[0].feature_grids[0], np.array([0., 1., 2., 3., 4., 7., 104.]))
    assert_array_equal(pdp_interact_out[0].feature_grids[1], np.array([0., 1., 2., 4., 6., 263.]))

    pdp_expected_0 = pd.DataFrame({'feat_24': {0: 0.0, 10: 6.0, 20: 2.0, 30: 0.0, 40: 6.0},
                                   'feat_67': {0: 0.0, 10: 1.0, 20: 3.0, 30: 7.0, 40: 104.0},
                                   'preds': {0: 0.032884062186876346, 10: 0.03716684443581914,
                                             20: 0.03466417789843976, 30: 0.04137835741297658,
                                             40: 0.05534923559263137}})
    pdp_expected_4 = pd.DataFrame({'feat_24': {0: 0.0, 10: 6.0, 20: 2.0, 30: 0.0, 40: 6.0},
                                   'feat_67': {0: 0.0, 10: 1.0, 20: 3.0, 30: 7.0, 40: 104.0},
                                   'preds': {0: 0.04705840524904602, 10: 0.044781343934847326,
                                             20: 0.04458595946863714, 30: 0.04361372377905445,
                                             40: 0.02914331426354952}})
    pdp_expected_8 = pd.DataFrame({'feat_24': {0: 0.0, 10: 6.0, 20: 2.0, 30: 0.0, 40: 6.0},
                                   'feat_67': {0: 0.0, 10: 1.0, 20: 3.0, 30: 7.0, 40: 104.0},
                                   'preds': {0: 0.06781360095671897, 10: 0.07097061960631817,
                                             20: 0.08264019522285312, 30: 0.09372765764890978,
                                             40: 0.12120268916255889}})

    assert_frame_equal(pdp_interact_out[0].pdp.iloc[[0, 10, 20, 30, 40]], pdp_expected_0,
                       check_like=True, check_dtype=False)
    assert_frame_equal(pdp_interact_out[4].pdp.iloc[[0, 10, 20, 30, 40]], pdp_expected_4,
                       check_like=True, check_dtype=False)
    assert_frame_equal(pdp_interact_out[8].pdp.iloc[[0, 10, 20, 30, 40]], pdp_expected_8,
                       check_like=True, check_dtype=False)