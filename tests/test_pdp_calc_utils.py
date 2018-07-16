
import numpy as np
from numpy.testing import assert_array_equal
from pandas.testing import assert_frame_equal
import pandas as pd
from pdpbox.pdp_calc_utils import _calc_ice_lines, _calc_ice_lines_inter, _prepare_pdp_count_data
import pytest


class TestCalcICELinesBinary(object):
    def test_ice_binary(self, titanic_data, titanic_model, titanic_features):
        # binary feature
        grid_results, _data = _calc_ice_lines(
            feature_grid=0, data=titanic_data, model=titanic_model, model_features=titanic_features,
            n_classes=2, feature='Sex', feature_type='binary', predict_kwds={}, data_transformer=None, unit_test=True)
        expected = pd.DataFrame({0: {0: 0.527877688407898, 150: 0.8060303926467896, 300: 0.7201764583587646,
                                     450: 0.9118255376815796, 600: 0.8612496852874756, 750: 0.9331579208374023}})
        assert_frame_equal(grid_results.iloc[[0, 150, 300, 450, 600, 750]], expected, check_like=True,
                           check_dtype=False)
        assert_array_equal(_data['Sex'].unique(), np.array([0]))

    def test_ice_numeric(self, titanic_data, titanic_model, titanic_features):
        # numeric feature
        grid_results, _data = _calc_ice_lines(
            feature_grid=10, data=titanic_data, model=titanic_model, model_features=titanic_features,
            n_classes=2, feature='Fare', feature_type='numeric', predict_kwds={}, data_transformer=None, unit_test=True)
        expected = pd.DataFrame({10: {0: 0.10624270886182785, 150: 0.09951823949813843, 300: 0.6190056204795837,
                                      450: 0.16398519277572632, 600: 0.7467048764228821, 750: 0.868721067905426}})
        assert_frame_equal(grid_results.iloc[[0, 150, 300, 450, 600, 750]], expected, check_like=True,
                           check_dtype=False)
        assert_array_equal(_data['Fare'].unique(), np.array([10]))

    def test_ice_onehot(self, titanic_data, titanic_model, titanic_features):
        # onehot encoding feature
        grid_results, _data = _calc_ice_lines(
            feature_grid='Embarked_C', data=titanic_data, model=titanic_model, model_features=titanic_features,
            n_classes=2, feature=['Embarked_C', 'Embarked_S', 'Embarked_Q'], feature_type='onehot',
            predict_kwds={}, data_transformer=None, unit_test=True)
        expected = pd.DataFrame(
            {'Embarked_C': {0: 0.19760717451572418, 150: 0.11059149354696274, 300: 0.7139607667922974,
                            450: 0.2575017809867859, 600: 0.9045996069908142, 750: 0.9531968832015991}})
        assert_frame_equal(grid_results.iloc[[0, 150, 300, 450, 600, 750]], expected, check_like=True,
                           check_dtype=False)
        assert_array_equal(_data[['Embarked_C', 'Embarked_S', 'Embarked_Q']].mean().values, np.array([1, 0, 0]))

    def test_ice_predict_kwds(self, titanic_data, titanic_model, titanic_features):
        # with predict_kwds
        grid_results, _ = _calc_ice_lines(
            feature_grid=0, data=titanic_data, model=titanic_model, model_features=titanic_features, n_classes=2,
            feature='Sex', feature_type='binary', predict_kwds={'ntree_limit': 10}, data_transformer=None,
            unit_test=True)
        expected = pd.DataFrame({0: {0: 0.5039686560630798, 150: 0.6007370352745056, 300: 0.5556174516677856,
                                     450: 0.643494725227356, 600: 0.643494725227356, 750: 0.65798020362854}})
        assert_frame_equal(grid_results.iloc[[0, 150, 300, 450, 600, 750]], expected, check_like=True,
                           check_dtype=False)

    def test_ice_data_transformer(self, titanic_data, titanic_model, titanic_features):
        # with data_transformer
        def embark_change(df):
            df.loc[df['Embarked_C'] == 1, 'Fare'] = 10
            df.loc[df['Embarked_S'] == 1, 'Fare'] = 20
            df.loc[df['Embarked_Q'] == 1, 'Fare'] = 30
            return df

        grid_results, _data = _calc_ice_lines(
            feature_grid='Embarked_C', data=titanic_data, model=titanic_model, model_features=titanic_features,
            n_classes=2, feature=['Embarked_C', 'Embarked_S', 'Embarked_Q'], feature_type='onehot',
            predict_kwds={}, data_transformer=embark_change, unit_test=True)
        expected = pd.DataFrame(
            {'Embarked_C': {0: 0.20869030058383942, 150: 0.10480280220508575, 300: 0.6179739832878113,
                            450: 0.18637187778949738, 600: 0.8106594085693359, 750: 0.8973860740661621}})
        assert_frame_equal(grid_results.iloc[[0, 150, 300, 450, 600, 750]], expected, check_like=True,
                           check_dtype=False)
        assert_array_equal(_data['Fare'].unique(), np.array([10.]))


def test_calc_ice_lines_regression(ross_data, ross_model, ross_features):
    grid_results, _data = _calc_ice_lines(
        feature_grid=1, data=ross_data, model=ross_model, model_features=ross_features, n_classes=0,
        feature='SchoolHoliday', feature_type='binary', predict_kwds={}, data_transformer=None, unit_test=True)
    assert_array_equal(_data['SchoolHoliday'].unique(), np.array([1]))
    expected = pd.DataFrame({1: {0: 8802.910080560769, 100000: 8293.287914628107, 200000: 5352.321273982288,
                                 300000: 5412.1717528683475, 400000: 7933.070072150073, 500000: 7520.956055932758,
                                 600000: 5493.134809064146, 700000: 5528.43699339258, 800000: 4877.434213535265}})
    assert_frame_equal(grid_results.iloc[[0, 100000, 200000, 300000, 400000, 500000, 600000, 700000, 800000]],
                       expected, check_like=True, check_dtype=False)


def test_calc_ice_lines_multiclass(otto_data, otto_model, otto_features):
    grid_results, _data = _calc_ice_lines(
        feature_grid=1, data=otto_data, model=otto_model, model_features=otto_features, n_classes=9,
        feature='feat_67', feature_type='numeric', predict_kwds={}, data_transformer=None, unit_test=True)
    assert len(grid_results) == 9
    assert_array_equal(_data['feat_67'].unique(), np.array([1]))

    expected_target_0 = pd.DataFrame({1: {0: 0.56, 10000: 0.0, 20000: 0.0, 30000: 0.01, 40000: 0.01,
                                          50000: 0.03, 60000: 0.04}})
    assert_frame_equal(grid_results[0].iloc[[0, 10000, 20000, 30000, 40000, 50000, 60000]],
                       expected_target_0, check_like=True, check_dtype=False)

    expected_target_3 = pd.DataFrame({1: {0: 0.0, 10000: 0.04, 20000: 0.03, 30000: 0.02, 40000: 0.0,
                                          50000: 0.0, 60000: 0.0}})
    assert_frame_equal(grid_results[3].iloc[[0, 10000, 20000, 30000, 40000, 50000, 60000]],
                       expected_target_3, check_like=True, check_dtype=False)

    expected_target_7 = pd.DataFrame({1: {0: 0.0, 10000: 0.03, 20000: 0.01, 30000: 0.02, 40000: 0.03,
                                          50000: 0.9, 60000: 0.02}})
    assert_frame_equal(grid_results[7].iloc[[0, 10000, 20000, 30000, 40000, 50000, 60000]],
                       expected_target_7, check_like=True, check_dtype=False)


class TestCalcICELinesInterBinary(object):
    def test_ice_inter_binary_numeric(self, titanic_data, titanic_model, titanic_features):
        # binary and numeric
        grid_results, _data = _calc_ice_lines_inter(
            feature_grids_combo=[0, 10], data=titanic_data, model=titanic_model, model_features=titanic_features,
            n_classes=2, feature_list=['Sex', 'Fare'], predict_kwds={}, data_transformer=None, unit_test=True)
        assert_array_equal(np.unique(_data[['Sex', 'Fare']].values, axis=0), np.array([[0, 10]]))
        expected = pd.DataFrame({'Fare': {0: 10, 150: 10, 300: 10, 450: 10, 600: 10, 750: 10},
                                 'Sex': {0: 0, 150: 0, 300: 0, 450: 0, 600: 0, 750: 0},
                                 'preds': {0: 0.37542039155960083, 150: 0.7539840340614319, 300: 0.6190056204795837,
                                           450: 0.8457906246185303, 600: 0.7467048764228821, 750: 0.868721067905426}})
        assert_frame_equal(grid_results.iloc[[0, 150, 300, 450, 600, 750]], expected, check_like=True,
                           check_dtype=False)

    def test_ice_inter_binary_onehot(self, titanic_data, titanic_model, titanic_features):
        # binary and onehot
        grid_results, _data = _calc_ice_lines_inter(
            feature_grids_combo=[1, 0, 1, 0], data=titanic_data, model=titanic_model, model_features=titanic_features,
            n_classes=2, feature_list=['Sex', 'Embarked_C', 'Embarked_S', 'Embarked_Q'], predict_kwds={},
            data_transformer=None, unit_test=True)
        assert_array_equal(np.unique(_data[['Sex', 'Embarked_C', 'Embarked_S', 'Embarked_Q']].values, axis=0),
                           np.array([[1, 0, 1, 0]]))
        expected = pd.DataFrame({'Embarked_C': {0: 0, 150: 0, 300: 0, 450: 0, 600: 0, 750: 0},
                                 'Embarked_Q': {0: 0, 150: 0, 300: 0, 450: 0, 600: 0, 750: 0},
                                 'Embarked_S': {0: 1, 150: 1, 300: 1, 450: 1, 600: 1, 750: 1},
                                 'Sex': {0: 1, 150: 1, 300: 1, 450: 1, 600: 1, 750: 1},
                                 'preds': {0: 0.1041787713766098, 150: 0.09937939792871475, 300: 0.09760041534900665,
                                           450: 0.22834216058254242, 600: 0.14207805693149567,
                                           750: 0.8904628753662109}})
        assert_frame_equal(grid_results.iloc[[0, 150, 300, 450, 600, 750]], expected, check_like=True,
                           check_dtype=False)

    def test_ice_inter_onehot_numeric(self, titanic_data, titanic_model, titanic_features):
        # onehot and numeric
        grid_results, _data = _calc_ice_lines_inter(
            feature_grids_combo=[0, 0, 1, 10], data=titanic_data, model=titanic_model, model_features=titanic_features,
            n_classes=2, feature_list=['Embarked_C', 'Embarked_S', 'Embarked_Q', 'Fare'], predict_kwds={},
            data_transformer=None, unit_test=True)
        assert_array_equal(np.unique(_data[['Embarked_C', 'Embarked_S', 'Embarked_Q', 'Fare']].values, axis=0),
                           np.array([[0, 0, 1, 10]]))
        expected = pd.DataFrame({'Embarked_C': {0: 0, 150: 0, 300: 0, 450: 0, 600: 0, 750: 0},
                                 'Embarked_Q': {0: 1, 150: 1, 300: 1, 450: 1, 600: 1, 750: 1},
                                 'Embarked_S': {0: 0, 150: 0, 300: 0, 450: 0, 600: 0, 750: 0},
                                 'Fare': {0: 10, 150: 10, 300: 10, 450: 10, 600: 10, 750: 10},
                                 'preds': {0: 0.14132696390151978, 150: 0.09227359294891357, 300: 0.6190056204795837,
                                           450: 0.13888351619243622, 600: 0.7998642325401306, 750: 0.8927221894264221}})
        assert_frame_equal(grid_results.iloc[[0, 150, 300, 450, 600, 750]], expected, check_like=True,
                           check_dtype=False)

    def test_ice_inter_predict_kwds(self, titanic_data, titanic_model, titanic_features):
        # with predict_kwds
        grid_results, _data = _calc_ice_lines_inter(
            feature_grids_combo=[0, 10], data=titanic_data, model=titanic_model, model_features=titanic_features,
            n_classes=2,
            feature_list=['Sex', 'Fare'], predict_kwds={'ntree_limit': 10}, data_transformer=None, unit_test=True)
        expected = pd.DataFrame({'Fare': {0: 10, 150: 10, 300: 10, 450: 10, 600: 10, 750: 10},
                                 'Sex': {0: 0, 150: 0, 300: 0, 450: 0, 600: 0, 750: 0},
                                 'preds': {0: 0.48284032940864563, 150: 0.5588331818580627, 300: 0.5346577763557434,
                                           450: 0.5927444100379944, 600: 0.5927444100379944, 750: 0.6094712615013123}})
        assert_frame_equal(grid_results.iloc[[0, 150, 300, 450, 600, 750]], expected, check_like=True,
                           check_dtype=False)

    def test_ice_inter_data_transformer(self, titanic_data, titanic_model, titanic_features):
        # with data_transformer
        def embark_change(df):
            df.loc[df['Embarked_C'] == 1, 'Fare'] = 10
            df.loc[df['Embarked_S'] == 1, 'Fare'] = 20
            df.loc[df['Embarked_Q'] == 1, 'Fare'] = 30
            return df

        grid_results, _data = _calc_ice_lines_inter(
            feature_grids_combo=[1, 0, 1, 0], data=titanic_data, model=titanic_model, model_features=titanic_features,
            n_classes=2, feature_list=['Sex', 'Embarked_C', 'Embarked_S', 'Embarked_Q'], predict_kwds={},
            data_transformer=embark_change, unit_test=True)
        assert_array_equal(_data['Fare'].unique(), np.array([20]))
        expected = pd.DataFrame({'Embarked_C': {0: 0, 150: 0, 300: 0, 450: 0, 600: 0, 750: 0},
                                 'Embarked_Q': {0: 0, 150: 0, 300: 0, 450: 0, 600: 0, 750: 0},
                                 'Embarked_S': {0: 1, 150: 1, 300: 1, 450: 1, 600: 1, 750: 1},
                                 'Sex': {0: 1, 150: 1, 300: 1, 450: 1, 600: 1, 750: 1},
                                 'preds': {0: 0.11097385734319687, 150: 0.10733838379383087, 300: 0.12938366830348969,
                                           450: 0.18971671164035797, 600: 0.11736717820167542,
                                           750: 0.9063724279403687}})
        assert_frame_equal(grid_results.iloc[[0, 150, 300, 450, 600, 750]], expected, check_like=True,
                           check_dtype=False)


def test_calc_ice_lines_inter_regression(ross_data, ross_model, ross_features):
    grid_results, _data = _calc_ice_lines_inter(
        feature_grids_combo=[1, 1, 0, 0, 0], data=ross_data, model=ross_model, model_features=ross_features,
        n_classes=0, feature_list=['SchoolHoliday', 'StoreType_a', 'StoreType_b', 'StoreType_c', 'StoreType_d'],
        predict_kwds={}, data_transformer=None, unit_test=True)
    assert_array_equal(np.unique(_data[['SchoolHoliday', 'StoreType_a', 'StoreType_b', 'StoreType_c',
                                        'StoreType_d']].values, axis=0), np.array([[1, 1, 0, 0, 0]]))

    expected = pd.DataFrame({'SchoolHoliday': {0: 1, 100000: 1, 200000: 1, 300000: 1, 400000: 1,
                                               500000: 1, 600000: 1, 700000: 1, 800000: 1},
                             'StoreType_a': {0: 1, 100000: 1, 200000: 1, 300000: 1, 400000: 1,
                                             500000: 1, 600000: 1, 700000: 1, 800000: 1},
                             'StoreType_b': {0: 0, 100000: 0, 200000: 0, 300000: 0, 400000: 0,
                                             500000: 0, 600000: 0, 700000: 0, 800000: 0},
                             'StoreType_c': {0: 0, 100000: 0, 200000: 0, 300000: 0, 400000: 0,
                                             500000: 0, 600000: 0, 700000: 0, 800000: 0},
                             'StoreType_d': {0: 0, 100000: 0, 200000: 0, 300000: 0, 400000: 0,
                                             500000: 0, 600000: 0, 700000: 0, 800000: 0},
                             'preds': {0: 8252.710201150694, 100000: 8293.287914628107, 200000: 5352.321273982288,
                                       300000: 5707.464798080147, 400000: 5307.592789748846, 500000: 7520.956055932758,
                                       600000: 5493.134809064146, 700000: 7040.829184296985, 800000: 4877.434213535266}})

    assert_frame_equal(grid_results.iloc[[0, 100000, 200000, 300000, 400000, 500000, 600000, 700000, 800000]],
                       expected, check_like=True, check_dtype=False)


# @pytest.mark.skip(reason="slow")
def test_calc_ice_lines_inter_multiclass(otto_data, otto_model, otto_features):
    grid_results, _data = _calc_ice_lines_inter(
        feature_grids_combo=[1, 1], data=otto_data, model=otto_model, model_features=otto_features,
        n_classes=9, feature_list=['feat_67', 'feat_32'], predict_kwds={}, data_transformer=None, unit_test=True)
    assert_array_equal(np.unique(_data[['feat_67', 'feat_32']].values, axis=0), np.array([[1, 1]]))

    expected = pd.DataFrame({'class_0_preds': {0: 0.55, 10000: 0.01, 20000: 0.0, 30000: 0.01,
                                               40000: 0.01, 50000: 0.03, 60000: 0.04},
                             'class_1_preds': {0: 0.0, 10000: 0.72, 20000: 0.19, 30000: 0.04,
                                               40000: 0.0, 50000: 0.0, 60000: 0.04},
                             'class_2_preds': {0: 0.0, 10000: 0.12, 20000: 0.7, 30000: 0.01,
                                               40000: 0.0, 50000: 0.0, 60000: 0.02},
                             'class_3_preds': {0: 0.0, 10000: 0.05, 20000: 0.03, 30000: 0.02,
                                               40000: 0.0, 50000: 0.0, 60000: 0.0},
                             'class_4_preds': {0: 0.0, 10000: 0.01, 20000: 0.0, 30000: 0.84,
                                               40000: 0.0, 50000: 0.0, 60000: 0.0},
                             'class_5_preds': {0: 0.44, 10000: 0.03, 20000: 0.04, 30000: 0.02,
                                               40000: 0.88, 50000: 0.0, 60000: 0.02},
                             'class_6_preds': {0: 0.01, 10000: 0.0, 20000: 0.03, 30000: 0.03,
                                               40000: 0.01, 50000: 0.07, 60000: 0.01},
                             'class_7_preds': {0: 0.0, 10000: 0.02, 20000: 0.01, 30000: 0.02,
                                               40000: 0.03, 50000: 0.9, 60000: 0.02},
                             'class_8_preds': {0: 0.0, 10000: 0.04, 20000: 0.0, 30000: 0.01,
                                               40000: 0.07, 50000: 0.0, 60000: 0.85},
                             'feat_32': {0: 1, 10000: 1, 20000: 1, 30000: 1, 40000: 1, 50000: 1, 60000: 1},
                             'feat_67': {0: 1, 10000: 1, 20000: 1, 30000: 1, 40000: 1, 50000: 1, 60000: 1}})

    assert_frame_equal(grid_results.iloc[[0, 10000, 20000, 30000, 40000, 50000, 60000]], expected,
                       check_like=True, check_dtype=False)


class TestPreparePDPCountData(object):
    def test_count_data_binary(self, titanic_data):
        # binary feature
        count_data = _prepare_pdp_count_data(feature='Sex', feature_type='binary', data=titanic_data,
                                             feature_grids=[0, 1])
        expected = pd.DataFrame({'count': {0: 314, 1: 577}, 'x': {0: 0, 1: 1},
                                 'count_norm': {0: 0.35241301907968575, 1: 0.6475869809203143}})
        assert_frame_equal(count_data, expected, check_like=True, check_dtype=False)

    def test_count_data_onehot(self, titanic_data):
        # onehot feature
        count_data = _prepare_pdp_count_data(feature=['Embarked_C', 'Embarked_S', 'Embarked_Q'], feature_type='onehot',
                                             data=titanic_data,
                                             feature_grids=['Embarked_C', 'Embarked_S', 'Embarked_Q'])
        expected = pd.DataFrame({'count': {0: 168, 1: 646, 2: 77},
                                 'count_norm': {0: 0.18855218855218855, 1: 0.7250280583613917, 2: 0.08641975308641975},
                                 'index': {0: 'Embarked_C', 1: 'Embarked_S', 2: 'Embarked_Q'}, 'x': {0: 0, 1: 1, 2: 2}})
        assert_frame_equal(count_data, expected, check_like=True, check_dtype=False)

    def test_count_data_numeric(self, titanic_data):
        # numeric feature
        count_data = _prepare_pdp_count_data(
            feature='Fare', feature_type='numeric', data=titanic_data,
            feature_grids=np.array([0., 7.73284444, 7.8958, 8.6625, 13., 16.7, 26., 35.11111111, 73.5, 512.3292]))
        expected = pd.DataFrame({'count': {0: 99, 1: 86, 2: 110, 3: 91, 4: 108, 5: 71, 6: 128, 7: 96, 8: 102},
                                 'count_norm': {0: 0.1111111111111111, 1: 0.09652076318742986, 2: 0.12345679012345678,
                                                3: 0.10213243546576879, 4: 0.12121212121212122, 5: 0.07968574635241302,
                                                6: 0.143658810325477, 7: 0.10774410774410774, 8: 0.11447811447811448},
                                 'x': {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8},
                                 'xticklabels': {0: '[0, 7.73)', 1: '[7.73, 7.9)', 2: '[7.9, 8.66)', 3: '[8.66, 13)',
                                                 4: '[13, 16.7)', 5: '[16.7, 26)', 6: '[26, 35.11)', 7: '[35.11, 73.5)',
                                                 8: '[73.5, 512.33]'}})
        assert_frame_equal(count_data, expected, check_like=True, check_dtype=False)

    def test_count_data_numeric_outlier(self, titanic_data):
        # numeric feature with outlier values
        count_data = _prepare_pdp_count_data(
            feature='Fare', feature_type='numeric', data=titanic_data,
            feature_grids=np.array([7.225, 7.75, 7.9104, 9., 13., 16.1, 26., 31., 56.4958, 112.07915]))
        expected = pd.DataFrame(
            {'count': {0: 43, 1: 63, 2: 117, 3: 88, 4: 75, 5: 99, 6: 80, 7: 101, 8: 89, 9: 91, 10: 45},
             'count_norm': {0: 0.04826038159371493, 1: 0.0707070707070707, 2: 0.13131313131313133,
                            3: 0.09876543209876543, 4: 0.08417508417508418, 5: 0.1111111111111111,
                            6: 0.08978675645342311, 7: 0.11335578002244669, 8: 0.09988776655443322,
                            9: 0.10213243546576879, 10: 0.050505050505050504},
             'x': {0: -1, 1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 9},
             'xticklabels': {0: '[0, 7.22)', 1: '[7.22, 7.75)', 2: '[7.75, 7.91)', 3: '[7.91, 9)',
                             4: '[9, 13)', 5: '[13, 16.1)', 6: '[16.1, 26)', 7: '[26, 31)',
                             8: '[31, 56.5)', 9: '[56.5, 112.08)', 10: '[112.08, 512.33]'}})
        assert_frame_equal(count_data, expected, check_like=True, check_dtype=False)







