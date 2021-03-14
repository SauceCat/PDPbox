import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
from pandas.testing import assert_frame_equal
import pandas as pd
import matplotlib

from pdpbox.pdp import pdp_isolate, pdp_plot


class TestPDPIsolateBinary(object):
    def test_pdp_isolate_binary_feature(
        self, titanic_model, titanic_data, titanic_features
    ):
        # feature_type: binary
        pdp_isolate_out = pdp_isolate(
            model=titanic_model,
            dataset=titanic_data,
            model_features=titanic_features,
            feature="Sex",
            num_grid_points=10,
            grid_type="percentile",
            percentile_range=None,
            grid_range=None,
            cust_grid_points=None,
            memory_limit=0.5,
            n_jobs=1,
            predict_kwds={},
            data_transformer=None,
        )

        assert pdp_isolate_out._type == "PDPIsolate_instance"
        assert pdp_isolate_out.n_classes == 2
        assert pdp_isolate_out.which_class is None
        assert pdp_isolate_out.feature == "Sex"
        assert pdp_isolate_out.feature_type == "binary"
        assert pdp_isolate_out.percentile_info == []
        assert pdp_isolate_out.display_columns == ["Sex_0", "Sex_1"]
        assert pdp_isolate_out.hist_data is None

    def test_pdp_isolate_onehot_feature(
        self, titanic_model, titanic_data, titanic_features
    ):
        # feature_type: onehot
        pdp_isolate_out = pdp_isolate(
            model=titanic_model,
            dataset=titanic_data,
            model_features=titanic_features,
            feature=["Embarked_C", "Embarked_S", "Embarked_Q"],
            num_grid_points=10,
            grid_type="percentile",
            percentile_range=None,
            grid_range=None,
            cust_grid_points=None,
            memory_limit=0.5,
            n_jobs=1,
            predict_kwds={},
            data_transformer=None,
        )

        assert pdp_isolate_out._type == "PDPIsolate_instance"
        assert pdp_isolate_out.n_classes == 2
        assert pdp_isolate_out.which_class is None
        assert pdp_isolate_out.feature == ["Embarked_C", "Embarked_S", "Embarked_Q"]
        assert pdp_isolate_out.feature_type == "onehot"
        assert pdp_isolate_out.percentile_info == []
        assert pdp_isolate_out.display_columns == [
            "Embarked_C",
            "Embarked_S",
            "Embarked_Q",
        ]
        assert pdp_isolate_out.hist_data is None

    def test_pdp_isolate_numeric_feature(
        self, titanic_model, titanic_data, titanic_features
    ):
        # feature_type: numeric
        pdp_isolate_out = pdp_isolate(
            model=titanic_model,
            dataset=titanic_data,
            model_features=titanic_features,
            feature="Fare",
            num_grid_points=10,
            grid_type="percentile",
            percentile_range=None,
            grid_range=None,
            cust_grid_points=None,
            memory_limit=0.5,
            n_jobs=1,
            predict_kwds={},
            data_transformer=None,
        )

        assert pdp_isolate_out._type == "PDPIsolate_instance"
        assert pdp_isolate_out.n_classes == 2
        assert pdp_isolate_out.which_class is None
        assert pdp_isolate_out.feature == "Fare"
        assert pdp_isolate_out.feature_type == "numeric"
        assert len(pdp_isolate_out.hist_data) == titanic_data.shape[0]

    def test_pdp_isolate_cust_grid_points(
        self, titanic_model, titanic_data, titanic_features
    ):
        # use cust_grid_points
        pdp_isolate_out = pdp_isolate(
            model=titanic_model,
            dataset=titanic_data,
            model_features=titanic_features,
            feature="Fare",
            num_grid_points=10,
            grid_type="percentile",
            percentile_range=None,
            grid_range=None,
            cust_grid_points=range(0, 100, 5),
            memory_limit=0.5,
            n_jobs=1,
            predict_kwds={},
            data_transformer=None,
        )

        assert pdp_isolate_out._type == "PDPIsolate_instance"
        assert pdp_isolate_out.n_classes == 2
        assert pdp_isolate_out.which_class is None
        assert pdp_isolate_out.feature == "Fare"
        assert pdp_isolate_out.feature_type == "numeric"
        assert pdp_isolate_out.percentile_info == []
        assert pdp_isolate_out.display_columns == [
            "0",
            "5",
            "10",
            "15",
            "20",
            "25",
            "30",
            "35",
            "40",
            "45",
            "50",
            "55",
            "60",
            "65",
            "70",
            "75",
            "80",
            "85",
            "90",
            "95",
        ]
        assert len(pdp_isolate_out.hist_data) == titanic_data.shape[0]


class TestPDPIsolateRegression(object):
    def test_pdp_isolate_regression(self, ross_model, ross_data, ross_features):
        pdp_isolate_out = pdp_isolate(
            model=ross_model,
            dataset=ross_data,
            model_features=ross_features,
            feature="SchoolHoliday",
            num_grid_points=10,
            grid_type="percentile",
            percentile_range=None,
            grid_range=None,
            cust_grid_points=None,
            memory_limit=0.5,
            n_jobs=1,
            predict_kwds={},
            data_transformer=None,
        )

        assert pdp_isolate_out._type == "PDPIsolate_instance"
        assert pdp_isolate_out.n_classes == 0
        assert pdp_isolate_out.which_class is None
        assert pdp_isolate_out.feature == "SchoolHoliday"
        assert pdp_isolate_out.feature_type == "binary"
        assert pdp_isolate_out.percentile_info == []
        assert pdp_isolate_out.display_columns == ["SchoolHoliday_0", "SchoolHoliday_1"]
        assert pdp_isolate_out.hist_data is None

    def test_pdp_isolate_n_jobs(self, ross_model, ross_data, ross_features):
        # test n_jobs > 1
        _ = pdp_isolate(
            model=ross_model,
            dataset=ross_data,
            model_features=ross_features,
            feature="SchoolHoliday",
            num_grid_points=10,
            grid_type="percentile",
            percentile_range=None,
            grid_range=None,
            cust_grid_points=None,
            memory_limit=0.5,
            n_jobs=2,
            predict_kwds={},
            data_transformer=None,
        )


def test_pdp_isolate_multiclass(otto_model, otto_data, otto_features):
    pdp_isolate_out = pdp_isolate(
        model=otto_model,
        dataset=otto_data,
        model_features=otto_features,
        feature="feat_67",
        num_grid_points=10,
        grid_type="percentile",
        percentile_range=None,
        grid_range=None,
        cust_grid_points=None,
        memory_limit=0.5,
        n_jobs=1,
        predict_kwds={},
        data_transformer=None,
    )

    assert len(pdp_isolate_out) == 9
    assert pdp_isolate_out[0]._type == "PDPIsolate_instance"
    assert pdp_isolate_out[0].n_classes == 9

    for i in range(9):
        assert pdp_isolate_out[i].which_class == i

    assert pdp_isolate_out[0].feature == "feat_67"
    assert pdp_isolate_out[0].feature_type == "numeric"


class TestPDPPlotSingle(object):
    @pytest.fixture
    def pdp_sex(self, titanic_data, titanic_model, titanic_features):
        result = pdp_isolate(
            model=titanic_model,
            dataset=titanic_data,
            model_features=titanic_features,
            feature="Sex",
        )
        return result

    def test_pdp_plot_single_default(self, pdp_sex):
        # single chart without data dist plot
        fig, axes = pdp_plot(pdp_sex, "sex")
        assert type(fig) == matplotlib.figure.Figure
        assert sorted(axes.keys()) == ["pdp_ax", "title_ax"]
        assert type(axes["pdp_ax"]) == matplotlib.axes._subplots.Subplot
        assert type(axes["title_ax"]) == matplotlib.axes._subplots.Subplot

    def test_pdp_plot_single_distplot(self, pdp_sex):
        # single chart with data dist plot
        fig, axes = pdp_plot(pdp_sex, "sex", plot_pts_dist=True)
        assert sorted(axes.keys()) == ["pdp_ax", "title_ax"]
        assert sorted(axes["pdp_ax"].keys()) == ["_count_ax", "_pdp_ax"]
        assert type(axes["pdp_ax"]["_pdp_ax"]) == matplotlib.axes._subplots.Subplot
        assert type(axes["pdp_ax"]["_count_ax"]) == matplotlib.axes._subplots.Subplot
        assert type(axes["title_ax"]) == matplotlib.axes._subplots.Subplot


class TestPDPPlotMulti(object):
    @pytest.fixture
    def pdp_feat_67_rf(self, otto_data, otto_model, otto_features):
        result = pdp_isolate(
            model=otto_model,
            dataset=otto_data,
            model_features=otto_features,
            feature="feat_67",
        )
        return result

    def test_pdp_plot_multi_default(self, pdp_feat_67_rf):
        # multi charts without data dist plot
        fig, axes = pdp_plot(
            pdp_isolate_out=pdp_feat_67_rf,
            feature_name="feat_67",
            center=True,
            x_quantile=True,
        )
        assert type(fig) == matplotlib.figure.Figure
        assert sorted(axes.keys()) == ["pdp_ax", "title_ax"]
        assert len(axes["pdp_ax"]) == 9
        assert type(axes["title_ax"]) == matplotlib.axes._subplots.Subplot
        assert type(axes["pdp_ax"][0]) == matplotlib.axes._subplots.Subplot

    def test_pdp_plot_multi_which_classes(self, pdp_feat_67_rf):
        # change which classes
        fig, axes = pdp_plot(
            pdp_feat_67_rf,
            "feat_67",
            center=True,
            x_quantile=True,
            ncols=2,
            which_classes=[0, 3, 7],
        )
        assert len(axes["pdp_ax"]) == 3

    def test_pdp_plot_multi_one_class(self, pdp_feat_67_rf):
        # only keep 1 class
        fig, axes = pdp_plot(
            pdp_feat_67_rf,
            "feat_67",
            center=True,
            x_quantile=True,
            ncols=2,
            which_classes=[5],
        )
        assert type(axes["pdp_ax"]) == matplotlib.axes._subplots.Subplot

    def test_pdp_plot_multi_distplot(self, pdp_feat_67_rf):
        # multi charts with data dist plot
        fig, axes = pdp_plot(
            pdp_isolate_out=pdp_feat_67_rf,
            feature_name="feat_67",
            center=True,
            x_quantile=True,
            plot_pts_dist=True,
        )
        assert sorted(axes.keys()) == ["pdp_ax", "title_ax"]
        assert len(axes["pdp_ax"]) == 9
        assert sorted(axes["pdp_ax"][0].keys()) == ["_count_ax", "_pdp_ax"]
        assert type(axes["pdp_ax"][0]["_count_ax"]) == matplotlib.axes._subplots.Subplot
        assert type(axes["pdp_ax"][0]["_pdp_ax"]) == matplotlib.axes._subplots.Subplot
