import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
from pandas.testing import assert_frame_equal
import pandas as pd
import matplotlib

from pdpbox.pdp import pdp_interact, pdp_interact_plot


class TestPDPInteractBinary(object):
    def test_binary_numeric(self, titanic_model, titanic_data, titanic_features):
        pdp_interact_out = pdp_interact(
            model=titanic_model,
            dataset=titanic_data,
            model_features=titanic_features,
            features=["Fare", "Sex"],
        )
        assert pdp_interact_out._type == "PDPInteract_instance"
        assert pdp_interact_out.n_classes == 2
        assert pdp_interact_out.which_class is None
        assert pdp_interact_out.features == ["Fare", "Sex"]
        assert pdp_interact_out.feature_types == ["numeric", "binary"]
        assert len(pdp_interact_out.feature_grids) == 2
        assert len(pdp_interact_out.pdp_isolate_outs) == 2

    def test_binary_onehot(self, titanic_model, titanic_data, titanic_features):
        pdp_interact_out = pdp_interact(
            model=titanic_model,
            dataset=titanic_data,
            model_features=titanic_features,
            features=["Sex", ["Embarked_C", "Embarked_S", "Embarked_Q"]],
        )
        assert pdp_interact_out._type == "PDPInteract_instance"
        assert pdp_interact_out.n_classes == 2
        assert pdp_interact_out.which_class is None
        assert pdp_interact_out.features == [
            "Sex",
            ["Embarked_C", "Embarked_S", "Embarked_Q"],
        ]
        assert pdp_interact_out.feature_types == ["binary", "onehot"]
        assert len(pdp_interact_out.feature_grids) == 2
        assert len(pdp_interact_out.pdp_isolate_outs) == 2


class TestPDPInteractRegression(object):
    def test_binary_numeric(self, ross_model, ross_data, ross_features):
        pdp_interact_out = pdp_interact(
            model=ross_model,
            dataset=ross_data,
            model_features=ross_features,
            features=["SchoolHoliday", "weekofyear"],
        )
        assert pdp_interact_out._type == "PDPInteract_instance"
        assert pdp_interact_out.n_classes == 0
        assert pdp_interact_out.which_class is None
        assert pdp_interact_out.features == ["SchoolHoliday", "weekofyear"]
        assert pdp_interact_out.feature_types == ["binary", "numeric"]
        assert len(pdp_interact_out.feature_grids) == 2
        assert len(pdp_interact_out.feature_grids) == 2

    def test_binary_onehot(self, ross_model, ross_data, ross_features):
        pdp_interact_out = pdp_interact(
            model=ross_model,
            dataset=ross_data,
            model_features=ross_features,
            features=[
                "SchoolHoliday",
                ["StoreType_a", "StoreType_b", "StoreType_c", "StoreType_d"],
            ],
        )
        assert pdp_interact_out._type == "PDPInteract_instance"
        assert pdp_interact_out.n_classes == 0
        assert pdp_interact_out.which_class is None
        assert pdp_interact_out.features == [
            "SchoolHoliday",
            ["StoreType_a", "StoreType_b", "StoreType_c", "StoreType_d"],
        ]
        assert pdp_interact_out.feature_types == ["binary", "onehot"]
        assert len(pdp_interact_out.feature_grids) == 2
        assert len(pdp_interact_out.feature_grids) == 2


class TestPDPInteractMulticlass(object):
    def test_numeric_numeric(self, otto_model, otto_data, otto_features):
        pdp_interact_out = pdp_interact(
            model=otto_model,
            dataset=otto_data,
            model_features=otto_features,
            features=["feat_67", "feat_25"],
        )
        assert len(pdp_interact_out) == 9
        assert pdp_interact_out[0]._type == "PDPInteract_instance"
        assert pdp_interact_out[0].n_classes == 9

        for i in range(9):
            assert pdp_interact_out[i].which_class == i

        assert pdp_interact_out[0].features == ["feat_67", "feat_25"]
        assert pdp_interact_out[0].feature_types == ["numeric", "numeric"]
        assert len(pdp_interact_out[0].feature_grids) == 2


class TestPDPInteractSingle(object):
    @pytest.fixture
    def pdp_interact_out(self, titanic_data, titanic_model, titanic_features):
        result = pdp_interact(
            model=titanic_model,
            dataset=titanic_data,
            model_features=titanic_features,
            features=["Age", "Fare"],
        )
        return result

    def test_contour(self, pdp_interact_out):
        fig, axes = pdp_interact_plot(
            pdp_interact_out=pdp_interact_out,
            feature_names=["age", "fare"],
            plot_type="contour",
            x_quantile=True,
        )
        assert type(fig) == matplotlib.figure.Figure
        assert sorted(axes.keys()) == ["pdp_inter_ax", "title_ax"]
        for k in axes.keys():
            assert type(axes[k]) == matplotlib.axes._subplots.Subplot

    def test_contour_pdp(self, pdp_interact_out):
        fig, axes = pdp_interact_plot(
            pdp_interact_out=pdp_interact_out,
            feature_names=["age", "fare"],
            plot_type="contour",
            x_quantile=True,
            plot_pdp=True,
        )
        assert type(fig) == matplotlib.figure.Figure
        assert sorted(axes.keys()) == ["pdp_inter_ax", "title_ax"]
        assert sorted(axes["pdp_inter_ax"].keys()) == [
            "_pdp_inter_ax",
            "_pdp_x_ax",
            "_pdp_y_ax",
        ]
        for k in axes["pdp_inter_ax"].keys():
            assert type(axes["pdp_inter_ax"][k]) == matplotlib.axes._subplots.Subplot

    def test_grid(self, pdp_interact_out):
        fig, axes = pdp_interact_plot(
            pdp_interact_out=pdp_interact_out,
            feature_names=["age", "fare"],
            plot_type="grid",
            x_quantile=True,
        )
        assert type(fig) == matplotlib.figure.Figure
        assert sorted(axes.keys()) == ["pdp_inter_ax", "title_ax"]
        for k in axes.keys():
            assert type(axes[k]) == matplotlib.axes._subplots.Subplot

    def test_grid_pdp(self, pdp_interact_out):
        fig, axes = pdp_interact_plot(
            pdp_interact_out=pdp_interact_out,
            feature_names=["age", "fare"],
            plot_type="grid",
            x_quantile=True,
            plot_pdp=True,
        )
        assert type(fig) == matplotlib.figure.Figure
        assert sorted(axes.keys()) == ["pdp_inter_ax", "title_ax"]
        assert sorted(axes["pdp_inter_ax"].keys()) == [
            "_pdp_inter_ax",
            "_pdp_x_ax",
            "_pdp_y_ax",
        ]
        for k in axes["pdp_inter_ax"].keys():
            assert type(axes["pdp_inter_ax"][k]) == matplotlib.axes._subplots.Subplot


class TestPDPInteractMulti(object):
    @pytest.fixture
    def pdp_interact_out(self, otto_data, otto_model, otto_features):
        result = pdp_interact(
            model=otto_model,
            dataset=otto_data,
            model_features=otto_features,
            features=["feat_67", "feat_24"],
        )
        return result

    def test_contour(self, pdp_interact_out):
        fig, axes = pdp_interact_plot(
            pdp_interact_out=pdp_interact_out,
            feature_names=["feat_67", "feat_24"],
            plot_type="contour",
            x_quantile=True,
        )
        assert type(fig) == matplotlib.figure.Figure
        assert sorted(axes.keys()) == ["pdp_inter_ax", "title_ax"]
        assert len(axes["pdp_inter_ax"]) == 9
        assert type(axes["title_ax"]) == matplotlib.axes._subplots.Subplot
        for i in range(9):
            assert type(axes["pdp_inter_ax"][i]) == matplotlib.axes._subplots.Subplot

    def test_contour_pdp(self, pdp_interact_out):
        fig, axes = pdp_interact_plot(
            pdp_interact_out=pdp_interact_out,
            feature_names=["feat_67", "feat_24"],
            plot_type="contour",
            x_quantile=True,
            plot_pdp=True,
        )
        assert type(fig) == matplotlib.figure.Figure
        assert sorted(axes.keys()) == ["pdp_inter_ax", "title_ax"]
        assert len(axes["pdp_inter_ax"]) == 9
        assert sorted(axes["pdp_inter_ax"][0].keys()) == [
            "_pdp_inter_ax",
            "_pdp_x_ax",
            "_pdp_y_ax",
        ]
        assert type(axes["title_ax"]) == matplotlib.axes._subplots.Subplot
        for i in range(9):
            for k in ["_pdp_inter_ax", "_pdp_x_ax", "_pdp_y_ax"]:
                assert (
                    type(axes["pdp_inter_ax"][i][k])
                    == matplotlib.axes._subplots.Subplot
                )

    def test_grid(self, pdp_interact_out):
        fig, axes = pdp_interact_plot(
            pdp_interact_out=pdp_interact_out,
            feature_names=["feat_67", "feat_24"],
            plot_type="grid",
            x_quantile=True,
        )
        assert type(fig) == matplotlib.figure.Figure
        assert sorted(axes.keys()) == ["pdp_inter_ax", "title_ax"]
        assert len(axes["pdp_inter_ax"]) == 9
        assert type(axes["title_ax"]) == matplotlib.axes._subplots.Subplot
        for i in range(9):
            assert type(axes["pdp_inter_ax"][i]) == matplotlib.axes._subplots.Subplot

    def test_grid_pdp(self, pdp_interact_out):
        fig, axes = pdp_interact_plot(
            pdp_interact_out=pdp_interact_out,
            feature_names=["feat_67", "feat_24"],
            plot_type="contour",
            x_quantile=True,
            plot_pdp=True,
        )
        assert type(fig) == matplotlib.figure.Figure
        assert sorted(axes.keys()) == ["pdp_inter_ax", "title_ax"]
        assert len(axes["pdp_inter_ax"]) == 9
        assert sorted(axes["pdp_inter_ax"][0].keys()) == [
            "_pdp_inter_ax",
            "_pdp_x_ax",
            "_pdp_y_ax",
        ]
        assert type(axes["title_ax"]) == matplotlib.axes._subplots.Subplot
        for i in range(9):
            for k in ["_pdp_inter_ax", "_pdp_x_ax", "_pdp_y_ax"]:
                assert (
                    type(axes["pdp_inter_ax"][i][k])
                    == matplotlib.axes._subplots.Subplot
                )

    def test_contour_3(self, pdp_interact_out):
        fig, axes = pdp_interact_plot(
            pdp_interact_out=pdp_interact_out,
            feature_names=["feat_67", "feat_24"],
            plot_type="contour",
            x_quantile=True,
            plot_pdp=True,
            which_classes=[1, 2, 3],
        )
        assert type(fig) == matplotlib.figure.Figure
        assert sorted(axes.keys()) == ["pdp_inter_ax", "title_ax"]
        assert len(axes["pdp_inter_ax"]) == 3
        assert sorted(axes["pdp_inter_ax"][0].keys()) == [
            "_pdp_inter_ax",
            "_pdp_x_ax",
            "_pdp_y_ax",
        ]
        assert type(axes["title_ax"]) == matplotlib.axes._subplots.Subplot
        for i in range(3):
            for k in ["_pdp_inter_ax", "_pdp_x_ax", "_pdp_y_ax"]:
                assert (
                    type(axes["pdp_inter_ax"][i][k])
                    == matplotlib.axes._subplots.Subplot
                )

    def test_contour_1(self, pdp_interact_out):
        fig, axes = pdp_interact_plot(
            pdp_interact_out=pdp_interact_out,
            feature_names=["feat_67", "feat_24"],
            plot_type="contour",
            x_quantile=True,
            plot_pdp=True,
            which_classes=[1],
        )
        assert type(fig) == matplotlib.figure.Figure
        assert sorted(axes.keys()) == ["pdp_inter_ax", "title_ax"]
        assert sorted(axes["pdp_inter_ax"].keys()) == [
            "_pdp_inter_ax",
            "_pdp_x_ax",
            "_pdp_y_ax",
        ]
        assert type(axes["title_ax"]) == matplotlib.axes._subplots.Subplot
        for k in ["_pdp_inter_ax", "_pdp_x_ax", "_pdp_y_ax"]:
            assert type(axes["pdp_inter_ax"][k]) == matplotlib.axes._subplots.Subplot
