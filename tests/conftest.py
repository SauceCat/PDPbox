import pytest
import joblib
import os
import json
import numpy as np
import pandas as pd
from itertools import product
import matplotlib
from abc import ABC, abstractmethod


class DummyDataGenerator:
    DUMMY_ONEHOT_LENGTH = 4
    DUMMY_DF_LENGTH = 100
    DUMMY_MULTI_CLASSES = 3
    FEATURE_TYPES = ["numeric", "onehot", "binary"]

    def __init__(self):
        self.dummy_features = self._init_dummy_features()

    def _init_dummy_features(self):
        return {
            "numeric": np.random.randn(self.DUMMY_DF_LENGTH),
            "onehot": np.array(
                [
                    np.eye(self.DUMMY_ONEHOT_LENGTH)[i]
                    for i in np.random.randint(
                        0, self.DUMMY_ONEHOT_LENGTH, self.DUMMY_DF_LENGTH
                    )
                ]
            ),
            "binary": np.random.randint(0, 2, self.DUMMY_DF_LENGTH),
        }

    def shuffle_array(self, array):
        shuffled = np.array(array, copy=True)
        np.random.shuffle(shuffled)
        return shuffled

    def get_dummy_dfs(self, target_type=None):
        feat_types_1 = self.shuffle_array(self.FEATURE_TYPES)
        feat_types_2 = self.shuffle_array(self.FEATURE_TYPES)

        for t1, t2 in product(feat_types_1, feat_types_2):
            df = {}
            features = []
            for i, t in enumerate((t1, t2), start=1):
                if t == "onehot":
                    for j in range(self.DUMMY_ONEHOT_LENGTH):
                        df[f"feature_{i}_{j}"] = self.dummy_features[t][:, j]
                    features.append(
                        [f"feature_{i}_{j}" for j in range(self.DUMMY_ONEHOT_LENGTH)]
                    )
                else:
                    df[f"feature_{i}"] = self.dummy_features[t]
                    features.append(f"feature_{i}")
            df = pd.DataFrame(df)
            if target_type == "regression":
                df["target"] = np.random.randn(self.DUMMY_DF_LENGTH)
            elif target_type == "binary":
                df["target"] = np.random.randint(0, 2, self.DUMMY_DF_LENGTH)
            elif target_type == "multi-class":
                for i in range(self.DUMMY_MULTI_CLASSES):
                    df[f"target_{i}"] = np.random.randint(0, 2, self.DUMMY_DF_LENGTH)

            yield df, features, (t1, t2)


class DummyModel:
    def __init__(self, model_type, feat, feat_type, interact=False):
        self.feat = feat
        self.feat_type = feat_type
        self.interact = interact

        if model_type == "regression":
            self.predict = self.dummy_regression_predict
        elif model_type == "binary":
            self.predict_proba = self.dummy_binary_predict_proba
            self.n_classes_ = 2
        elif model_type == "multi-class":
            self.predict_proba = self.dummy_multi_class_predict_proba
            self.n_classes_ = DummyDataGenerator.DUMMY_MULTI_CLASSES

    def _compute_feature(self, x, feat, feat_type):
        if feat_type == "onehot":
            return np.argmax(x[feat].to_numpy(), axis=1)
        else:
            return x[feat].values

    def _compute_score(self, x, feat, feat_type):
        if feat_type == "onehot":
            return (np.argmax(x[feat].to_numpy(), axis=1) + 1) / len(feat)
        else:
            return (
                np.ones(len(x))
                if x[feat].max() == 0
                else x[feat].values / x[feat].max()
            )

    def dummy_regression_predict(self, x):
        if self.interact:
            return sum(
                self._compute_feature(x, f, t)
                for f, t in zip(self.feat, self.feat_type)
            )
        else:
            return self._compute_feature(x, self.feat, self.feat_type)

    def predict_proba_func(self, x):
        if self.interact:
            values = [
                self._compute_score(x, f, t) for f, t in zip(self.feat, self.feat_type)
            ]
            return values[0] * values[1]
        else:
            return self._compute_score(x, self.feat, self.feat_type)

    def dummy_binary_predict_proba(self, x):
        scores = self.predict_proba_func(x)
        return np.stack((1 - scores, scores)).T

    def dummy_multi_class_predict_proba(self, x):
        scores = self.predict_proba_func(x)
        each_scores = (1 - scores) / (self.n_classes_ - 1)
        proba = np.zeros((len(x), self.n_classes_))
        proba[:, 0] = scores
        proba[:, 1:] = each_scores[:, np.newaxis]
        return proba


class PlotTestBase(ABC):
    def setup(self):
        self.model_types = ["regression", "binary", "multi-class"]
        self.data_gen = DummyDataGenerator()

    @staticmethod
    def close_plt(params):
        if params.get("engine", "plotly") == "matplotlib":
            matplotlib.pyplot.close()

    @abstractmethod
    def get_plot_objs(self, model_type):
        pass

    @abstractmethod
    def check_common(self, plot_obj, model_type):
        pass

    @abstractmethod
    def test_plot_obj(self, model_type):
        pass

    @abstractmethod
    def test_plot(self, params):
        pass


@pytest.fixture(scope="session")
def root_path():
    return os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    )


@pytest.fixture(scope="session")
def titanic(root_path):
    DIR = os.path.join(root_path, "pdpbox")
    with open(os.path.join(DIR, "examples/titanic/titanic_info.json"), "r") as fin:
        info = json.load(fin)

    dataset = {
        "data": pd.read_csv(os.path.join(DIR, "examples/titanic/titanic_data.csv")),
        "xgb_model": joblib.load(
            os.path.join(DIR, "examples/titanic/titanic_model.pkl")
        ),
        "features": info["features"],
        "target": info["target"],
    }

    return dataset


@pytest.fixture(scope="module")
def titanic_data(titanic):
    return titanic["data"]


@pytest.fixture(scope="module")
def titanic_features(titanic):
    return titanic["features"]


@pytest.fixture(scope="module")
def titanic_target(titanic):
    return titanic["target"]


@pytest.fixture(scope="module")
def titanic_model(titanic):
    return titanic["xgb_model"]
