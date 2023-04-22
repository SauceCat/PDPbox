import pytest
import joblib
import os
import json
import pandas as pd


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
