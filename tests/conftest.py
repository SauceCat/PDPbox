import pytest
import joblib
import os
import json
import pandas as pd


def pytest_addoption(parser):
    parser.addoption("--runslow", action="store_true", help="run slow tests")

    parser.addoption("--rundisplay", action="store_true", help="run display tests")


def pytest_runtest_setup(item):
    if "slow" in item.keywords and not item.config.getvalue("runslow"):
        pytest.skip("need --runslow option to run")

    if "display" in item.keywords and not item.config.getvalue("rundisplay"):
        pytest.skip("need --rundisplay option to run")


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


@pytest.fixture(scope="session")
def ross(root_path):
    DIR = os.path.join(root_path, "pdpbox")
    with open(os.path.join(DIR, "examples/ross/ross_info.json"), "r") as fin:
        info = json.load(fin)

    dataset = {
        "data": pd.read_csv(os.path.join(DIR, "examples/ross/ross_data.csv")),
        "rf_model": joblib.load(os.path.join(DIR, "examples/ross/ross_model.pkl")),
        "features": info["features"],
        "target": info["target"],
    }

    return dataset


@pytest.fixture(scope="module")
def ross_data(ross):
    return ross["data"]


@pytest.fixture(scope="module")
def ross_features(ross):
    return ross["features"]


@pytest.fixture(scope="module")
def ross_target(ross):
    return ross["target"]


@pytest.fixture(scope="module")
def ross_model(ross):
    return ross["rf_model"]


@pytest.fixture(scope="session")
def otto(root_path):
    DIR = os.path.join(root_path, "pdpbox")
    with open(os.path.join(DIR, "examples/otto/otto_info.json"), "r") as fin:
        info = json.load(fin)

    dataset = {
        "data": pd.read_csv(os.path.join(DIR, "examples/otto/otto_data.csv")),
        "rf_model": joblib.load(os.path.join(DIR, "examples/otto/otto_model.pkl")),
        "features": info["features"],
        "target": info["target"],
    }

    return dataset


@pytest.fixture(scope="module")
def otto_data(otto):
    return otto["data"]


@pytest.fixture(scope="module")
def otto_features(otto):
    return otto["features"]


@pytest.fixture(scope="module")
def otto_target(otto):
    return otto["target"]


@pytest.fixture(scope="module")
def otto_model(otto):
    return otto["rf_model"]
