import joblib
import json
import pandas as pd
import os

DIR = os.path.dirname(os.path.realpath(__file__))


def titanic():
    with open(os.path.join(DIR, "examples/titanic/titanic_info.json"), "r") as fin:
        info = json.load(fin)

    example = {
        "data": pd.read_csv(os.path.join(DIR, "examples/titanic/titanic_data.csv")),
        "xgb_model": joblib.load(
            os.path.join(DIR, "examples/titanic/titanic_model.pkl")
        ),
        "features": info["features"],
        "target": info["target"],
    }

    return example


def ross():
    with open(os.path.join(DIR, "examples/ross/ross_info.json"), "r") as fin:
        info = json.load(fin)

    example = {
        "data": pd.read_csv(os.path.join(DIR, "examples/ross/ross_data.csv")),
        "rf_model": joblib.load(os.path.join(DIR, "examples/ross/ross_model.pkl")),
        "features": info["features"],
        "target": info["target"],
    }

    return example


def otto():
    with open(os.path.join(DIR, "examples/otto/otto_info.json"), "r") as fin:
        info = json.load(fin)

    example = {
        "data": pd.read_csv(os.path.join(DIR, "examples/otto/otto_data.csv")),
        "rf_model": joblib.load(os.path.join(DIR, "examples/otto/otto_model.pkl")),
        "features": info["features"],
        "target": info["target"],
    }

    return example
