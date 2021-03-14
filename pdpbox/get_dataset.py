
import joblib
import json
import pandas as pd
import os

DIR = os.path.dirname(os.path.realpath(__file__))


def titanic():
	with open(os.path.join(DIR, 'datasets/titanic/titanic_info.json'), 'r') as fin:
		info = json.load(fin)

	dataset = {
		'data': pd.read_csv(os.path.join(DIR, 'datasets/titanic/titanic_data.csv')),
		'xgb_model': joblib.load(os.path.join(DIR, 'datasets/titanic/titanic_model.pkl')),
		'features': info['features'],
		'target': info['target']
	}

	return dataset


def ross():
	with open(os.path.join(DIR, 'datasets/ross/ross_info.json'), 'r') as fin:
		info = json.load(fin)

	dataset = {
		'data': pd.read_csv(os.path.join(DIR, 'datasets/ross/ross_data.csv')),
		'rf_model': joblib.load(os.path.join(DIR, 'datasets/ross/ross_model.pkl')),
		'features': info['features'],
		'target': info['target']
	}

	return dataset


def otto():
	with open(os.path.join(DIR, 'datasets/otto/otto_info.json'), 'r') as fin:
		info = json.load(fin)

	dataset = {
		'data': pd.read_csv(os.path.join(DIR, 'datasets/otto/otto_data.csv')),
		'rf_model': joblib.load(os.path.join(DIR, 'datasets/otto/otto_model.pkl')),
		'features': info['features'],
		'target': info['target']
	}

	return dataset