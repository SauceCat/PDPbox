
import joblib
import os

DIR = os.path.dirname(os.path.realpath(__file__))


def titanic():
	dataset = joblib.load(os.path.join(DIR, 'datasets/test_titanic.pkl'))
	return dataset


def ross():
	dataset = joblib.load(os.path.join(DIR, 'datasets/test_ross.pkl'))
	return dataset


def otto():
	dataset = joblib.load(os.path.join(DIR, 'datasets/test_otto.pkl'))
	return dataset