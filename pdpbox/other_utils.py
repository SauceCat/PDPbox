
import numpy as np


def _check_feature(feature, df):
	# check feature
	if type(feature) == str:
		if feature not in df.columns.values:
			raise ValueError('feature does not exist: %s' % feature)
		if sorted(list(np.unique(df[feature]))) == [0, 1]:
			feature_type = 'binary'
		else:
			feature_type = 'numeric'
	elif type(feature) == list:
		if len(feature) < 2:
			raise ValueError('one-hot encoding feature should contain more than 1 element')
		if not set(feature) < set(df.columns.values):
			raise ValueError('feature does not exist: %s' % str(feature))
		feature_type = 'onehot'
	else:
		raise ValueError('feature: please pass a string or a list (for onehot encoding feature)')

	return feature_type


def _check_percentile_range(percentile_range):
	if percentile_range is not None:
		if len(percentile_range) != 2:
			raise ValueError('percentile_range: should contain 2 elements')
		if np.max(percentile_range) > 100 or np.min(percentile_range) < 0:
			raise ValueError('percentile_range: should be between 0 and 100')


def _check_target(target, df):
	if type(target) == str:
		if target not in df.columns.values:
			raise ValueError('target does not exist: %s' % target)
		if sorted(list(np.unique(df[target]))) == [0, 1]:
			target_type = 'binary'
		else:
			target_type = 'regression'
	elif type(target) == list:
		if len(target) < 2:
			raise ValueError('multi-class target should contain more than 1 element')
		if not set(target) < set(df.columns.values):
			raise ValueError('target does not exist: %s' % (str(target)))
		for target_idx in range(len(target)):
			if sorted(list(np.unique(df[target[target_idx]]))) != [0, 1]:
				raise ValueError('multi-class targets should be one-hot encoded: %s' % (str(target[target_idx])))
		target_type = 'multi-class'
	else:
		raise ValueError('target: please pass a string or a list (for multi-class targets)')

	return target_type


def _make_list(x):
	if type(x) == list:
		return x
	return [x]


def _expand_default(x, default):
	if x is None:
		return [default] * 2
	return x


def _check_model(model):
	try:
		n_classes = len(model.classes_)
		classifier = True
		predict = model.predict_proba
	except:
		n_classes = 0
		classifier = False
		predict = model.predict

	return n_classes, classifier, predict
