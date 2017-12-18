import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import matplotlib.patches as mpatches
import copy

import warnings
warnings.filterwarnings('ignore')

class pdp_isolate_obj:
	def __init__(self, n_classes, classifier, model_features, feature, feature_type, feature_grids, 
		actual_columns, display_columns, ice_lines, pdp):
		self._type = 'pdp_isolate_instance'
		self.n_classes = n_classes
		self.classifier = classifier
		self.model_features = model_features
		self.feature = feature
		self.feature_type = feature_type
		self.feature_grids = feature_grids
		self.actual_columns = actual_columns
		self.display_columns = display_columns
		self.ice_lines = ice_lines
		self.pdp = pdp


class pdp_interact_obj:
	def __init__(self, n_classes, classifier, model_features, features, feature_types, feature_grids, 
		pdp_isolate_out1, pdp_isolate_out2, pdp):
		self._type = 'pdp_interact_instance'
		self.n_classes = n_classes
		self.classifier = classifier
		self.model_features = model_features
		self.features = features
		self.feature_types = feature_types
		self.feature_grids = feature_grids
		self.pdp_isolate_out1 = pdp_isolate_out1
		self.pdp_isolate_out2 = pdp_isolate_out2
		self.pdp = pdp


def pdp_isolate(model, train_X, feature, num_grid_points=10, percentile_range=None, cust_grid_points=None, predict_kwds={}):
	'''
	model: sklearn model
		a fitted model
	train_X: pandas DataFrame
		dataset on which the model is trained
	feature: string or list
		column to investigate (for one-hot encoding features, a list of columns should be provided)
	num_grid_points: integer, default=10
		number of grid points for numeric features
	percentile_range: (low, high), default=None
		percentile range to consider for numeric features
	cust_grid_points: list, default=None
		customized grid points
	predict_kwds: dict, default={}
		keywords to be passed to the model's predict function
	'''
	# check model
	try:
		n_classes = len(model.classes_)
		classifier = True
		predict = model.predict_proba
	except:
		n_classes = 0
		classifier = False
		predict = model.predict 
		
	# check input dataset
	if type(train_X) != pd.core.frame.DataFrame:
		raise ValueError('train_X: only accept pandas DataFrame')

	# check num_grid_points
	if num_grid_points is not None and type(num_grid_points) != int:
		raise ValueError('num_grid_points: should be an integer')

	# check percentile_range
	if percentile_range is not None:
		if type(percentile_range) != tuple:
			raise ValueError('percentile_range: should be a tuple')
		if len(percentile_range) != 2:
			raise ValueError('percentile_range: should contain 2 elements')
		if np.max(percentile_range) > 100 or np.min(percentile_range) < 0:
			raise ValueError('percentile_range: should be between 0 and 100')

	_train_X = train_X.copy()
	model_features = _train_X.columns.values
	
	# check feature
	if type(feature) == str:
		if feature not in _train_X.columns.values:
			raise ValueError('feature does not exist: %s' %(feature))
		if sorted(list(np.unique(_train_X[feature]))) == [0, 1]:
			feature_type='binary'
		else:
			feature_type='numeric'
	elif type(feature) == list:
		if len(feature) < 2:
			raise ValueError('one-hot encoding feature should contain more than 1 element')
		if not set(feature) < set(_train_X.columns.values):
			raise ValueError('feature does not exist: %s' %(str(feature)))
		feature_type='onehot'
	else:
		raise ValueError('feature: please pass a string or a list (for onehot encoding feature)')

	# check cust_grid_points
	if (feature_type != 'numeric') and (cust_grid_points is not None):
		raise ValueError('only numeric feature can accept cust_grid_points')
	if (cust_grid_points is not None) and (type(cust_grid_points) != list):
		raise ValueError('cust_grid_points: should be a list')

	# create feature grids
	if feature_type == 'binary':
		feature_grids = np.array([0, 1])
		display_columns = ['%s_0' %(feature), '%s_1' %(feature)]
	if feature_type == 'numeric':
		if cust_grid_points is None:
			feature_grids = _get_grids(_train_X[feature], num_grid_points, percentile_range)
		else:
			feature_grids = np.array(sorted(cust_grid_points))
		display_columns = feature_grids
	if feature_type == 'onehot':
		feature_grids = np.array(feature)
		display_columns = feature

	# store the ICE lines
	# for multi-classifier, a dictionary is created
	if n_classes > 2:
		ice_lines = {}
		for n_class in range(n_classes):
			ice_lines['class_%d' %(n_class)] = pd.DataFrame()
	else:
		ice_lines = pd.DataFrame()
	
	# do prediction chunk by chunk to save memory usage
	data_chunk_size = int(_train_X.shape[0] / feature_grids.size)
	if data_chunk_size == 0:
		data_chunk_size = _train_X.shape[0]
	
	# get the actual prediction and actual values
	actual_preds = predict(_train_X, **predict_kwds)
	actual_columns = []
	if feature_type == 'onehot':
		for feat in feature:
			actual_column = 'actual_%s' %(feat)
			_train_X[actual_column] = _train_X[feat]
			actual_columns.append(actual_column)
	else:
		actual_columns.append('actual_%s' %(feature))
		_train_X[actual_columns[0]] = _train_X[feature]

	if n_classes > 2:
		for n_class in range(n_classes):
			_train_X['actual_preds_class_%d' %(n_class)] = actual_preds[:, n_class]
	else:
		if classifier:
			_train_X['actual_preds'] = actual_preds[:, 1]
		else:
			_train_X['actual_preds'] = actual_preds
	
	# get ice lines
	for i in range(0, len(_train_X), data_chunk_size):
		data_chunk = _train_X[i : (i + data_chunk_size)].reset_index(drop=True)
		ice_chunk = _make_ice_data(data_chunk[model_features], feature, feature_type, feature_grids)
		preds = predict(ice_chunk[model_features], **predict_kwds)
		
		if n_classes > 2:
			for n_class in range(n_classes):
				ice_chunk_result = pd.DataFrame(preds[:, n_class].reshape((data_chunk.shape[0], feature_grids.size)), columns=display_columns)
				ice_chunk_result = pd.concat([ice_chunk_result, data_chunk[actual_columns]], axis=1)
				ice_chunk_result['actual_preds'] = data_chunk['actual_preds_class_%d' %(n_class)].values
				ice_lines['class_%d' %(n_class)] = pd.concat([ice_lines['class_%d' %(n_class)], ice_chunk_result])
		else:
			if classifier:
				ice_chunk_result = pd.DataFrame(preds[:, 1].reshape((data_chunk.shape[0], feature_grids.size)), columns=display_columns)
			else:
				ice_chunk_result = pd.DataFrame(preds.reshape((data_chunk.shape[0], feature_grids.size)), columns=display_columns)
			ice_chunk_result = pd.concat([ice_chunk_result, data_chunk[actual_columns]], axis=1)
			ice_chunk_result['actual_preds'] = data_chunk['actual_preds'].values
			ice_lines = pd.concat([ice_lines, ice_chunk_result])

		ice_chunk.drop(ice_chunk.columns.values, axis=1, inplace=True)
		data_chunk.drop(data_chunk.columns.values, axis=1, inplace=True)
			
	# calculate pdp
	if n_classes > 2:
		pdp = {}
		for n_class in range(n_classes):
			pdp['class_%d' %(n_class)] = ice_lines['class_%d' %(n_class)][display_columns].mean().values
	else:
		pdp = ice_lines[display_columns].mean().values
		
	if n_classes > 2:
		pdp_isolate_out = {}
		for n_class in range(n_classes):
			pdp_isolate_out['class_%d' %(n_class)] = pdp_isolate_obj(n_classes=n_classes, classifier=classifier, model_features = model_features, 
				feature=feature, feature_type=feature_type, feature_grids=feature_grids, actual_columns=actual_columns, 
				display_columns=display_columns, ice_lines=ice_lines['class_%d' %(n_class)], pdp=pdp['class_%d' %(n_class)])
	else:  
		pdp_isolate_out = pdp_isolate_obj(n_classes=n_classes, classifier=classifier, model_features=model_features, feature=feature, 
			feature_type=feature_type, feature_grids=feature_grids, actual_columns=actual_columns, display_columns=display_columns, 
			ice_lines=ice_lines, pdp=pdp)
			
	return pdp_isolate_out


def _get_grids(x, num_grid_points, percentile_range):    
	if num_grid_points >= np.unique(x).size:
		grids = np.unique(x)
	else:
		if percentile_range is not None:
			grids = np.unique(np.percentile(x, np.linspace(np.min(percentile_range), np.max(percentile_range), num_grid_points)))
		else:
			grids = np.unique(np.percentile(x, np.linspace(0, 100, num_grid_points)))
			
	return np.array([round(val, 2) for val in grids])


def _make_ice_data(data, feature, feature_type, feature_grids):
	ice_data = pd.DataFrame(np.repeat(data.values, feature_grids.size, axis=0), columns=data.columns)
	
	if feature_type == 'onehot':
		for i, col in enumerate(feature):
			col_value = [0] * feature_grids.size
			col_value[i] = 1
			ice_data[col] = np.tile(col_value, data.shape[0])
	else:
		ice_data[feature] = np.tile(feature_grids, data.shape[0])

	return ice_data


def pdp_interact(model, train_X, features, num_grid_points=[10, 10], percentile_ranges=[None, None], cust_grid_points=[None, None], predict_kwds={}):
	'''
	model: sklearn model
		a fitted model
	train_X: pandas DataFrame
		dataset on which the model is trained
	features: list
		a list containing two features
	num_grid_points: list, default=[10, 10]
		a list of number of grid points for each feature
	percentile_ranges: list, default=[None, None]
		a list of percentile range to consider for each feature
	cust_grid_points: list, default=None
		a list of customized grid points
	predict_kwds: dict, default={}
		keywords to be passed to the model's predict function
	'''

	# check input dataset
	if type(train_X) != pd.core.frame.DataFrame:
		raise ValueError('train_X: only accept pandas DataFrame')
	_train_X = train_X.copy()

	#check features
	if type(features) != list:
		raise ValueError('features: a list of features should be passed')
	if len(features) != 2:
		raise ValueError('features: only 2 features should be passed')
	
	# check num_grid_points
	if (type(num_grid_points) != list) or (len(num_grid_points) != 2):
		raise ValueError('num_grid_points: should be a list with 2 elements')
	if ((num_grid_points[0] is not None) and (type(num_grid_points[0])!=int)) or ((num_grid_points[1] is not None) and (type(num_grid_points[1])!=int)):
		raise ValueError('num_grid_points: each element should be None or an integer')

	# check percentile_range
	if type(percentile_ranges) != list:
		raise ValueError('percentile_ranges: should be a list')
	if len(percentile_ranges) != 2:
		raise ValueError('percentile_ranges: should only contain 2 elements')

	# check cust_grid_points
	if type(cust_grid_points) != list:
		raise ValueError('cust_grid_points: should be a list')
	if len(cust_grid_points) != 2:
		raise ValueError('cust_grid_points: should only contain 2 elements')

	# check model
	try:
		n_classes = len(model.classes_)
		classifier = True
		predict = model.predict_proba
	except:
		n_classes = 0
		classifier = False
		predict = model.predict 

	# calculate pdp_isolate for each feature
	pdp_isolate_out1 = pdp_isolate(model, _train_X, features[0], num_grid_points=num_grid_points[0], percentile_range=percentile_ranges[0], cust_grid_points=cust_grid_points[0])
	pdp_isolate_out2 = pdp_isolate(model, _train_X, features[1], num_grid_points=num_grid_points[1], percentile_range=percentile_ranges[1], cust_grid_points=cust_grid_points[1])

	# whether it is for multiclassifier
	if type(pdp_isolate_out1) == dict:
		model_features = pdp_isolate_out1['class_0'].model_features
		feature_grids = [pdp_isolate_out1['class_0'].feature_grids, pdp_isolate_out2['class_0'].feature_grids]
		feature_types = [pdp_isolate_out1['class_0'].feature_type, pdp_isolate_out2['class_0'].feature_type]
	else:
		model_features = pdp_isolate_out1.model_features
		feature_grids = [pdp_isolate_out1.feature_grids, pdp_isolate_out2.feature_grids]
		feature_types = [pdp_isolate_out1.feature_type, pdp_isolate_out2.feature_type]

	# make features into list
	feature_list = []
	for feat in features:
		if type(feat) == list:
			feature_list += feat
		else:
			feature_list.append(feat)
		
	# do prediction chunk by chunk to save memory usage
	ice_lines = pd.DataFrame()
	grids_size = len(feature_grids[0]) * len(feature_grids[1])

	data_chunk_size = int(_train_X.shape[0] / grids_size)
	if data_chunk_size == 0:
		data_chunk_size = _train_X.shape[0]
	
	for i in range(0, len(_train_X), data_chunk_size):
		data_chunk = _train_X[i:(i + data_chunk_size)].reset_index(drop=True)
		ice_chunk = _make_ice_data_inter(data_chunk[model_features], features, feature_types, feature_grids)
		preds = predict(ice_chunk[model_features], **predict_kwds)
		result_chunk = ice_chunk[feature_list].copy()
		
		if n_classes > 2:
			for n_class in range(n_classes):
				result_chunk['class_%d_preds' %(n_class)] = preds[:, n_class]
		else:
			if classifier:
				result_chunk['preds'] = preds[:, 1]
			else:
				result_chunk['preds'] = preds
		ice_lines = pd.concat([ice_lines, result_chunk])
		ice_chunk.drop(ice_chunk.columns.values, axis=1, inplace=True)
		data_chunk.drop(data_chunk.columns.values, axis=1, inplace=True)

	pdp = ice_lines.groupby(feature_list, as_index=False).mean()

	if n_classes > 2:
		pdp_interact_out = {}
		for n_class in range(n_classes):
			_pdp = pdp[feature_list + ['class_%d_preds' %(n_class)]].rename(columns={'class_%d_preds' %(n_class): 'preds'})
			pdp_interact_out['class_%d' %(n_class)] = pdp_interact_obj(n_classes=n_classes, classifier=classifier, model_features=model_features,
				features=features, feature_types=feature_types, feature_grids=feature_grids,
				pdp_isolate_out1=pdp_isolate_out1['class_%d' %(n_class)], pdp_isolate_out2=pdp_isolate_out2['class_%d' %(n_class)], pdp=_pdp)
	else:  
		pdp_interact_out = pdp_interact_obj(n_classes=n_classes, classifier=classifier, model_features=model_features,
			features=features, feature_types=feature_types, feature_grids=feature_grids, 
			pdp_isolate_out1=pdp_isolate_out1, pdp_isolate_out2=pdp_isolate_out2, pdp=pdp)
			
	return pdp_interact_out


def _make_ice_data_inter(data, features, feature_types, feature_grids):
	grids_size = len(feature_grids[0]) * len(feature_grids[1])
	ice_data = pd.DataFrame(np.repeat(data.values, grids_size, axis=0), columns=data.columns)

	if feature_types[0] == 'onehot':
		for i, col in enumerate(features[0]):
			col_value = [0] * feature_grids[0].size
			col_value[i] = 1
			ice_data[col] = np.tile(col_value, data.shape[0] * feature_grids[1].size)
	else:
		ice_data[features[0]] = np.tile(feature_grids[0], data.shape[0] * feature_grids[1].size)

	if feature_types[1] == 'onehot':
		for i, col in enumerate(features[1]):
			col_value = [0] * feature_grids[1].size
			col_value[i] = 1
			ice_data[col] = np.tile(np.repeat(col_value, feature_grids[0].size, axis=0), data.shape[0])
	else:
		ice_data[features[1]] = np.tile(np.repeat(feature_grids[1], feature_grids[0].size, axis=0), data.shape[0])

	return ice_data


def pdp_plot(pdp_isolate_out, feature_name, center=True, plot_org_pts=False, plot_lines=False, frac_to_plot=1, cluster=False, n_cluster_centers=None, 
	cluster_method=None, x_quantile=False, figsize=None, ncols=None, plot_params=None, multi_flag=False, which_class=None):
	'''
	pdp_isolate_out: instance of pdp_isolate_obj
		a calculated pdp_isolate_obj instance
	feature_name: string
		name of the feature, not necessary the same as the column name
	center: boolean, default=True
		whether to center the plot
	plot_org_pts: boolean, default=False
		whether to plot out the original points
	plot_lines: boolean, default=False
		whether to plot out the individual lines
	frac_to_plot: float or integer, default=1
		how many points or lines to plot, can be a integer or a float
	cluster: boolean, default=False
		whether to cluster the individual lines and only plot out the cluster centers
	n_cluster_centers: integer, default=None
		number of cluster centers
	cluster_method: string, default=None
		cluster method to use, default is KMeans, if 'approx' is passed, MiniBatchKMeans is used
	x_quantile: boolean, default=False
		whether to construct x axis ticks using quantiles
	figsize: (width, height), default=None
		figure size
	ncols: integer, default=None
		used under multiclass mode
	plot_params: dict, default=None
		values of plot parameters
	multi_flag: boolean, default=False
		whether it is a subplot of a multiclass plot
	which_class: integer, default=None
		which class to plot
	'''

	# check pdp_isolate_out
	if type(pdp_isolate_out) == dict:
		try:
			if pdp_isolate_out['class_0']._type != 'pdp_isolate_instance':
				raise ValueError('pdp_isolate_out: should be an instance of pdp_isolate_obj')
		except: 
			raise ValueError('pdp_isolate_out: should be an instance of pdp_isolate_obj')
	else:
		try:
			if pdp_isolate_out._type != 'pdp_isolate_instance':
				raise ValueError('pdp_isolate_out: should be an instance of pdp_isolate_obj')
		except: 
			raise ValueError('pdp_isolate_out: should be an instance of pdp_isolate_obj')

	# check feature_name
	if type(feature_name) != str:
		raise ValueError('feature_name: should be a string')

	# check other inputs (all boolean values)
	if type(center) != bool:
		raise ValueError('center: should be a boolean value')
	if type(plot_org_pts) != bool:
		raise ValueError('plot_org_pts: should be a boolean value')
	if type(plot_lines) != bool:
		raise ValueError('plot_lines: should be a boolean value')
	if type(cluster) != bool:
		raise ValueError('cluster: should be a boolean value')
	if type(x_quantile) != bool:
		raise ValueError('x_quantile: should be a boolean value')
	if type(multi_flag) != bool:
		raise ValueError('multi_flag: should be a boolean value')

	# check frac_to_plot
	if (type(frac_to_plot) != float) and (type(frac_to_plot) != int):
		raise ValueError('frac_to_plot: should be a float or an integer')
	if type(frac_to_plot) == float:
		if (frac_to_plot <= 0.0) or (frac_to_plot > 1.0):
			raise ValueError('frac_to_plot: should in range(0, 1) when it is a float')
	if type(pdp_isolate_out) == dict:
		if (type(frac_to_plot) == int) and (frac_to_plot > pdp_isolate_out['class_0'].ice_lines.shape[0]):
			raise ValueError('frac_to_plot: sample size should not be larger than the population size')
	else:
		if (type(frac_to_plot) == int) and (frac_to_plot > pdp_isolate_out.ice_lines.shape[0]):
			raise ValueError('frac_to_plot: sample size should not be larger than the population size')

	# check n_cluster_centers
	if cluster:
		if n_cluster_centers is None:
			raise ValueError('n_cluster_centers: should not be None under clustering mode')
		if type(n_cluster_centers) != int:
			raise ValueError('n_cluster_centers: should be an integer')
		if cluster_method is not None:
			if (cluster_method != 'accurate') and (cluster_method != 'approx'):
				raise ValueError("cluster_method: should be 'accurate' or 'approx'")

	# check figsize
	if figsize is not None:
		if type(figsize) != tuple:
			raise ValueError('figsize: should be a tuple')
		if len(figsize) != 2:
			raise ValueError('figsize: should contain 2 elements: (width, height)')

	# check ncols
	if (ncols is not None) and (type(ncols) != int):
		raise ValueError('ncols: should be an integer')

	# check plot_params
	if (plot_params is not None) and (type(plot_params) != dict):
		raise ValueError('plot_params: should be a dictionary')

	# check which_class
	if multi_flag:
		if type(pdp_isolate_out) != dict:
			raise ValueError('multi_flag: can only be used under multi-class mode')
		if which_class is None:
			raise ValueError('which_class: should not be None when multi_flag is on')
		if type(which_class) != int:
			raise ValueError('which_class: should be an integer')
		if which_class >= len(pdp_isolate_out.keys()):
			raise ValueError('which_class: class does not exist')

	if type(pdp_isolate_out) == dict:
		n_classes = len(pdp_isolate_out.keys())

		if multi_flag:
			if figsize is None:
				plt.figure(figsize=(16, 9))
			else:
				plt.figure(figsize=figsize)

			gs = GridSpec(5, 1)
			ax1 = plt.subplot(gs[0, :])
			_pdp_plot_title(pdp_isolate_out=pdp_isolate_out['class_%d' %(which_class)], feature_name=feature_name, ax=ax1, figsize=figsize, multi_flag=multi_flag, 
				which_class=which_class, plot_params=plot_params)

			ax2 = plt.subplot(gs[1:, :])
			_pdp_plot(pdp_isolate_out=pdp_isolate_out['class_%d' %(which_class)], feature_name=feature_name, center=center, 
				plot_org_pts=plot_org_pts, plot_lines=plot_lines, frac_to_plot=frac_to_plot, cluster=cluster, n_cluster_centers=n_cluster_centers, 
				cluster_method=cluster_method, x_quantile=x_quantile, ax=ax2, plot_params=plot_params)
		else:
			if ncols == None:
				ncols = 2
			nrows = np.ceil(float(n_classes) / ncols)

			if figsize is None:
				figwidth = 16
			else:
				figwidth = figsize[0]

			plt.figure(figsize=(figwidth, figwidth / 8.0))
			ax1 = plt.subplot(111)
			_pdp_plot_title(pdp_isolate_out=pdp_isolate_out['class_0'], feature_name=feature_name, ax=ax1, figsize=figsize, multi_flag=multi_flag, 
				which_class=which_class, plot_params=plot_params)

			plt.figure(figsize=(figwidth, (figwidth/ncols)*nrows))
			for n_class in range(n_classes):
				ax2 = plt.subplot(nrows, ncols, n_class+1)
				_pdp_plot(pdp_isolate_out=pdp_isolate_out['class_%d' %(n_class)], feature_name=feature_name + ' class_%d' %(n_class), center=center, 
					plot_org_pts=plot_org_pts, plot_lines=plot_lines, frac_to_plot=frac_to_plot, cluster=cluster, n_cluster_centers=n_cluster_centers, 
					cluster_method=cluster_method, x_quantile=x_quantile, ax=ax2, plot_params=plot_params)
	else:
		if figsize is None:
			plt.figure(figsize=(16, 9))
		else:
			plt.figure(figsize=figsize)
			
		gs = GridSpec(5, 1)
		ax1 = plt.subplot(gs[0, :])
		_pdp_plot_title(pdp_isolate_out=pdp_isolate_out, feature_name=feature_name, ax=ax1, figsize=figsize, multi_flag=False, which_class=None, 
			plot_params=plot_params)

		ax2 = plt.subplot(gs[1:, :])
		_pdp_plot(pdp_isolate_out=pdp_isolate_out, feature_name=feature_name, center=center, plot_org_pts=plot_org_pts, plot_lines=plot_lines, 
			frac_to_plot=frac_to_plot, cluster=cluster, n_cluster_centers=n_cluster_centers, cluster_method=cluster_method, x_quantile=x_quantile, 
			ax=ax2, plot_params=plot_params)


def _pdp_plot_title(pdp_isolate_out, feature_name, ax, figsize, multi_flag, which_class, plot_params):

	font_family = 'Arial'
	title = 'ICEplot for %s' %(feature_name)
	subtitle = "Number of unique grid points: %d" %(len(pdp_isolate_out.feature_grids))

	if figsize is not None:
		title_fontsize = np.max([15 * (figsize[0] / 16.0), 10])
		subtitle_fontsize = np.max([12 * (figsize[0] / 16.0), 8])
	else:
		title_fontsize=15
		subtitle_fontsize=12

	if plot_params is not None:
		if 'font_family' in plot_params.keys():
			font_family = plot_params['font_family']
		if 'title' in plot_params.keys():
			title = plot_params['title']
		if 'title_fontsize' in plot_params.keys():
			title_fontsize = plot_params['title_fontsize']
		if 'subtitle_fontsize' in plot_params.keys():
			subtitle_fontsize = plot_params['subtitle_fontsize']

	ax.set_facecolor('white')
	if multi_flag:
		ax.text(0, 0.7, title, va="top", ha="left", fontsize=title_fontsize, fontname=font_family)
		ax.text(0, 0.45, "For Class %d" %(which_class), va="top", ha="left", fontsize=subtitle_fontsize, fontname=font_family)
		ax.text(0, 0.25, subtitle, va="top", ha="left", fontsize=subtitle_fontsize, fontname=font_family, color='grey')
	else:
		ax.text(0, 0.7, title, va="top", ha="left", fontsize=title_fontsize, fontname=font_family)
		ax.text(0, 0.4, subtitle, va="top", ha="left", fontsize=subtitle_fontsize, fontname=font_family, color='grey')
	ax.axis('off')


def _pdp_plot(pdp_isolate_out, feature_name, center, plot_org_pts, plot_lines, frac_to_plot, cluster, n_cluster_centers, 
	cluster_method, x_quantile, ax, plot_params):

	font_family='Arial'
	xticks_rotation = 0
	if plot_params is not None:
		if 'font_family' in plot_params.keys():
			font_family = plot_params['font_family']
		if 'xticks_rotation' in plot_params.keys():
			xticks_rotation = plot_params['xticks_rotation']

	_axis_modify(font_family, ax)
	ax.set_xlabel(feature_name, fontsize=10)

	feature_type = pdp_isolate_out.feature_type
	display_columns = pdp_isolate_out.display_columns
	actual_columns = pdp_isolate_out.actual_columns

	if feature_type == 'binary' or feature_type == 'onehot' or x_quantile:
		x = range(len(display_columns))
		ax.set_xticks(x)
		ax.set_xticklabels(display_columns, rotation=xticks_rotation)
	else:
		x = display_columns

	ice_lines = copy.deepcopy(pdp_isolate_out.ice_lines)
	pdp_y = copy.deepcopy(pdp_isolate_out.pdp)
	std_fill = True
	std_hl = False

	if center:
		pdp_y -= pdp_y[0]
		for col in display_columns[1:]:
			ice_lines[col] -= ice_lines[display_columns[0]]
		ice_lines['actual_preds'] -= ice_lines[display_columns[0]]
		ice_lines[display_columns[0]] = 0

	if plot_lines:
		if (frac_to_plot > 1) and (frac_to_plot > len(ice_lines)):
			raise ValueError("frac_to_plot: sample amount %d larger than population %d" %(frac_to_plot, len(ice_lines))) 
		ice_plot_data = _sample_data(ice_lines=ice_lines, frac_to_plot=frac_to_plot)
		std_fill = False
		std_hl = True
		if not cluster:
			_ice_line_plot(x=x, ice_plot_data=ice_plot_data, display_columns=display_columns, ax=ax, plot_params=plot_params)  

	if plot_org_pts:
		ice_lines_temp = ice_lines.copy()
		if feature_type == 'onehot':
			ice_lines_temp['x'] = ice_lines_temp[actual_columns].apply(lambda x: _find_onehot_actual(x), axis=1)
			ice_lines_temp = ice_lines_temp[ice_lines_temp['x'].isnull()==False].reset_index(drop=True)
		elif feature_type=='numeric':
			feature_grids = pdp_isolate_out.feature_grids
			ice_lines_temp = ice_lines_temp[(ice_lines_temp[actual_columns[0]] >= feature_grids[0]) & (ice_lines_temp[actual_columns[0]] <= feature_grids[-1])]
			if x_quantile:
				ice_lines_temp['x'] = ice_lines_temp[actual_columns[0]].apply(lambda x : _find_closest(x, feature_grids))
			else:
				ice_lines_temp['x'] = ice_lines_temp[actual_columns[0]]
		else:
			ice_lines_temp['x'] = ice_lines_temp[actual_columns[0]]

		if (frac_to_plot > 1) and (frac_to_plot > len(ice_lines_temp)):
			raise ValueError("frac_to_plot: sample amount %d larger than population %d" %(frac_to_plot, len(ice_lines_temp))) 
		ice_plot_data_pts = _sample_data(ice_lines=ice_lines_temp, frac_to_plot=frac_to_plot)
		_ice_plot_pts(ice_plot_data_pts=ice_plot_data_pts, ax=ax, plot_params=plot_params)

	if cluster:
		std_fill = False
		std_hl = True
		_ice_cluster_plot(x=x, ice_lines=ice_lines, display_columns=display_columns, n_cluster_centers=n_cluster_centers, 
			cluster_method=cluster_method, ax=ax, plot_params=plot_params)

	std = ice_lines[display_columns].std().values
	_pdp_std_plot(x=x, y=pdp_y, std=std, std_fill=std_fill, std_hl=std_hl, plot_org_pts=plot_org_pts, plot_lines=plot_lines, ax=ax, plot_params=plot_params)


def _find_onehot_actual(x):
	try:
		value = list(x).index(1)
	except:
		value = np.nan 
	return value


def _find_closest(x, feature_grids):
	values = list(feature_grids)
	return values.index(min(values, key=lambda y:abs(y-x)))


def _sample_data(ice_lines, frac_to_plot):
	if frac_to_plot < 1.:
		ice_plot_data = ice_lines.sample(int(ice_lines.shape[0] * frac_to_plot))
	elif frac_to_plot > 1:
		ice_plot_data = ice_lines.sample(frac_to_plot)
	else:
		ice_plot_data = ice_lines.copy()
	ice_plot_data = ice_plot_data.reset_index(drop=True)
	return ice_plot_data


def _pdp_std_plot(x, y, std, std_fill, std_hl, plot_org_pts, plot_lines, ax, plot_params):
	upper = y + std
	lower = y - std

	pdp_color = '#1A4E5D'
	pdp_hl_color = '#FEDC00'
	pdp_linewidth = 2
	zero_color = '#E75438'
	zero_linewidth = 1.5
	fill_color = '#66C2D7'
	fill_alpha = 0.2
	markersize = 5

	if plot_params is not None:
		if 'pdp_color' in plot_params.keys():
			pdp_color = plot_params['pdp_color']
		if 'pdp_hl_color' in plot_params.keys():
			pdp_hl_color = plot_params['pdp_hl_color']
		if 'pdp_linewidth' in plot_params.keys():
			pdp_linewidth = plot_params['pdp_linewidth']
		if 'zero_color' in plot_params.keys():
			zero_color = plot_params['zero_color']
		if 'zero_linewidth' in plot_params.keys():
			zero_linewidth = plot_params['zero_linewidth']
		if 'fill_color' in plot_params.keys():
			fill_color = plot_params['fill_color']
		if 'fill_alpha' in plot_params.keys():
			fill_alpha = plot_params['fill_alpha']
		if 'markersize' in plot_params.keys():
			markersize = plot_params['markersize']

	if std_hl:
		ax.plot(x, y, color=pdp_hl_color, linewidth=pdp_linewidth * 3, alpha=0.8)

	ax.plot(x, y, color=pdp_color, linewidth=pdp_linewidth, marker='o', markersize=markersize)
	ax.plot(x, [0] * y, linestyle='--', linewidth=zero_linewidth, color=zero_color)

	if std_fill:
		ax.fill_between(x, upper, lower, alpha=fill_alpha, color=fill_color)

	#if not plot_org_pts and not plot_lines:
	ax.set_ylim(np.min([np.min(lower) * 2, 0]), np.max([np.max(upper) * 2, 0]))
			

def _ice_plot_pts(ice_plot_data_pts, ax, plot_params):
	point_size = 50
	point_pos_color = '#5BB573'
	point_neg_color = '#E75438'

	if plot_params is not None:
		if 'point_size' in plot_params.keys():
			point_size = plot_params['point_size']
		if 'point_pos_color' in plot_params.keys():
			point_pos_color = plot_params['point_pos_color']
		if 'point_neg_color' in plot_params.keys():
			point_neg_color = plot_params['point_neg_color']

	ice_plot_data_pts['color'] = ice_plot_data_pts['actual_preds'].apply(lambda x : point_pos_color if x >=0 else point_neg_color)
	ax.scatter(ice_plot_data_pts['x'], ice_plot_data_pts['actual_preds'], s=point_size, marker="+", linewidth=1, color=ice_plot_data_pts['color'])     


def _ice_line_plot(x, ice_plot_data, display_columns, ax, plot_params):
	linewidth = np.max([1.0 / np.log10(ice_plot_data.shape[0]), 0.3])
	linealpha = np.max([1.0 / np.log10(ice_plot_data.shape[0]), 0.3])

	line_cmap = 'Blues'
	if plot_params is not None:
		if 'line_cmap' in plot_params.keys():
			line_cmap = plot_params['line_cmap']
			
	colors = plt.get_cmap(line_cmap)(np.linspace(0,1,20))[5:15]

	for i in range(len(ice_plot_data)):
		y = list(ice_plot_data[display_columns].iloc[i].values)
		ax.plot(x, y, linewidth=linewidth, c=colors[i%10], alpha=linealpha)


def _ice_cluster_plot(x, ice_lines, display_columns, n_cluster_centers, cluster_method, ax, plot_params):
	if cluster_method == 'approx':
		from sklearn.cluster import MiniBatchKMeans
		kmeans = MiniBatchKMeans(n_clusters=n_cluster_centers, random_state=0, verbose=0)
	else:
		from sklearn.cluster import KMeans
		kmeans = KMeans(n_clusters=n_cluster_centers, random_state=0, n_jobs=1)
		
	kmeans.fit(ice_lines[display_columns])
	cluster_plot_data = pd.DataFrame(kmeans.cluster_centers_, columns=display_columns)

	cluster_cmap = 'Blues'

	if plot_params is not None:
		if 'cluster_cmap' in plot_params.keys():
			cluster_cmap = plot_params['cluster_cmap']
			
	colors = plt.get_cmap(cluster_cmap)(np.linspace(0,1,20))[5:15]

	for i in range(len(cluster_plot_data)):
		y = list(cluster_plot_data[display_columns].iloc[i].values)
		ax.plot(x, y, linewidth=1, c=colors[i%10])


def pdp_interact_plot(pdp_interact_out, feature_names, center=True, plot_org_pts=False, plot_lines=False, 
			frac_to_plot=1., cluster=False, n_cluster_centers=None, cluster_method=None, x_quantile=False,
			figsize=None, plot_params=None, multi_flag=False, which_class=None, only_inter=False, ncols=None):
	'''
	pdp_interact_out: pdp_interact_obj
		a calculated pdp_interact_obj
	feature_names: list
		a list of feature names
	center: boolean, default=True
		whether to center the individual pdp plot
	plot_org_pts: boolean, default=False
		whether to plot out the original points for the individual pdp plot
	plot_lines: boolean, default=False
		whether to plot out the individual lines for the individual pdp plot
	frac_to_plot: integer or float, default=1
		how many lines or points to plot for the individual pdp plot
	cluster: boolean, default=False
		whether to cluster the the individual pdp plot
	n_cluster_centers: integer, default=None
		number of cluster centers for the individual pdp plot under clustering mode
	cluster_method: string, default=None
		clustering method to use
	x_quantile: boolean, default=False
		whether to construct x axis ticks using quantiles
	figsize: (width, height), default=None
		figure size
	plot_params: dict, default=None
		values of plot parameters
	multi_flag: boolean, default=False
		whether it is a subplot of a multi-class plot
	which_class: integer, default=None
		must not be None under multi-class mode
	only_inter: boolean, default=False
		only plot the contour plot
	ncols: integer, default=None
		used under multi-class mode when only contour plots are generated
	'''

	# check pdp_interact_out
	if type(pdp_interact_out) == dict:
		try:
			if pdp_interact_out['class_0']._type != 'pdp_interact_instance':
				raise ValueError('pdp_interact_out: should be an instance of pdp_interact_obj')
		except: 
			raise ValueError('pdp_interact_out: should be an instance of pdp_interact_obj')
	else:
		try:
			if pdp_interact_out._type != 'pdp_interact_instance':
				raise ValueError('pdp_interact_out: should be an instance of pdp_interact_obj')
		except: 
			raise ValueError('pdp_interact_out: should be an instance of pdp_interact_obj')

	# check feature_names
	if type(feature_names) != list:
		raise ValueError('feature_names: should be a list')
	if len(feature_names) != 2:
		raise ValueError('feature_names: should contain 2 elements')

	# check other inputs (all boolean values)
	if type(center) != bool:
		raise ValueError('center: should be a boolean value')
	if type(plot_org_pts) != bool:
		raise ValueError('plot_org_pts: should be a boolean value')
	if type(plot_lines) != bool:
		raise ValueError('plot_lines: should be a boolean value')
	if type(cluster) != bool:
		raise ValueError('cluster: should be a boolean value')
	if type(x_quantile) != bool:
		raise ValueError('x_quantile: should be a boolean value')
	if type(multi_flag) != bool:
		raise ValueError('multi_flag: should be a boolean value')
	if type(only_inter) != bool:
		raise ValueError('only_inter: should be a boolean value')


	# check frac_to_plot
	if (type(frac_to_plot) != float) and (type(frac_to_plot) != int):
		raise ValueError('frac_to_plot: should be a float or an integer')
	if type(frac_to_plot) == float:
		if (frac_to_plot <= 0.0) or (frac_to_plot > 1.0):
			raise ValueError('frac_to_plot: should in range(0, 1) when it is a float')
	if type(pdp_interact_out) == dict:
		if (type(frac_to_plot) == int) and (frac_to_plot > pdp_interact_out['class_0'].pdp_isolate_out1.ice_lines.shape[0]):
			raise ValueError('frac_to_plot: sample size should not be larger than the population size')
	else:
		if (type(frac_to_plot) == int) and (frac_to_plot > pdp_interact_out.pdp_isolate_out1.ice_lines.shape[0]):
			raise ValueError('frac_to_plot: sample size should not be larger than the population size')

	# check n_cluster_centers
	if cluster:
		if n_cluster_centers is None:
			raise ValueError('n_cluster_centers: should not be None under clustering mode')
		if type(n_cluster_centers) != int:
			raise ValueError('n_cluster_centers: should be an integer')
		if cluster_method is not None:
			if (cluster_method != 'accurate') and (cluster_method != 'approx'):
				raise ValueError("cluster_method: should be 'accurate' or 'approx'")

	# check figsize
	if figsize is not None:
		if type(figsize) != tuple:
			raise ValueError('figsize: should be a tuple')
		if len(figsize) != 2:
			raise ValueError('figsize: should contain 2 elements: (width, height)')

	# check ncols
	if (ncols is not None) and (type(ncols) != int):
		raise ValueError('ncols: should be an integer')

	# check plot_params
	if (plot_params is not None) and (type(plot_params) != dict):
		raise ValueError('plot_params: should be a dictionary')

	# check which_class
	if multi_flag:
		if type(pdp_interact_out) != dict:
			raise ValueError('multi_flag: can only be used under multi-class mode')
		if which_class is None:
			raise ValueError('which_class: should not be None when multi_flag is on')
		if type(which_class) != int:
			raise ValueError('which_class: should be an integer')
		if which_class >= len(pdp_interact_out.keys()):
			raise ValueError('which_class: class does not exist')

	# only the contour plot
	if only_inter:
		if type(pdp_interact_out) == dict:
			if multi_flag:
				if figsize is None:
					fig=plt.figure(figsize=(8, 10))
				else:
					fig=plt.figure(figsize=figsize)
					
				gs = GridSpec(5, 1)
				gs.update(wspace=0, hspace=0)
				ax0 = plt.subplot(gs[0, :])
				_pdp_interact_plot_title(pdp_interact_out=pdp_interact_out['class_%d' %(which_class)], feature_names=feature_names, ax=ax0, figsize=figsize, 
					multi_flag=multi_flag, which_class=which_class, only_inter=only_inter, plot_params=plot_params)

				ax1 = plt.subplot(gs[1:, :])
				_pdp_contour_plot(pdp_interact_out=pdp_interact_out['class_%d' %(which_class)], feature_names=feature_names, x_quantile=x_quantile, ax=ax1, 
					fig=fig, plot_params=plot_params)
			else:
				n_classes = len(pdp_interact_out.keys())
				if ncols == None:
					ncols = 2

				if figsize is None:
					figwidth = 15
				else:
					figwidth = figsize[0]

				nrows = np.ceil(float(n_classes) / ncols)
				plt.figure(figsize=(figwidth, figwidth / 7.5))

				ax0 = plt.subplot(111)
				_pdp_interact_plot_title(pdp_interact_out=pdp_interact_out, feature_names=feature_names, ax=ax0, figsize=figsize, 
					multi_flag=multi_flag, which_class=which_class, only_inter=only_inter, plot_params=plot_params)

				plt.figure(figsize=(figwidth, (figwidth/ncols)*nrows))
				for n_class in range(n_classes):
					ax1 = plt.subplot(nrows, ncols, n_class+1)
					_pdp_contour_plot(pdp_interact_out['class_%d' %(n_class)], 
						feature_names=[feature_names[0] + ' class_%d' %(n_class), feature_names[1] +' class_%d' %(n_class)], x_quantile=x_quantile, ax=ax1, fig=None, 
						plot_params=plot_params)
		else:
			if figsize is None:
				fig=plt.figure(figsize=(8, 10))
			else:
				fig=plt.figure(figsize=figsize)
				
			gs = GridSpec(5, 1)
			gs.update(wspace=0, hspace=0)
			ax0 = plt.subplot(gs[0, :])
			_pdp_interact_plot_title(pdp_interact_out=pdp_interact_out, feature_names=feature_names, ax=ax0, figsize=figsize, 
				multi_flag=multi_flag, which_class=which_class, only_inter=only_inter, plot_params=plot_params)

			ax1 = plt.subplot(gs[1:, :])
			_pdp_contour_plot(pdp_interact_out=pdp_interact_out, feature_names=feature_names, x_quantile=x_quantile, ax=ax1, fig=fig, plot_params=plot_params)

	else:
		if type(pdp_interact_out) == dict:
			if multi_flag:
				_pdp_interact_plot(pdp_interact_out=pdp_interact_out['class_%d' %(which_class)], feature_names=feature_names, center=center, 
					plot_org_pts=plot_org_pts, plot_lines=plot_lines, frac_to_plot=frac_to_plot, cluster=cluster, n_cluster_centers=n_cluster_centers, 
					cluster_method=cluster_method, x_quantile=x_quantile, figsize=figsize, plot_params=plot_params, multi_flag=multi_flag, which_class=which_class)
			else:
				n_classes = len(pdp_interact_out.keys())
				for n_class in range(n_classes):
					_pdp_interact_plot(pdp_interact_out=pdp_interact_out['class_%d' %(n_class)], feature_names=feature_names, center=center, 
						plot_org_pts=plot_org_pts, plot_lines=plot_lines, frac_to_plot=frac_to_plot, cluster=cluster, n_cluster_centers=n_cluster_centers, 
						cluster_method=cluster_method, x_quantile=x_quantile, figsize=figsize, plot_params=plot_params, multi_flag=True, which_class=n_class)
		else:
			_pdp_interact_plot(pdp_interact_out=pdp_interact_out, feature_names=feature_names, center=center, 
				plot_org_pts=plot_org_pts, plot_lines=plot_lines, frac_to_plot=frac_to_plot, cluster=cluster, n_cluster_centers=n_cluster_centers, 
				cluster_method=cluster_method, x_quantile=x_quantile, figsize=figsize, plot_params=plot_params, multi_flag=False, which_class=None)


def _pdp_interact_plot_title(pdp_interact_out, feature_names, ax, figsize, multi_flag, which_class, only_inter, plot_params):
	
	font_family = 'Arial'
	title = 'Interaction plot between %s and %s' %(feature_names[0], feature_names[1])

	if figsize is not None:
		title_fontsize = np.max([14 * (figsize[0] / 15.0), 10])
		subtitle_fontsize = np.max([12 * (figsize[0] / 15.0), 8])
	else:
		title_fontsize=14
		subtitle_fontsize=12

	if type(pdp_interact_out) == dict:
		subtitle1 = 'Number of unique grid points of %s: %d' %(feature_names[0], len(pdp_interact_out['class_0'].feature_grids[0]))
		subtitle2 = 'Number of unique grid points of %s: %d' %(feature_names[1], len(pdp_interact_out['class_0'].feature_grids[1]))
	else:
		subtitle1 = 'Number of unique grid points of %s: %d' %(feature_names[0], len(pdp_interact_out.feature_grids[0]))
		subtitle2 = 'Number of unique grid points of %s: %d' %(feature_names[1], len(pdp_interact_out.feature_grids[1]))     

	if plot_params is not None:
		if 'pdp_inter' in plot_params.keys():
			if 'font_family' in plot_params.keys():
				font_family = plot_params['font_family']
			if 'title' in plot_params.keys():
				title = plot_params['title']
			if 'title_fontsize' in plot_params.keys():
				title_fontsize = plot_params['title_fontsize']
			if 'subtitle_fontsize' in plot_params.keys():
				subtitle_fontsize = plot_params['subtitle_fontsize']

	ax.set_facecolor('white')
	if only_inter:
		ax.text(0, 0.8, title, va="top", ha="left", fontsize=title_fontsize, fontname=font_family)
		if multi_flag:
			ax.text(0, 0.62, "For Class %d" %(which_class), va="top", ha="left", fontsize=title_fontsize, fontname=font_family)
			ax.text(0, 0.45, subtitle1, va="top", ha="left", fontsize=subtitle_fontsize, fontname=font_family, color='grey')
			ax.text(0, 0.3, subtitle2, va="top", ha="left", fontsize=subtitle_fontsize, fontname=font_family, color='grey')
		else:
			ax.text(0, 0.55, subtitle1, va="top", ha="left", fontsize=subtitle_fontsize, fontname=font_family, color='grey')
			ax.text(0, 0.4, subtitle2, va="top", ha="left", fontsize=subtitle_fontsize, fontname=font_family, color='grey')
	else:
		ax.text(0, 0.6, title, va="top", ha="left", fontsize=title_fontsize, fontname=font_family)
		if multi_flag:
			ax.text(0, 0.53, "For Class %d" %(which_class), va="top", ha="left", fontsize=title_fontsize, fontname=font_family)
			ax.text(0, 0.4, subtitle1, va="top", ha="left", fontsize=subtitle_fontsize, fontname=font_family, color='grey')
			ax.text(0, 0.35, subtitle2, va="top", ha="left", fontsize=subtitle_fontsize, fontname=font_family, color='grey')
		else:
			ax.text(0, 0.4, subtitle1, va="top", ha="left", fontsize=subtitle_fontsize, fontname=font_family, color='grey')
			ax.text(0, 0.35, subtitle2, va="top", ha="left", fontsize=subtitle_fontsize, fontname=font_family, color='grey')
	ax.axis('off')


def _pdp_interact_plot(pdp_interact_out, feature_names, center, plot_org_pts, plot_lines, frac_to_plot, cluster, n_cluster_centers, 
	cluster_method, x_quantile, figsize, plot_params, multi_flag, which_class):

	if figsize is None:
		fig=plt.figure(figsize=(15, 15))
	else:
		fig=plt.figure(figsize=figsize)

	pdp_plot_params = None
	if plot_params is not None:
		if 'pdp' in plot_params.keys():
			pdp_plot_params = plot_params['pdp']

	gs = GridSpec(2, 2)
	ax0 = plt.subplot(gs[0, 0])
	
	_pdp_interact_plot_title(pdp_interact_out=pdp_interact_out, feature_names=feature_names, ax=ax0, figsize=figsize, multi_flag=multi_flag, 
		which_class=which_class, only_inter=False, plot_params=plot_params)

	ax1 = plt.subplot(gs[0, 1])
	_pdp_plot(pdp_isolate_out=pdp_interact_out.pdp_isolate_out1, feature_name=feature_names[0], center=center, plot_org_pts=plot_org_pts, 
		plot_lines=plot_lines, frac_to_plot=frac_to_plot, cluster=cluster, n_cluster_centers=n_cluster_centers, cluster_method=cluster_method, 
		x_quantile=x_quantile, ax=ax1, plot_params=pdp_plot_params)

	ax2 = plt.subplot(gs[1, 0])
	_pdp_plot(pdp_isolate_out=pdp_interact_out.pdp_isolate_out2, feature_name=feature_names[1], center=center, plot_org_pts=plot_org_pts, 
		plot_lines=plot_lines, frac_to_plot=frac_to_plot, cluster=cluster, n_cluster_centers=n_cluster_centers, cluster_method=cluster_method, 
		x_quantile=x_quantile, ax=ax2, plot_params=pdp_plot_params)

	ax3 = plt.subplot(gs[1, 1])
	_pdp_contour_plot(pdp_interact_out=pdp_interact_out, feature_names=feature_names, x_quantile=x_quantile, ax=ax3, fig=fig, plot_params=plot_params)


def _pdp_contour_plot(pdp_interact_out, feature_names, x_quantile, ax, fig, plot_params):

	font_family='Arial'
	contour_color = 'white'
	contour_label_fontsize = 9
	contour_cmap = 'viridis'
	xticks_rotation = 0

	if plot_params is not None:
		if 'pdp_inter' in plot_params.keys():
			if 'contour_color' in plot_params['pdp_inter'].keys():
				contour_color = plot_params['pdp_inter']['contour_color']
			if 'contour_label_fontsize' in plot_params['pdp_inter'].keys():
				contour_label_fontsize = plot_params['pdp_inter']['contour_label_fontsize']
			if 'contour_cmap' in plot_params['pdp_inter'].keys():
				contour_cmap = plot_params['pdp_inter']['contour_cmap']
			if 'font_family' in plot_params['pdp_inter'].keys():
				font_family = plot_params['pdp_inter']['font_family']
			if 'xticks_rotation' in plot_params.keys():
				xticks_rotation = plot_params['xticks_rotation']

	_axis_modify(font_family, ax)

	feature_types = pdp_interact_out.feature_types
	pdp = copy.deepcopy(pdp_interact_out.pdp)

	new_feature_names = []
	for i, feature_type in enumerate(feature_types):
		if feature_type == 'onehot':
			new_col = 'onehot_%d' %(i)
			pdp[new_col] = pdp.apply(lambda x : list(x[pdp_interact_out.features[i]]).index(1), axis=1)
			new_feature_names.append(new_col)
		else:
			new_feature_names.append(pdp_interact_out.features[i])

	if (feature_types[0] == 'numeric') and x_quantile:
		pdp[new_feature_names[0]] = pdp[new_feature_names[0]].apply(lambda x : list(pdp_interact_out.feature_grids[0]).index(x))

	if (feature_types[1] == 'numeric') and x_quantile:
		pdp[new_feature_names[1]] = pdp[new_feature_names[1]].apply(lambda x : list(pdp_interact_out.feature_grids[1]).index(x))

	X, Y = np.meshgrid(pdp[new_feature_names[0]].unique(), pdp[new_feature_names[1]].unique())
	Z = []
	for i in range(X.shape[0]):
		zs = []
		for j in range(X.shape[1]):
			x = X[i, j]
			y = Y[i, j]
			z = pdp[(pdp[new_feature_names[0]] == x) & (pdp[new_feature_names[1]] == y)]['preds'].values[0]
			zs.append(z)
		Z.append(zs)
	Z = np.array(Z)

	if feature_types[0] == 'onehot':
		ax.set_xticks(range(X.shape[1]))
		ax.set_xticklabels(pdp_interact_out.pdp_isolate_out1.display_columns, rotation=xticks_rotation)
	elif feature_types[0] == 'binary':
		ax.set_xticks([0, 1])
		ax.set_xticklabels(pdp_interact_out.pdp_isolate_out1.display_columns, rotation=xticks_rotation)
	else:
		if x_quantile:
			ax.set_xticks(range(len(pdp_interact_out.feature_grids[0])))
			ax.set_xticklabels(pdp_interact_out.feature_grids[0], rotation=xticks_rotation)

	if feature_types[1] == 'onehot':    
		ax.set_yticks(range(Y.shape[0]))
		ax.set_yticklabels(pdp_interact_out.pdp_isolate_out2.display_columns)
	elif feature_types[1] == 'binary':
		ax.set_yticks([0, 1])
		ax.set_yticklabels(pdp_interact_out.pdp_isolate_out2.display_columns)
	else:
		if x_quantile:
			ax.set_yticks(range(len(pdp_interact_out.feature_grids[1])))
			ax.set_yticklabels(pdp_interact_out.feature_grids[1])

	level = np.min([X.shape[0], X.shape[1]])
	c1 = ax.contourf(X, Y, Z, N=level, origin='lower', cmap=contour_cmap)
	c2 = ax.contour(c1, levels=c1.levels, colors=contour_color, origin='lower')
	ax.clabel(c2, contour_label_fontsize=9, inline=1)

	ax.set_xlabel(feature_names[0], fontsize=10)
	ax.set_ylabel(feature_names[1], fontsize=10)
	ax.get_yaxis().tick_right()

	class ColorBarLocator(object):
		def __init__(self, pax, pad=60, width=20):
			self.pax = pax
			self.pad = pad
			self.width = width

		def __call__(self, ax, renderer):
			x, y, w, h = self.pax.get_position().bounds
			fig = self.pax.get_figure()
			inv_trans = fig.transFigure.inverted()
			pad, _ = inv_trans.transform([self.pad, 0])
			width, _ = inv_trans.transform([self.width, 0])
			return [x, y-pad, w, width]

	if fig is not None:
		cax = fig.add_axes([0,0,0,0], axes_locator=ColorBarLocator(ax))
		fig.colorbar(c1, cax = cax, orientation='horizontal')


def _axis_modify(font_family, ax):
	for tick in ax.get_xticklabels():
		tick.set_fontname(font_family)
	for tick in ax.get_yticklabels():
		tick.set_fontname(font_family)

	ax.tick_params(axis='both', which='major', labelsize=8, labelcolor='#424242', colors='#9E9E9E')
	ax.set_facecolor('white')
	ax.spines['top'].set_visible(False)
	ax.spines['bottom'].set_visible(False)
	ax.spines['right'].set_visible(False)
	ax.spines['left'].set_visible(False)

	ax.get_xaxis().tick_bottom()
	ax.get_yaxis().tick_left()
	ax.grid(True, 'major', 'x', ls='--', lw=.5, c='k', alpha=.3)
	ax.grid(True, 'major', 'y', ls='--', lw=.5, c='k', alpha=.3)
	

def actual_plot(pdp_isolate_out, feature_name, figsize=None, plot_params=None, multi_flag=False, which_class=None, ncols=None):
	'''
	pdp_isolate_out: instance of pdp_isolate_obj
		a calculated pdp_isolate_obj instance
	feature_name: string
		name of the feature, not necessary the same as the column name
	figsize: (width, height), default=None
		figure size
	plot_params: dict, default=None
		values of plot parameters
	multi_flag: boolean, default=False
		whether it is a subplot of a multiclass plot
	which_class: integer, default=None
		which class to plot
	ncols: integer, default=None
		used under multiclass mode
	'''

	# check pdp_isolate_out
	if type(pdp_isolate_out) == dict:
		try:
			if pdp_isolate_out['class_0']._type != 'pdp_isolate_instance':
				raise ValueError('pdp_isolate_out: should be an instance of pdp_isolate_obj')
		except: 
			raise ValueError('pdp_isolate_out: should be an instance of pdp_isolate_obj')
	else:
		try:
			if pdp_isolate_out._type != 'pdp_isolate_instance':
				raise ValueError('pdp_isolate_out: should be an instance of pdp_isolate_obj')
		except: 
			raise ValueError('pdp_isolate_out: should be an instance of pdp_isolate_obj')

	# check feature_name
	if type(feature_name) != str:
		raise ValueError('feature_name: should be a string')

	# check figsize
	if figsize is not None:
		if type(figsize) != tuple:
			raise ValueError('figsize: should be a tuple')
		if len(figsize) != 2:
			raise ValueError('figsize: should contain 2 elements: (width, height)')

		# check plot_params
	if (plot_params is not None) and (type(plot_params) != dict):
		raise ValueError('plot_params: should be a dictionary')

	if type(multi_flag) != bool:
		raise ValueError('multi_flag: should be a boolean value')

	# check ncols
	if (ncols is not None) and (type(ncols) != int):
		raise ValueError('ncols: should be an integer')

	# check which_class
	if multi_flag:
		if type(pdp_isolate_out) != dict:
			raise ValueError('multi_flag: can only be used under multi-class mode')
		if which_class is None:
			raise ValueError('which_class: should not be None when multi_flag is on')
		if type(which_class) != int:
			raise ValueError('which_class: should be an integer')
		if which_class >= len(pdp_isolate_out.keys()):
			raise ValueError('which_class: class does not exist')

	if figsize is None:
		figwidth = 16
	else:
		figwidth = figsize[0]

	plt.figure(figsize=(figwidth, figwidth / 6.7))
	ax1 = plt.subplot(111)

	if type(pdp_isolate_out) == dict:
		n_classes = len(pdp_isolate_out.keys())

		if multi_flag:
			_actual_plot_title(pdp_isolate_out=pdp_isolate_out['class_%d' %(which_class)], feature_name=feature_name, ax=ax1, figsize=figsize,
				multi_flag=multi_flag, which_class=which_class, plot_params=plot_params)

			_actual_plot(pdp_isolate_out=pdp_isolate_out['class_%d' %(which_class)], feature_name=feature_name, figwidth=figwidth, plot_params=plot_params, outer=None)
		else:
			_actual_plot_title(pdp_isolate_out=pdp_isolate_out['class_0'], feature_name=feature_name, ax=ax1, figsize=figsize,
				multi_flag=multi_flag, which_class=which_class, plot_params=plot_params)

			if ncols == None:
				ncols = 2
			nrows = int(np.ceil(float(n_classes) / ncols))

			plt.figure(figsize=(figwidth, (figwidth/ncols)*nrows))
			outer = GridSpec(nrows, ncols, wspace=0.2, hspace=0.2)

			for n_class in range(n_classes):
				_actual_plot(pdp_isolate_out=pdp_isolate_out['class_%d' %(n_class)], feature_name=feature_name + ' class_%d' %(n_class),
							 figwidth=figwidth, plot_params=plot_params, outer=outer[n_class])
	else:
		_actual_plot_title(pdp_isolate_out=pdp_isolate_out, feature_name=feature_name, ax=ax1, figsize=figsize,
			multi_flag=multi_flag, which_class=which_class, plot_params=plot_params)

		_actual_plot(pdp_isolate_out=pdp_isolate_out, feature_name=feature_name, figwidth=figwidth, plot_params=plot_params, outer=None)


def _actual_plot_title(pdp_isolate_out, feature_name, ax, figsize, multi_flag, which_class, plot_params):

	font_family = 'Arial'
	title = 'Actual predictions plot for %s' %(feature_name)
	subtitle = 'Each point is clustered to the closest grid point.'

	if figsize is not None:
		title_fontsize = np.max([15 * (figsize[0] / 16.0), 10])
		subtitle_fontsize = np.max([12 * (figsize[0] / 16.0), 8])
	else:
		title_fontsize=15
		subtitle_fontsize=12

	if plot_params is not None:
		if 'font_family' in plot_params.keys():
			font_family = plot_params['font_family']
		if 'title' in plot_params.keys():
			title = plot_params['title']
		if 'title_fontsize' in plot_params.keys():
			title_fontsize = plot_params['title_fontsize']
		if 'subtitle_fontsize' in plot_params.keys():
			subtitle_fontsize = plot_params['subtitle_fontsize']

	ax.set_facecolor('white')
	if multi_flag:
		ax.text(0, 0.7, title, va="top", ha="left", fontsize=title_fontsize, fontname=font_family)
		ax.text(0, 0.45, "For Class %d" %(which_class), va="top", ha="left", fontsize=subtitle_fontsize, fontname=font_family)
		ax.text(0, 0.25, subtitle, va="top", ha="left", fontsize=subtitle_fontsize, fontname=font_family, color='grey')
	else:
		ax.text(0, 0.7, title, va="top", ha="left", fontsize=title_fontsize, fontname=font_family)
		ax.text(0, 0.4, subtitle, va="top", ha="left", fontsize=subtitle_fontsize, fontname=font_family, color='grey')
	ax.axis('off')


def _actual_plot(pdp_isolate_out, feature_name, figwidth, plot_params, outer):
	try:
		import seaborn as sns
	except:
		raise RuntimeError('seaborn is necessary for the actual plot.')

	if outer is None:
		plt.figure(figsize=(figwidth, figwidth/1.6))
		gs = GridSpec(2, 1)
		ax1 = plt.subplot(gs[0])
		ax2 = plt.subplot(gs[1], sharex=ax1)
	else:
		inner = GridSpecFromSubplotSpec(2, 1, subplot_spec=outer, wspace=0, hspace=0)
		ax1 = plt.subplot(inner[0])
		ax2 = plt.subplot(inner[1], sharex=ax1)

	font_family = 'Arial'
	boxcolor = '#66C2D7'
	linecolor = '#1A4E5D'
	barcolor = '#5BB573'
	xticks_rotation = 0

	if plot_params is not None:
		if 'font_family' in plot_params.keys():
			font_family = plot_params['font_family']
		if 'boxcolor' in plot_params.keys():
			boxcolor = plot_params['boxcolor']
		if 'linecolor' in plot_params.keys():
			linecolor = plot_params['linecolor']
		if 'barcolor' in plot_params.keys():
			barcolor = plot_params['barcolor']
		if 'xticks_rotation' in plot_params.keys():
			xticks_rotation = plot_params['xticks_rotation']


	_axis_modify(font_family, ax1)
	_axis_modify(font_family, ax2)

	df = copy.deepcopy(pdp_isolate_out.ice_lines)
	actual_columns = pdp_isolate_out.actual_columns
	feature_type = pdp_isolate_out.feature_type
	feature_grids = pdp_isolate_out.feature_grids
	df = df[actual_columns + ['actual_preds']]

	if pdp_isolate_out.feature_type == 'binary':
		df['x'] = df[actual_columns[0]]
	elif pdp_isolate_out.feature_type == 'onehot':
		df['x'] = df[actual_columns].apply(lambda x : _find_onehot_actual(x), axis=1)
		df = df[df['x'].isnull()==False].reset_index(drop=True)
	else:
		df = df[(df[actual_columns[0]] >= feature_grids[0]) & (df[actual_columns[0]] <= feature_grids[-1])].reset_index(drop=True)
		df['x'] = df[actual_columns[0]].apply(lambda x : _find_closest(x, feature_grids))

	pred_median_gp = df.groupby('x', as_index=False).agg({'actual_preds': 'median'})
	pred_count_gp = df.groupby('x', as_index=False).agg({'actual_preds': 'count'})
	pred_mean_gp = df.groupby('x', as_index=False).agg({'actual_preds': 'mean'}).rename(columns={'actual_preds': 'preds_mean'})
	pred_std_gp = df.groupby('x', as_index=False).agg({'actual_preds': 'std'}).rename(columns={'actual_preds': 'preds_std'})
	pred_outlier_gp = pred_mean_gp.merge(pred_std_gp, on='x', how='left')
	pred_outlier_gp['outlier_upper'] = pred_outlier_gp['preds_mean'] + 3 * pred_outlier_gp['preds_std']
	pred_outlier_gp['outlier_lower'] = pred_outlier_gp['preds_mean'] - 3 * pred_outlier_gp['preds_std']
	
	boxwith = np.min([0.5, 0.5 / (10.0 / len(feature_grids))])
	sns.boxplot(x=df['x'], y=df['actual_preds'], width=boxwith, ax=ax1, color=boxcolor, linewidth=1, saturation=1, notch=True)
	sns.pointplot(x=pred_median_gp['x'], y=pred_median_gp['actual_preds'], ax=ax1, color=linecolor)
	ax1.set_xlabel('')
	ax1.set_ylabel('actual_preds')
	ax1.set_ylim(pred_outlier_gp['outlier_lower'].min(), pred_outlier_gp['outlier_upper'].max())

	rects = ax2.bar(pred_count_gp['x'], pred_count_gp['actual_preds'], width=boxwith, color=barcolor, alpha=0.5)
	ax2.set_xlabel(feature_name)
	ax2.set_ylabel('count')
	plt.xticks(range(len(feature_grids)), pdp_isolate_out.feature_grids, rotation=xticks_rotation)

	_autolabel(rects, ax2, barcolor)


def target_plot(df, feature, feature_name, target, num_grid_points=10, percentile_range=None, cust_grid_points=None, figsize=None, plot_params=None):
	'''
	df: pandas DataFrame
		the whole dataset to investigate, including at least the feature to investigate as well as the target values
	feature: string or list
		column to investigate (for one-hot encoding features, a list of columns should be provided)
	feature_name: string
		name of the feature, not necessary the same as the column name
	target: string or list
		the column name of the target value
		for multi-class problem, a list of one-hot encoding target values could be provided
	num_grid_points: integer, default=10
		number of grid points for numeric features
	percentile_range: (low, high), default=None
		percentile range to consider for numeric features
	cust_grid_points: list, default=None
		customized grid points
	figsize: (width, height), default=None
		figure size
	plot_params: dict, default=None
		values of plot parameters
	'''

	# check input dataset
	if type(df) != pd.core.frame.DataFrame:
		raise ValueError('df: only accept pandas DataFrame')

	# check feature
	if type(feature) == str:
		if feature not in df.columns.values:
			raise ValueError('feature does not exist: %s' %(feature))
		if sorted(list(np.unique(df[feature]))) == [0, 1]:
			feature_type='binary'
		else:
			feature_type='numeric'
	elif type(feature) == list:
		if len(feature) < 2:
			raise ValueError('one-hot encoding feature should contain more than 1 element')
		if not set(feature) < set(df.columns.values):
			raise ValueError('feature does not exist: %s' %(str(feature)))
		feature_type='onehot'
	else:
		raise ValueError('feature: please pass a string or a list (for onehot encoding feature)')

	# check feature_name
	if type(feature_name) != str:
		raise ValueError('feature_name: should be a string')

	# check num_grid_points
	if num_grid_points is not None and type(num_grid_points) != int:
		raise ValueError('num_grid_points: should be an integer')

	# check percentile_range
	if percentile_range is not None:
		if type(percentile_range) != tuple:
			raise ValueError('percentile_range: should be a tuple')
		if len(percentile_range) != 2:
			raise ValueError('percentile_range: should contain 2 elements')
		if np.max(percentile_range) > 100 or np.min(percentile_range) < 0:
			raise ValueError('percentile_range: should be between 0 and 100')

	# check cust_grid_points
	if (feature_type != 'numeric') and (cust_grid_points is not None):
		raise ValueError('only numeric feature can accept cust_grid_points')
	if (cust_grid_points is not None) and (type(cust_grid_points) != list):
		raise ValueError('cust_grid_points: should be a list')

	# check figsize
	if figsize is not None:
		if type(figsize) != tuple:
			raise ValueError('figsize: should be a tuple')
		if len(figsize) != 2:
			raise ValueError('figsize: should contain 2 elements: (width, height)')

	# check plot_params
	if (plot_params is not None) and (type(plot_params) != dict):
		raise ValueError('plot_params: should be a dictionary')

	# check target values and calculate target rate through feature grids
	if type(target) == str:
		if target not in df.columns.values:
			raise ValueError('target does not exist: %s' %(target))
		if sorted(list(np.unique(df[target]))) == [0, 1]:
			target_type='binary'
		else:
			target_type='regression'
	elif type(target) == list:
		if len(target) < 2:
			raise ValueError('multi-class target should contain more than 1 element')
		if not set(target) < set(df.columns.values):
			raise ValueError('target does not exist: %s' %(str(target)))
		for target_idx in range(len(target)):
			if sorted(list(np.unique(df[target[target_idx]]))) != [0, 1]:
				raise ValueError('multi-class targets should be one-hot encoded: %s' %(str(target[target_idx])))
		target_type='multi-class'
	else:
		raise ValueError('target: please pass a string or a list (for multi-class targets)')

	# create feature grids and bar counts
	useful_features = []
	if type(feature) == list:
		useful_features += feature
	else:
		useful_features.append(feature)
	if type(target) == list:
		useful_features += target 
	else:
		useful_features.append(target)

	bar_counts = df[useful_features].copy()
	if feature_type == 'binary':
		feature_grids = np.array([0, 1])
		bar_counts['x'] = bar_counts[feature]
	if feature_type == 'numeric':
		if cust_grid_points is None:
			feature_grids = _get_grids(df[feature], num_grid_points, percentile_range)
		else:
			feature_grids = np.array(sorted(cust_grid_points))
		bar_counts = bar_counts[(bar_counts[feature] >= feature_grids[0]) & (bar_counts[feature] <= feature_grids[-1])].reset_index(drop=True)
		bar_counts['x'] = bar_counts[feature].apply(lambda x : _find_closest(x, feature_grids))
	if feature_type == 'onehot':
		feature_grids = np.array(feature)
		bar_counts['x'] = bar_counts[feature].apply(lambda x : _find_onehot_actual(x), axis=1)
		bar_counts = bar_counts[bar_counts['x'].isnull()==False].reset_index(drop=True)

	bar_counts['fake_count'] = 1
	bar_counts_gp = bar_counts.groupby('x', as_index=False).agg({'fake_count': 'count'})

	target_lines = []
	if target_type in ['binary', 'regression']:
		target_line = bar_counts.groupby('x', as_index=False).agg({target: 'mean'})
		target_lines.append(target_line)
	else:
		for target_idx in range(len(target)):
			target_line = bar_counts.groupby('x', as_index=False).agg({target[target_idx]: 'mean'})
			target_lines.append(target_line)

	if figsize is None:
		figwidth = 16
		figheight = 7
	else:
		figwidth = figsize[0]
		figheight = figsize[1]

	font_family = 'Arial'
	linecolor = '#1A4E5D'
	barcolor = '#5BB573'
	linewidth = 2
	xticks_rotation = 0

	if plot_params is not None:
		if 'font_family' in plot_params.keys():
			font_family = plot_params['font_family']
		if 'linecolor' in plot_params.keys():
			linecolor = plot_params['linecolor']
		if 'barcolor' in plot_params.keys():
			barcolor = plot_params['barcolor']
		if 'linewidth' in plot_params.keys():
			linewidth = plot_params['linewidth']
		if 'xticks_rotation' in plot_params.keys():
			xticks_rotation = plot_params['xticks_rotation']

	plt.figure(figsize=(figwidth, figwidth / 6.7))
	ax1 = plt.subplot(111)
	_target_plot_title(feature_name=feature_name, ax=ax1, figsize=figsize, plot_params=plot_params)

	boxwith = np.min([0.5, 0.5 / (10.0 / len(feature_grids))])

	plt.figure(figsize=(figwidth, figheight))
	ax1 = plt.subplot(111)
	rects = ax1.bar(bar_counts_gp['x'], bar_counts_gp['fake_count'], width=boxwith, color=barcolor, alpha=0.5)
	ax1.set_xlabel(feature_name)
	ax1.set_ylabel('count')
	plt.xticks(range(len(feature_grids)), feature_grids, rotation=xticks_rotation)
	_autolabel(rects, ax1, barcolor)

	ax2 = ax1.twinx()
	if len(target_lines) == 1:
		target_line = target_lines[0]
		ax2.plot(target_line['x'], target_line[target], linewidth=linewidth, c=linecolor, marker='o')
		for idx in range(target_line.shape[0]):
			bbox_props = {'facecolor':linecolor, 'edgecolor':'none', 'boxstyle': "square,pad=0.5"}
			ax2.text(idx, target_line.iloc[idx][target], '%.3f'%(round(target_line.iloc[idx][target], 3)), 
				ha="center", va="bottom", size=10, bbox=bbox_props, color='#ffffff', weight='bold')
	else:
		linecolors = plt.get_cmap('tab10')(range(10))
		for target_idx in range(len(target)):
			linecolor = linecolors[target_idx]
			target_line = target_lines[target_idx]
			ax2.plot(target_line['x'], target_line[target[target_idx]], linewidth=linewidth, c=linecolor, marker='o', label=target[target_idx])
			for idx in range(target_line.shape[0]):
				bbox_props = {'facecolor':linecolor, 'edgecolor':'none', 'boxstyle': "square,pad=0.5"}
				ax2.text(idx, target_line.iloc[idx][target[target_idx]], '%.3f'%(round(target_line.iloc[idx][target[target_idx]], 3)), 
					ha="center", va="top", size=10, bbox=bbox_props, color='#ffffff', weight='bold')
			plt.legend()

	_axis_modify(font_family, ax2)
	ax2.get_yaxis().tick_right()
	ax2.grid(False)
	ax2.set_ylabel('target_avg')		

	_axis_modify(font_family, ax1)

	
def _autolabel(rects, ax, barcolor):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        bbox_props = {'facecolor': 'white', 'edgecolor':barcolor, 'boxstyle': "square,pad=0.5"}
        ax.text(rect.get_x() + rect.get_width()/2., height,
                '%d' % int(height),
                ha='center', va='center', bbox=bbox_props, color=barcolor, weight='bold')

def _target_plot_title(feature_name, ax, figsize, plot_params):

	font_family = 'Arial'
	title = 'Real target plot for %s' %(feature_name)
	subtitle = 'Each point is clustered to the closest grid point.'

	if figsize is not None:
		title_fontsize = np.max([15 * (figsize[0] / 16.0), 10])
		subtitle_fontsize = np.max([12 * (figsize[0] / 16.0), 8])
	else:
		title_fontsize=15
		subtitle_fontsize=12

	if plot_params is not None:
		if 'font_family' in plot_params.keys():
			font_family = plot_params['font_family']
		if 'title' in plot_params.keys():
			title = plot_params['title']
		if 'title_fontsize' in plot_params.keys():
			title_fontsize = plot_params['title_fontsize']
		if 'subtitle_fontsize' in plot_params.keys():
			subtitle_fontsize = plot_params['subtitle_fontsize']

	ax.set_facecolor('white')
	ax.text(0, 0.7, title, va="top", ha="left", fontsize=title_fontsize, fontname=font_family)
	ax.text(0, 0.4, subtitle, va="top", ha="left", fontsize=subtitle_fontsize, fontname=font_family, color='grey')
	ax.axis('off')


