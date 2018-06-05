
## Documentation
- version v0.1

#### pdpbox.pdp.pdp_isolate
```python
def pdp_isolate(model, train_X, feature, num_grid_points=10, percentile_range=None, 
		cust_grid_points=None, predict_kwds={}):
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
```


#### pdpbox.pdp.pdp_interact
```python
def pdp_interact(model, train_X, features, num_grid_points=[10, 10], percentile_ranges=[None, None], 
		 cust_grid_points=[None, None], predict_kwds={}):
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
```


#### pdpbox.pdp.pdp_plot
```python
def pdp_plot(pdp_isolate_out, feature_name, center=True, plot_org_pts=False, plot_lines=False, frac_to_plot=1, 
	     cluster=False, n_cluster_centers=None, cluster_method=None, x_quantile=False, figsize=None, 
	     ncols=None, plot_params=None, multi_flag=False, which_class=None):
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
```


#### pdpbox.pdp.pdp_interact_plot
```python
def pdp_interact_plot(pdp_interact_out, feature_names, center=True, plot_org_pts=False, plot_lines=False, 
		      frac_to_plot=1., cluster=False, n_cluster_centers=None, cluster_method=None, 
		      x_quantile=False, figsize=None, plot_params=None, multi_flag=False, 
		      which_class=None, only_inter=False, ncols=None):
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
```


#### pdpbox.pdp.actual_plot
```python
def actual_plot(pdp_isolate_out, feature_name, figsize=None, plot_params=None, multi_flag=False, 
		which_class=None, ncols=None):
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
```


#### pdpbox.pdp.target_plot
```python
def target_plot(df, feature, feature_name, target, num_grid_points=10, percentile_range=None, cust_grid_points=None, 
		figsize=None, plot_params=None):
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
```
