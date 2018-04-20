
import pdp_calc_utils
import pdp_plot_utils

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from joblib import Parallel, delayed
import psutil

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


def pdp_isolate(model, train_X, feature, num_grid_points=10, grid_type='percentile',
                percentile_range=None, grid_range=None, cust_grid_points=None, memory_limit=0.5, n_jobs=1,
                predict_kwds={}, data_transformer=None):
    """
    Calculate PDP isolation plot

    :param model: a fitted sklearn model
    :param train_X: pandas DataFrame
        data set on which the model is trained
    :param feature: string or list
        column to investigate (for one-hot encoding features, a list of columns should be provided)
    :param num_grid_points: integer, default=10
        number of grid points for numeric features
    :param grid_type, default='percentile'
        can be 'percentile' or 'equal'
    :param percentile_range: (low, high), default=None
        percentile range to consider for numeric features
    :param grid_range: (low, high), default=None
        value range to consider for numeric features
    :param cust_grid_points: list, default=None
        customized grid points
    :param memory_limit: float, default=0.5
        fraction of RAM can be used to do the calculation
    :param n_jobs: integer, default=1
        the number of jobs to run in parallel
    :param predict_kwds: dict, default={}
        keywords to be passed to the model's predict function
    :param data_transformer: function
        function to transform the data set as some features changing values

    :return: instance of pdp_isolate_obj
    """

    # check model
    try:
        n_classes = len(model.classes_)
        classifier = True
        predict = model.predict_proba
    except:
        n_classes = 0
        classifier = False
        predict = model.predict

    # check input data set
    if type(train_X) != pd.core.frame.DataFrame:
        raise ValueError('train_X: only accept pandas DataFrame')

    # check percentile_range
    if percentile_range is not None:
        if type(percentile_range) != tuple:
            raise ValueError('percentile_range: should be a tuple')
        if len(percentile_range) != 2:
            raise ValueError('percentile_range: should contain 2 elements')
        if np.max(percentile_range) > 100 or np.min(percentile_range) < 0:
            raise ValueError('percentile_range: should be between 0 and 100')

    # check grid_range
    if grid_range is not None:
        if type(grid_range) != tuple:
            raise ValueError('grid_range: should be a tuple')
        if len(grid_range) != 2:
            raise ValueError('grid_range: should contain 2 elements')

    # copy training data set and get the model features
    # it's extremely important to keep the original feature order
    _train_X = train_X.copy()
    model_features = _train_X.columns.values

    # check feature
    if type(feature) == str:
        if feature not in _train_X.columns.values:
            raise ValueError('feature does not exist: %s' % feature)
        if sorted(list(np.unique(_train_X[feature]))) == [0, 1]:
            feature_type = 'binary'
        else:
            feature_type = 'numeric'
    elif type(feature) == list:
        if len(feature) < 2:
            raise ValueError('one-hot encoding feature should contain more than 1 element')
        if not set(feature) < set(_train_X.columns.values):
            raise ValueError('feature does not exist: %s' % (str(feature)))
        feature_type = 'onehot'
    else:
        raise ValueError('feature: please pass a string or a list (for onehot encoding feature)')

    # check cust_grid_points
    if (feature_type != 'numeric') and (cust_grid_points is not None):
        raise ValueError('only numeric feature can accept cust_grid_points')

    # check memory_limit
    if memory_limit <= 0 or memory_limit >= 1:
        raise ValueError('memory_limit: should be (0, 1)')

    # create feature grids
    if feature_type == 'binary':
        feature_grids = np.array([0, 1])
        display_columns = ['%s_0' % feature, '%s_1' % feature]
    elif feature_type == 'onehot':
        feature_grids = np.array(feature)
        display_columns = feature
    else:
        # calculate grid points for numeric features
        if cust_grid_points is None:
            # check grid_range
            if grid_range is not None:
                if np.min(grid_range) < np.min(_train_X[feature].values) \
                        or np.max(grid_range) > np.max(_train_X[feature].values):
                    warnings.warn('grid_range: out of bound.')
            feature_grids = pdp_calc_utils._get_grids(_train_X[feature].values, num_grid_points, grid_type,
                                                      percentile_range, grid_range)
        else:
            feature_grids = np.array(sorted(cust_grid_points))
        display_columns = feature_grids

    # get the actual prediction and actual values
    actual_columns = []
    if feature_type == 'onehot':
        for feat in feature:
            actual_column = 'actual_%s' % feat
            _train_X[actual_column] = _train_X[feat]
            actual_columns.append(actual_column)
    else:
        actual_columns.append('actual_%s' % feature)
        _train_X[actual_columns[0]] = _train_X[feature]

    actual_preds = predict(_train_X[model_features], **predict_kwds)
    if n_classes > 2:
        for n_class in range(n_classes):
            _train_X['actual_preds_class_%d' % n_class] = actual_preds[:, n_class]
    else:
        if classifier:
            _train_X['actual_preds'] = actual_preds[:, 1]
        else:
            _train_X['actual_preds'] = actual_preds

    # new from here
    # calculate memory usage
    unit_memory = _train_X.memory_usage(deep=True).sum()
    free_memory = psutil.virtual_memory()[1] * memory_limit
    num_units = int(np.floor(free_memory / unit_memory))

    true_n_jobs = np.min([num_units, n_jobs])

    grid_results = Parallel(n_jobs=true_n_jobs)(
        delayed(pdp_calc_utils._calc_ice_lines)(_train_X.copy(), model, classifier, model_features, n_classes,
                                                feature, feature_type, feature_grids[i], display_columns[i],
                                                predict_kwds, data_transformer) for i in range(len(feature_grids)))

    if n_classes > 2:
        ice_lines = []
        for n_class in range(n_classes):
            ice_line_n_class = pd.concat([grid_result[n_class] for grid_result in grid_results], axis=1)
            ice_line_n_class = pd.concat([ice_line_n_class, _train_X[actual_columns],
                                          _train_X['actual_preds_class_%d' % n_class]], axis=1)
            ice_line_n_class = ice_line_n_class.rename(columns={'actual_preds_class_%d' % n_class: 'actual_preds'})
            ice_lines.append(ice_line_n_class)
    else:
        ice_lines = pd.concat(grid_results, axis=1)
        ice_lines = pd.concat([ice_lines, _train_X[actual_columns], _train_X['actual_preds']], axis=1)

    # check whether the results is for multi-class
    # combine the final results
    if n_classes > 2:
        pdp_isolate_out = {}
        for n_class in range(n_classes):
            pdp = ice_lines[n_class][display_columns].mean().values
            pdp_isolate_out['class_%d' % n_class] = \
                pdp_isolate_obj(n_classes=n_classes, classifier=classifier, model_features=model_features,
                                feature=feature, feature_type=feature_type, feature_grids=feature_grids,
                                actual_columns=actual_columns, display_columns=display_columns, ice_lines=ice_lines[n_class], pdp=pdp)
    else:
        pdp = ice_lines[display_columns].mean().values
        pdp_isolate_out = pdp_isolate_obj(n_classes=n_classes, classifier=classifier, model_features=model_features,
                                          feature=feature, feature_type=feature_type, feature_grids=feature_grids,
                                          actual_columns=actual_columns, display_columns=display_columns,
                                          ice_lines=ice_lines, pdp=pdp)

    return pdp_isolate_out


def pdp_plot(pdp_isolate_out, feature_name, center=True, plot_org_pts=False, plot_lines=False, frac_to_plot=1,
             cluster=False, n_cluster_centers=None, cluster_method=None, x_quantile=False, figsize=None,
             ncols=None, plot_params=None, multi_flag=False, which_class=None):
    """
    Plot partial dependent plot

    :param pdp_isolate_out: instance of pdp_isolate_obj
        a calculated pdp_isolate_obj instance
    :param feature_name: string
        name of the feature, not necessary the same as the column name
    :param center: boolean, default=True
        whether to center the plot
    :param plot_org_pts: boolean, default=False
        whether to plot out the original points
    :param plot_lines: boolean, default=False
        whether to plot out the individual lines
    :param frac_to_plot: float or integer, default=1
        how many points or lines to plot, can be a integer or a float
    :param cluster: boolean, default=False
        whether to cluster the individual lines and only plot out the cluster centers
    :param n_cluster_centers: integer, default=None
        number of cluster centers
    :param cluster_method: string, default=None
        cluster method to use, default is KMeans, if 'approx' is passed, MiniBatchKMeans is used
    :param x_quantile: boolean, default=False
        whether to construct x axis ticks using quantiles
    :param figsize: (width, height), default=None
        figure size
    :param ncols: integer, default=None
        used under multiclass mode
    :param plot_params: dict, default=None
        values of plot parameters
    :param multi_flag: boolean, default=False
        whether it is a subplot of a multi-classes plot
    :param which_class: integer, default=None
        which class to plot
    """

    # check frac_to_plot
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
        if cluster_method is not None:
            if (cluster_method != 'accurate') and (cluster_method != 'approx'):
                raise ValueError("cluster_method: should be 'accurate' or 'approx'")

    # check which_class
    if multi_flag and which_class >= len(pdp_isolate_out.keys()):
        raise ValueError('which_class: class does not exist')

    if type(pdp_isolate_out) == dict and not multi_flag:
        # for multi-classes
        n_classes = len(pdp_isolate_out.keys())
        if ncols is None:
            ncols = 2
        nrows = np.ceil(float(n_classes) / ncols)

        if figsize is None:
            figwidth = 16
        else:
            figwidth = figsize[0]

        # draw the title
        plt.figure(figsize=(figwidth, figwidth / 8.))
        ax1 = plt.subplot(111)
        n_grids = len(pdp_isolate_out['class_0'].feature_grids)
        pdp_plot_utils._pdp_plot_title(n_grids=n_grids, feature_name=feature_name, ax=ax1,
                                       multi_flag=multi_flag, which_class=which_class, plot_params=plot_params)

        # PDP plots for each class
        plt.figure(figsize=(figwidth, (figwidth / ncols) * nrows))
        for n_class in range(n_classes):
            ax2 = plt.subplot(nrows, ncols, n_class + 1)
            pdp_plot_utils._pdp_plot(pdp_isolate_out=pdp_isolate_out['class_%d' % (n_class)],
                                     feature_name=feature_name + ' class_%d' % (n_class), center=center,
                                     plot_org_pts=plot_org_pts, plot_lines=plot_lines, frac_to_plot=frac_to_plot,
                                     cluster=cluster, n_cluster_centers=n_cluster_centers, cluster_method=cluster_method,
                                     x_quantile=x_quantile, ax=ax2, plot_params=plot_params)
    else:
        if figsize is None:
            plt.figure(figsize=(16, 9))
        else:
            plt.figure(figsize=figsize)

        gs = GridSpec(5, 1)
        ax1 = plt.subplot(gs[0, :])

        if multi_flag:
            n_grids = len(pdp_isolate_out['class_0'].feature_grids)
            _pdp_isolate_out = pdp_isolate_out['class_%d' % which_class]
        else:
            n_grids = len(pdp_isolate_out.feature_grids)
            _pdp_isolate_out = pdp_isolate_out
        pdp_plot_utils._pdp_plot_title(n_grids=n_grids, feature_name=feature_name, ax=ax1,
                                       multi_flag=False, which_class=None, plot_params=plot_params)

        ax2 = plt.subplot(gs[1:, :])
        pdp_plot_utils._pdp_plot(pdp_isolate_out=_pdp_isolate_out, feature_name=feature_name, center=center,
                                 plot_org_pts=plot_org_pts, plot_lines=plot_lines, frac_to_plot=frac_to_plot,
                                 cluster=cluster, n_cluster_centers=n_cluster_centers, cluster_method=cluster_method,
                                 x_quantile=x_quantile, ax=ax2, plot_params=plot_params)


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


def pdp_interact(model, train_X, features, num_grid_points=[10, 10], grid_types=['percentile', 'percentile'],
                 percentile_ranges=[None, None], grid_ranges=[None, None], cust_grid_points=[None, None],
                 memory_limit=0.5, n_jobs=1, predict_kwds={}, data_transformer=None):
    """
    Calculate PDP interaction plot

    :param model: sklearn model
        a fitted model
    :param train_X: pandas DataFrame
        dataset on which the model is trained
    :param features: list
        a list containing two features
    :param num_grid_points: list, default=[10, 10]
        a list of number of grid points for each feature
    :param grid_types: list, default=['percentile', 'percentile']
        a list of grid types for each feature
    :param percentile_ranges: list, default=[None, None]
        a list of percentile range to consider for each feature
    :param grid_ranges: list, default=[None, None]
        a list of grid range to consider for each feature
    :param cust_grid_points: list, default=None
        a list of customized grid points
    :param memory_limit: float, default=0.5
        fraction of RAM can be used to do the calculation
    :param n_jobs: integer, default=1
        the number of jobs to run in parallel
    :param predict_kwds: dict, default={}
        keywords to be passed to the model's predict function
    :param data_transformer: function
        function to transform the data set as some features changing values

    :return:
        instance of pdp_interact_obj
    """

    # check input dataset
    if type(train_X) != pd.core.frame.DataFrame:
        raise ValueError('train_X: only accept pandas DataFrame')
    _train_X = train_X.copy()

    # calculate pdp_isolate for each feature
    pdp_isolate_out1 = pdp_isolate(model, _train_X, features[0], num_grid_points=num_grid_points[0],
                                   grid_type=grid_types[0], percentile_range=percentile_ranges[0],
                                   grid_range=grid_ranges[0], cust_grid_points=cust_grid_points[0],
                                   memory_limit=memory_limit, n_jobs=n_jobs, predict_kwds=predict_kwds,
                                   data_transformer=data_transformer)
    pdp_isolate_out2 = pdp_isolate(model, _train_X, features[1], num_grid_points=num_grid_points[1],
                                   grid_type=grid_types[1], percentile_range=percentile_ranges[1],
                                   grid_range=grid_ranges[1], cust_grid_points=cust_grid_points[1],
                                   memory_limit=memory_limit, n_jobs=n_jobs, predict_kwds=predict_kwds,
                                   data_transformer=data_transformer)

    # whether it is for multi-classes
    if type(pdp_isolate_out1) == dict:
        model_features = pdp_isolate_out1['class_0'].model_features
        feature_grids = [pdp_isolate_out1['class_0'].feature_grids, pdp_isolate_out2['class_0'].feature_grids]
        feature_types = [pdp_isolate_out1['class_0'].feature_type, pdp_isolate_out2['class_0'].feature_type]
        classifier = pdp_isolate_out1['class_0'].classifier
        n_classes = pdp_isolate_out1['class_0'].n_classes
    else:
        model_features = pdp_isolate_out1.model_features
        feature_grids = [pdp_isolate_out1.feature_grids, pdp_isolate_out2.feature_grids]
        feature_types = [pdp_isolate_out1.feature_type, pdp_isolate_out2.feature_type]
        classifier = pdp_isolate_out1.classifier
        n_classes = pdp_isolate_out1.n_classes

    # make features into list
    feature_list = []
    for feat in features:
        if type(feat) == list:
            feature_list += feat
        else:
            feature_list.append(feat)

    # new from here
    # calculate memory usage
    unit_memory = _train_X.memory_usage(deep=True).sum()
    free_memory = psutil.virtual_memory()[1] * memory_limit
    num_units = int(np.floor(free_memory / unit_memory))

    true_n_jobs = np.min([num_units, n_jobs])

    # create grid combination
    grids1, grids2 = feature_grids
    if feature_types[0] == 'onehot':
        grids1 = range(len(grids1))
    if feature_types[1] == 'onehot':
        grids2 = range(len(grids2))

    grids_combo_temp = np.matrix(np.array(np.meshgrid(grids1, grids2)).T.reshape(-1, 2))
    grids_combo1, grids_combo2 = grids_combo_temp[:, 0], grids_combo_temp[:, 1]
    if feature_types[0] == 'onehot':
        grids_combo1_temp = np.array(grids_combo1.T, dtype=np.int64)[0]
        grids_combo1 = np.zeros((len(grids_combo1), len(grids1)), dtype=int)
        grids_combo1[range(len(grids_combo1)), grids_combo1_temp] = 1
    if feature_types[1] == 'onehot':
        grids_combo2_temp = np.array(grids_combo2.T, dtype=np.int64)[0]
        grids_combo2 = np.zeros((len(grids_combo2), len(grids2)), dtype=int)
        grids_combo2[range(len(grids_combo2)), grids_combo2_temp] = 1

    grids_combo = np.array(np.concatenate((grids_combo1, grids_combo2), axis=1))

    grid_results = Parallel(n_jobs=true_n_jobs)(
        delayed(pdp_calc_utils._calc_ice_lines_inter)(_train_X.copy(), model, classifier, model_features, n_classes,
                                                      feature_list, grids_combo[i], predict_kwds, data_transformer) for i in range(len(grids_combo)))

    ice_lines = pd.concat(grid_results, axis=0).reset_index(drop=True)
    pdp = ice_lines.groupby(feature_list, as_index=False).mean()

    if n_classes > 2:
        pdp_interact_out = {}
        for n_class in range(n_classes):
            pdp_interact_out['class_%d' % n_class] = \
                pdp_interact_obj(n_classes=n_classes, classifier=classifier, model_features=model_features,
                                 features=features, feature_types=feature_types, feature_grids=feature_grids,
                                 pdp_isolate_out1=pdp_isolate_out1['class_%d' % n_class],
                                 pdp_isolate_out2=pdp_isolate_out2['class_%d' % n_class],
                                 pdp=pdp[feature_list + ['class_%d_preds' % n_class]].rename(
                                     columns={'class_%d_preds' % n_class: 'preds'}))
    else:
        pdp_interact_out = pdp_interact_obj(n_classes=n_classes, classifier=classifier, model_features=model_features,
                                            features=features, feature_types=feature_types, feature_grids=feature_grids,
                                            pdp_isolate_out1=pdp_isolate_out1, pdp_isolate_out2=pdp_isolate_out2, pdp=pdp)

    return pdp_interact_out


def pdp_interact_plot(pdp_interact_out, feature_names, center=True, plot_org_pts=False, plot_lines=False,
                      frac_to_plot=1., cluster=False, n_cluster_centers=None, cluster_method=None, x_quantile=False,
                      figsize=None, plot_params=None, multi_flag=False, which_class=None, only_inter=False, ncols=None):
    """
    PDP two variables interaction plot

    :param pdp_interact_out: pdp_interact_obj
        a calculated pdp_interact_obj
    :param feature_names: list
        a list of feature names
    :param center: boolean, default=True
        whether to center the individual pdp plot
    :param plot_org_pts: boolean, default=False
        whether to plot out the original points for the individual pdp plot
    :param plot_lines: boolean, default=False
        whether to plot out the individual lines for the individual pdp plot
    :param frac_to_plot: integer or float, default=1
        how many lines or points to plot for the individual pdp plot
    :param cluster: boolean, default=False
        whether to cluster the the individual pdp plot
    :param n_cluster_centers: integer, default=None
        number of cluster centers for the individual pdp plot under clustering mode
    :param cluster_method: string, default=None
        clustering method to use
    :param x_quantile: boolean, default=False
        whether to construct x axis ticks using quantiles
    :param figsize: (width, height), default=None
        figure size
    :param plot_params: dict, default=None
        values of plot parameters
    :param multi_flag: boolean, default=False
        whether it is a subplot of a multi-class plot
    :param which_class: integer, default=None
        must not be None under multi-class mode
    :param only_inter: boolean, default=False
        only plot the contour plot
    :param ncols: integer, default=None
        used under multi-class mode when only contour plots are generated
    """

    # check frac_to_plot
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
        if cluster_method is not None:
            if (cluster_method != 'accurate') and (cluster_method != 'approx'):
                raise ValueError("cluster_method: should be 'accurate' or 'approx'")

    # check which_class
    if multi_flag and which_class >= len(pdp_interact_out.keys()):
        raise ValueError('which_class: class does not exist')

    # only the contour plot
    if only_inter:
        if type(pdp_interact_out) == dict and not multi_flag:
            n_classes = len(pdp_interact_out.keys())

            if ncols is None:
                ncols = 2
            nrows = np.ceil(float(n_classes) / ncols)

            if figsize is None:
                figwidth = 15
            else:
                figwidth = figsize[0]

            # draw graph title
            plt.figure(figsize=(figwidth, figwidth / 7.5))
            ax0 = plt.subplot(111)
            pdp_plot_utils._pdp_interact_plot_title(pdp_interact_out=pdp_interact_out, feature_names=feature_names,
                                                    ax=ax0, multi_flag=multi_flag, which_class=which_class,
                                                    only_inter=only_inter, plot_params=plot_params)

            plt.figure(figsize=(figwidth, (figwidth / ncols) * nrows))
            for n_class in range(n_classes):
                ax1 = plt.subplot(nrows, ncols, n_class + 1)
                pdp_plot_utils._pdp_contour_plot(pdp_interact_out['class_%d' % n_class],
                                                 feature_names=[feature_names[0] + ' class_%d' % n_class,
                                                                feature_names[1] + ' class_%d' % n_class],
                                                 x_quantile=x_quantile, ax=ax1, fig=None, plot_params=plot_params)
        else:
            if figsize is None:
                fig = plt.figure(figsize=(8, 10))
            else:
                fig = plt.figure(figsize=figsize)

            gs = GridSpec(5, 1)
            gs.update(wspace=0, hspace=0)
            ax0 = plt.subplot(gs[0, :])

            if multi_flag:
                _pdp_interact_out = pdp_interact_out['class_%d' % which_class]
            else:
                _pdp_interact_out = pdp_interact_out

            pdp_plot_utils._pdp_interact_plot_title(pdp_interact_out=_pdp_interact_out, feature_names=feature_names,
                                                    ax=ax0, multi_flag=multi_flag, which_class=which_class,
                                                    only_inter=only_inter, plot_params=plot_params)

            ax1 = plt.subplot(gs[1:, :])
            pdp_plot_utils._pdp_contour_plot(pdp_interact_out=_pdp_interact_out, feature_names=feature_names,
                                             x_quantile=x_quantile, ax=ax1, fig=fig, plot_params=plot_params)
    else:
        if type(pdp_interact_out) == dict and not multi_flag:
            n_classes = len(pdp_interact_out.keys())
            for n_class in range(n_classes):
                pdp_plot_utils._pdp_interact_plot(pdp_interact_out=pdp_interact_out['class_%d' % n_class],
                                                  feature_names=feature_names, center=center, plot_org_pts=plot_org_pts,
                                                  plot_lines=plot_lines, frac_to_plot=frac_to_plot, cluster=cluster,
                                                  n_cluster_centers=n_cluster_centers, cluster_method=cluster_method,
                                                  x_quantile=x_quantile, figsize=figsize, plot_params=plot_params,
                                                  multi_flag=True, which_class=n_class)
        else:
            if multi_flag:
                _pdp_interact_out = pdp_interact_out['class_%d' % which_class]
            else:
                _pdp_interact_out = pdp_interact_out

            pdp_plot_utils._pdp_interact_plot(pdp_interact_out=_pdp_interact_out, feature_names=feature_names, center=center,
                                              plot_org_pts=plot_org_pts, plot_lines=plot_lines, frac_to_plot=frac_to_plot,
                                              cluster=cluster, n_cluster_centers=n_cluster_centers, cluster_method=cluster_method,
                                              x_quantile=x_quantile, figsize=figsize, plot_params=plot_params,
                                              multi_flag=False, which_class=None)

