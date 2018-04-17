import pandas as pd
import numpy as np

from sklearn.externals.joblib import Parallel, delayed

import warnings
warnings.filterwarnings('ignore')

import psutil


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


def _get_grids(x, num_grid_points, grid_type, percentile_range, grid_range):
    """
    Calculate grid points for numeric features

    :param x: array of feature values
    :param num_grid_points: number of grid points to calculate
    :param grid_type, default='percentile'
        can be 'percentile' or 'equal'
    :param percentile_range: (low, high), default=None
        percentile range to consider for numeric features
    :param grid_range: (low, high), default=None
        value range to consider for numeric features

    :return:
        array of calculated grid points
    """

    if grid_type == 'percentile':
        if num_grid_points >= np.unique(x).size:
            grids = np.unique(x)
        else:
            # grid points are calculated based on percentile in unique level
            # thus the final number of grid points might be smaller than num_grid_points
            if percentile_range is not None:
                grids = np.unique(
                    np.percentile(x, np.linspace(np.min(percentile_range), np.max(percentile_range), num_grid_points)))
            else:
                grids = np.unique(np.percentile(x, np.linspace(0, 100, num_grid_points)))
    else:
        if grid_range is not None:
            grids = np.linspace(np.min(grid_range), np.max(grid_range), num_grid_points)
        else:
            grids = np.linspace(np.min(x), np.max(x), num_grid_points)

    return np.array([round(val, 2) for val in grids])


def _calc_ice_lines(data, model, classifier, model_features, n_classes, feature, feature_type,
                    feature_grid, display_column, predict_kwds, data_transformer):
    if feature_type == 'onehot':
        other_grids = list(set(feature) - set([feature_grid]))
        data[feature_grid] = 1
        for grid in other_grids:
            data[grid] = 0
    else:
        data[feature] = feature_grid

    if data_transformer is not None:
        data = data_transformer(data)

    if classifier:
        predict = model.predict_proba
    else:
        predict = model.predict

    # get predictions for this chunk
    preds = predict(data[model_features], **predict_kwds)

    if n_classes > 2:
        grid_results = []
        for n_class in range(n_classes):
            grid_result = pd.DataFrame(preds[:, n_class], columns=[display_column])
            grid_results.append(grid_result)
    else:
        if classifier:
            grid_results = pd.DataFrame(preds[:, 1], columns=[display_column])
        else:
            grid_results = pd.DataFrame(preds, columns=[display_column])

    return grid_results


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
            feature_grids = _get_grids(_train_X[feature].values, num_grid_points, grid_type,
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

    true_n_jobs = np.min([num_units-2, n_jobs])

    grid_results = Parallel(n_jobs=true_n_jobs)(
        delayed(_calc_ice_lines)(_train_X.copy(), model, classifier, model_features, n_classes, feature,
                                 feature_type, feature_grids[i], display_columns[i], predict_kwds, data_transformer)
        for i in range(len(feature_grids)))

    if n_classes > 2:
        ice_lines = []
        for n_class in range(n_classes):
            ice_line_n_class = pd.concat([grid_result[n_class] for grid_result in grid_results], axis=1)
            ice_line_n_class = pd.concat([ice_line_n_class, _train_X[actual_columns],
                                          _train_X['actual_preds_class_%d' % n_class]], axis=1)
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


def _make_ice_data_inter(data, features, feature_types, feature_grids):
    """
    Prepare data for calculating interaction ice lines

    :param data: chunk of data to calculate on
    :param features: feature column names
    :param feature_types: feature types
    :param feature_grids: feature grids

    :return:
        the extended data chunk (extended by feature grids)
    """

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


def _calc_ice_lines_inter(data_chunk, model, classifier, model_features, n_classes, features, feature_types,
                          feature_grids, feature_list, predict_kwds):
    """
    Calculate interaction ice lines

    :param data_chunk: chunk of data to calculate on
    :param model: the fitted model
    :param classifier: whether the model is a classifier
    :param model_features: features used by the model
    :param n_classes: number of classes if it is a classifier
    :param features: feature column names
    :param feature_types: feature types
    :param feature_grids: feature grids
    :param feature_list: list of features
    :param predict_kwds: other parameters pass to the predictor

    :return:
        a dataframe containing changing feature values and corresponding predictions
    """

    ice_chunk = _make_ice_data_inter(data_chunk[model_features], features, feature_types, feature_grids)

    if classifier:
        predict = model.predict_proba
    else:
        predict = model.predict

    preds = predict(ice_chunk[model_features], **predict_kwds)
    result_chunk = ice_chunk[feature_list].copy()

    if n_classes > 2:
        for n_class in range(n_classes):
            result_chunk['class_%d_preds' % (n_class)] = preds[:, n_class]
    else:
        if classifier:
            result_chunk['preds'] = preds[:, 1]
        else:
            result_chunk['preds'] = preds

    return result_chunk


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
                                   grid_range=grid_ranges[0], cust_grid_points=cust_grid_points[0], n_jobs=n_jobs)
    pdp_isolate_out2 = pdp_isolate(model, _train_X, features[1], num_grid_points=num_grid_points[1],
                                   grid_type=grid_types[1], percentile_range=percentile_ranges[1],
                                   grid_range=grid_ranges[1], cust_grid_points=cust_grid_points[1], n_jobs=n_jobs)

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

    true_n_jobs = np.min([num_units - 2, n_jobs])

    

    grid_results = Parallel(n_jobs=true_n_jobs)(
        delayed(_calc_ice_lines_inter)(_train_X.copy(), model, classifier, model_features, n_classes, features,
                                      feature_types, feature_grids[i], display_columns[i], predict_kwds,
                                      data_transformer)
        for i in range(len(feature_grids)))

    # do prediction chunk by chunk to save memory usage
    grids_size = len(feature_grids[0]) * len(feature_grids[1])
    data_chunk_size = int(_train_X.shape[0] / grids_size)
    if data_chunk_size == 0:
        data_chunk_size = _train_X.shape[0]

    # parallel get ice lines
    ice_params = {
        'model': model,
        'classifier': classifier,
        'model_features': model_features,
        'n_classes': n_classes,
        'features': features,
        'feature_types': feature_types,
        'feature_grids': feature_grids,
        'feature_list': feature_list,
        'predict_kwds': predict_kwds
    }
    ice_chunk_results = Parallel(n_jobs=n_jobs)(delayed(pdp_calc_utils._calc_ice_lines_inter)
                                                (_train_X[i: (i + data_chunk_size)].reset_index(drop=True),
                                                 **ice_params) for i in range(0, len(_train_X), data_chunk_size))

    ice_lines = pd.concat(ice_chunk_results, axis=0).reset_index(drop=True)

    pdp = ice_lines.groupby(feature_list, as_index=False).mean()

    if n_classes > 2:
        pdp_interact_out = {}
        for n_class in range(n_classes):
            pdp_interact_out['class_%d' % n_class] = \
                pdp_interact_obj(n_classes=n_classes, classifier=classifier, model_features=model_features,
                                 features=features, feature_types=feature_types, feature_grids=feature_grids,
                                 pdp_isolate_out1=pdp_isolate_out1['class_%d' % n_class],
                                 pdp_isolate_out2=pdp_isolate_out2['class_%d' % n_class],
                                 pdp=pdp[feature_list + ['class_%d_preds' % n_class]].rename(columns={'class_%d_preds' % n_class: 'preds'}))
    else:
        pdp_interact_out = pdp_interact_obj(n_classes=n_classes, classifier=classifier, model_features=model_features,
                                            features=features, feature_types=feature_types, feature_grids=feature_grids,
                                            pdp_isolate_out1=pdp_isolate_out1, pdp_isolate_out2=pdp_isolate_out2, pdp=pdp)

    return pdp_interact_out