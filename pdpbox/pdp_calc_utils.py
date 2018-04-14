import pandas as pd
import numpy as np


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


def _make_ice_data(data, feature, feature_type, feature_grids):
    """
    Prepare data for calculating ice lines

    :param data: chunk of data to calculate on
    :param feature: column name of the feature
    :param feature_type: type of the feature
    :param feature_grids: array of grids to calculate on

    :return:
        the extended data chunk (extended by feature_grids)
    """

    ice_data = pd.DataFrame(np.repeat(data.values, feature_grids.size, axis=0), columns=data.columns)

    if feature_type == 'onehot':
        for i, col in enumerate(feature):
            col_value = [0] * feature_grids.size
            col_value[i] = 1
            ice_data[col] = np.tile(col_value, data.shape[0])
    else:
        ice_data[feature] = np.tile(feature_grids, data.shape[0])

    return ice_data


def _calc_ice_lines(data_chunk, model, classifier, model_features, n_classes, feature, feature_type,
                    feature_grids, display_columns, actual_columns, predict_kwds):
    """
    Calculate ice lines

    :param data_chunk: chunk of data to calculate on
    :param model: the fitted model
    :param classifier: whether the model is a classifier
    :param model_features: features used by the model
    :param n_classes: number of classes if it is a classifier
    :param feature: column name of the feature
    :param feature_type: type of the feature
    :param feature_grids: array of grids to calculate on
    :param display_columns: column names to display
    :param actual_columns: column names of the actual values
    :param predict_kwds: other parameters pass to the predictor

    :return:
        a dataframe (or a list of dataframes, when it is multi-class problem) of calculated ice lines
        each row in a dataframe represents one line
    """

    ice_chunk = _make_ice_data(data_chunk[model_features], feature, feature_type, feature_grids)

    if classifier:
        predict = model.predict_proba
    else:
        predict = model.predict

    # get predictions for this chunk
    preds = predict(ice_chunk[model_features], **predict_kwds)

    # if it is multi-class problem, the return result is a list of dataframes
    # display_columns are columns for pdp predictions
    if n_classes > 2:
        ice_chunk_results = []
        for n_class in range(n_classes):
            ice_chunk_result = pd.DataFrame(preds[:, n_class].reshape((data_chunk.shape[0], feature_grids.size)),
                                            columns=display_columns)
            ice_chunk_result = pd.concat([ice_chunk_result, data_chunk[actual_columns]], axis=1)
            ice_chunk_result['actual_preds'] = data_chunk['actual_preds_class_%d' % n_class].values
            ice_chunk_results.append(ice_chunk_result)
    else:
        if classifier:
            ice_chunk_result = pd.DataFrame(preds[:, 1].reshape((data_chunk.shape[0], feature_grids.size)),
                                            columns=display_columns)
        else:
            ice_chunk_result = pd.DataFrame(preds.reshape((data_chunk.shape[0], feature_grids.size)),
                                            columns=display_columns)
        ice_chunk_result = pd.concat([ice_chunk_result, data_chunk[actual_columns]], axis=1)
        ice_chunk_result['actual_preds'] = data_chunk['actual_preds'].values

    if n_classes > 2:
        return ice_chunk_results
    else:
        return ice_chunk_result


def _sample_data(ice_lines, frac_to_plot):
    """
    Get sample ice lines to plot

    :param ice_lines: all ice lines
    :param frac_to_plot: fraction to plot

    :return: the sampled ice lines
    """

    if frac_to_plot < 1.:
        ice_plot_data = ice_lines.sample(int(ice_lines.shape[0] * frac_to_plot))
    elif frac_to_plot > 1:
        ice_plot_data = ice_lines.sample(frac_to_plot)
    else:
        ice_plot_data = ice_lines.copy()

    ice_plot_data = ice_plot_data.reset_index(drop=True)
    return ice_plot_data


def _find_onehot_actual(x):
    """
    Find the actual value for one-hot encoding feature

    :param x: all related one-hot encoding values

    :return: index of the actual value
    """

    try:
        value = list(x).index(1)
    except:
        value = np.nan
    return value


def _find_closest(x, feature_grids):
    """
    Find the closest feature grid for x

    :param x: value
    :param feature_grids: array of feature grids

    :return:
    """
    values = list(feature_grids)
    return values.index(min(values, key=lambda y: abs(y-x)))


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
