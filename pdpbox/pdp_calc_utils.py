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


def _calc_ice_lines_inter(data, model, classifier, model_features, n_classes, feature_list,
                          feature_grids_combo, predict_kwds, data_transformer):

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

    for idx in range(len(feature_list)):
        data[feature_list[idx]] = feature_grids_combo[idx]

    if data_transformer is not None:
        data = data_transformer(data)

    if classifier:
        predict = model.predict_proba
    else:
        predict = model.predict

    preds = predict(data[model_features], **predict_kwds)
    result = data[feature_list].copy()

    if n_classes > 2:
        for n_class in range(n_classes):
            result['class_%d_preds' % n_class] = preds[:, n_class]
    else:
        if classifier:
            result['preds'] = preds[:, 1]
        else:
            result['preds'] = preds

    return result
