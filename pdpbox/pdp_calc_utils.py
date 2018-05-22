import pandas as pd
import numpy as np


def _get_grids(x, num_grid_points, grid_type, percentile_range, grid_range):
    """calculate grid points for numeric feature

    Parameters:
    -----------

    :param x: Series, 1d-array, or list.
        values to calculate grid points
    :param num_grid_points: integer
        number of grid points for numeric feature
    :param grid_type: string
        'percentile' or 'equal'
        type of grid points for numeric feature
    :param percentile_range: tuple or None
        percentile range to investigate
        for numeric feature when grid_type='percentile'
    :param grid_range: tuple or None
        value range to investigate
        for numeric feature when grid_type='equal'

    Returns:
    --------

    :return feature_grids: 1d-array
        calculated grid points
    :return percentile_info: 1d-array or []
        percentile information for feature_grids
        exists when grid_type='percentile'
    """

    if grid_type == 'percentile':
        # grid points are calculated based on percentile in unique level
        # thus the final number of grid points might be smaller than num_grid_points
        start, end = 0, 100
        if percentile_range is not None:
            start, end = np.min(percentile_range), np.max(percentile_range)

        percentile_grids = np.linspace(start=start, stop=end, num=num_grid_points)
        value_grids = np.percentile(x, percentile_grids)

        grids_df = pd.DataFrame()
        grids_df['percentile_grids'] = [round(v, 2) for v in percentile_grids]
        grids_df['value_grids'] = value_grids
        grids_df = grids_df.groupby(['value_grids'], as_index=False).agg(
            {'percentile_grids': lambda v: str(tuple(v)).replace(',)', ')')}).sort_values('value_grids', ascending=True)

        return grids_df['value_grids'].values, grids_df['percentile_grids'].values
    else:
        if grid_range is not None:
            value_grids = np.linspace(np.min(grid_range), np.max(grid_range), num_grid_points)
        else:
            value_grids = np.linspace(np.min(x), np.max(x), num_grid_points)

        return value_grids, []


def _calc_ice_lines(data, model, classifier, model_features, n_classes, feature, feature_type,
                    feature_grid, predict_kwds, data_transformer):

    _data = data.copy()
    if feature_type == 'onehot':
        other_grids = [grid for grid in feature if grid != feature_grid]
        _data[feature_grid] = 1
        for grid in other_grids:
            _data[grid] = 0
    else:
        _data[feature] = feature_grid

    if data_transformer is not None:
        _data = data_transformer(_data)

    if classifier:
        predict = model.predict_proba
    else:
        predict = model.predict

    # get predictions for this chunk
    preds = predict(_data[model_features], **predict_kwds)

    if n_classes > 2:
        grid_results = []
        for n_class in range(n_classes):
            grid_result = pd.DataFrame(preds[:, n_class], columns=[feature_grid])
            grid_results.append(grid_result)
    else:
        if classifier:
            grid_results = pd.DataFrame(preds[:, 1], columns=[feature_grid])
        else:
            grid_results = pd.DataFrame(preds, columns=[feature_grid])

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
    """map one-hot value to one-hot name

    :param x: 1-d array
        example: [0, 1, 0, 0]

    :return value: string
        index of the mapped one-hot name
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


def _find_bucket(x, feature_grids, endpoint):
    # map value into value bucket
    if x < feature_grids[0]:
        bucket = 0
    else:
        if endpoint:
            if x > feature_grids[-1]:
                bucket = len(feature_grids)
            else:
                bucket = len(feature_grids) - 1
                for i in range(len(feature_grids) - 2):
                    if feature_grids[i] <= x < feature_grids[i + 1]:
                        bucket = i + 1
        else:
            if x >= feature_grids[-1]:
                bucket = len(feature_grids)
            else:
                bucket = len(feature_grids) - 1
                for i in range(len(feature_grids) - 2):
                    if feature_grids[i] <= x < feature_grids[i + 1]:
                        bucket = i + 1
    return bucket


def _make_bucket_column_names(feature_grids, endpoint):
    # create bucket names
    column_names = []
    bound_lows = [np.nan]
    bound_ups = [feature_grids[0]]

    # number of buckets: len(feature_grids) - 1
    for i in range(len(feature_grids) - 1):
        column_name = '[%.2f, %.2f)' % (feature_grids[i], feature_grids[i + 1])
        bound_lows.append(feature_grids[i])
        bound_ups.append(feature_grids[i + 1])

        if (i == len(feature_grids) - 2) and endpoint:
            column_name = '[%.2f, %.2f]' % (feature_grids[i], feature_grids[i + 1])

        column_names.append(column_name)

    if endpoint:
        column_names = ['< %.2f' % feature_grids[0]] + column_names + ['> %.2f' % feature_grids[-1]]
    else:
        column_names = ['< %.2f' % feature_grids[0]] + column_names + ['>= %.2f' % feature_grids[-1]]
    bound_lows.append(feature_grids[-1])
    bound_ups.append(np.nan)

    return column_names, bound_lows, bound_ups


def _make_bucket_column_names_percentile(percentile_info, endpoint):
    # create percentile bucket names
    percentile_column_names = []
    percentile_bound_lows = [0]
    percentile_bound_ups = [np.min(np.array(percentile_info[0].replace('(', '').replace(
        ')', '').split(', ')).astype(np.float64))]

    for i in range(len(percentile_info) - 1):
        # for each grid point, percentile information is in tuple format
        # (percentile1, percentile2, ...)
        # some grid points would belong to multiple percentiles
        low = np.min(np.array(percentile_info[i].replace('(', '').replace(')', '').split(', ')).astype(np.float64))
        high = np.max(np.array(percentile_info[i + 1].replace('(', '').replace(')', '').split(', ')).astype(np.float64))
        percentile_column_name = '[%.2f, %.2f)' % (low, high)
        percentile_bound_lows.append(low)
        percentile_bound_ups.append(high)

        if i == len(percentile_info) - 2:
            if endpoint:
                percentile_column_name = '[%.2f, %.2f]' % (low, high)
            else:
                percentile_column_name = '[%.2f, %.2f)' % (low, high)

        percentile_column_names.append(percentile_column_name)

    low = np.min(np.array(percentile_info[0].replace('(', '').replace(')', '').split(', ')).astype(np.float64))
    high = np.max(np.array(percentile_info[-1].replace('(', '').replace(')', '').split(', ')).astype(np.float64))
    if endpoint:
        percentile_column_names = ['< %.2f' % low] + percentile_column_names + ['> %.2f' % high]
    else:
        percentile_column_names = ['< %.2f' % low] + percentile_column_names + ['>= %.2f' % high]
    percentile_bound_lows.append(high)
    percentile_bound_ups.append(100)

    return percentile_column_names, percentile_bound_lows, percentile_bound_ups


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
