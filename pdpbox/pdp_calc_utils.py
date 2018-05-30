import pandas as pd
import numpy as np





def _get_grid_combos(feature_grids, feature_types):
    # create grid combination
    grids1, grids2 = feature_grids
    if feature_types[0] == 'onehot':
        grids1 = range(len(grids1))
    if feature_types[1] == 'onehot':
        grids2 = range(len(grids2))

    grid_combos_temp = np.matrix(np.array(np.meshgrid(grids1, grids2)).T.reshape(-1, 2))
    grid_combos1, grid_combos2 = grid_combos_temp[:, 0], grid_combos_temp[:, 1]
    if feature_types[0] == 'onehot':
        grid_combos1_temp = np.array(grid_combos1.T, dtype=np.int64)[0]
        grid_combos1 = np.zeros((len(grid_combos1), len(grids1)), dtype=int)
        grid_combos1[range(len(grid_combos1)), grid_combos1_temp] = 1
    if feature_types[1] == 'onehot':
        grid_combos2_temp = np.array(grid_combos2.T, dtype=np.int64)[0]
        grid_combos2 = np.zeros((len(grid_combos2), len(grids2)), dtype=int)
        grid_combos2[range(len(grid_combos2)), grid_combos2_temp] = 1

    grid_combos = np.array(np.concatenate((grid_combos1, grid_combos2), axis=1))

    return grid_combos


def _calc_ice_lines(feature_grid, data, model, model_features, n_classes, feature, feature_type,
                    predict_kwds, data_transformer):
    """Apply predict function on a feature_grid

    Returns
    -------
    Predicted result on this feature_grid
    """

    _data = data.copy()
    if feature_type == 'onehot':
        # for onehot encoding feature, need to change all levels together
        other_grids = [grid for grid in feature if grid != feature_grid]
        _data[feature_grid] = 1
        for grid in other_grids:
            _data[grid] = 0
    else:
        _data[feature] = feature_grid

    # if there are other features highly depend on the investigating feature
    # other features should also adjust based on the changed feature value
    # Example:
    # there are 3 features: a, b, a_b_ratio
    # if feature a is the investigated feature
    # data_transformer should be:
    # def data_transformer(df):
    #   df["a_b_ratio"] = df["a"] / df["b"]
    #   return df
    if data_transformer is not None:
        _data = data_transformer(_data)

    if n_classes == 0:
        predict = model.predict
    else:
        predict = model.predict_proba

    # get predictions for this chunk
    preds = predict(_data[model_features], **predict_kwds)

    if n_classes == 0:
        grid_results = pd.DataFrame(preds, columns=[feature_grid])
    elif n_classes == 2:
        grid_results = pd.DataFrame(preds[:, 1], columns=[feature_grid])
    else:
        grid_results = []
        for n_class in range(n_classes):
            grid_result = pd.DataFrame(preds[:, n_class], columns=[feature_grid])
            grid_results.append(grid_result)

    return grid_results





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
    _data = data.copy()
    for idx in range(len(feature_list)):
        _data[feature_list[idx]] = feature_grids_combo[idx]

    if data_transformer is not None:
        _data = data_transformer(_data)

    if classifier:
        predict = model.predict_proba
    else:
        predict = model.predict

    preds = predict(_data[model_features], **predict_kwds)
    result = _data[feature_list].copy()

    if n_classes > 2:
        for n_class in range(n_classes):
            result['class_%d_preds' % n_class] = preds[:, n_class]
    else:
        if classifier:
            result['preds'] = preds[:, 1]
        else:
            result['preds'] = preds

    return result


def _pdp_count_dist_xticklabels(feature_grids):
    """Create bucket names based on feature grids"""
    column_names = []

    # number of buckets: len(feature_grids) - 1
    for i in range(len(feature_grids) - 1):
        column_name = '[%.2f, %.2f)' % (feature_grids[i], feature_grids[i + 1])

        if i == len(feature_grids) - 2:
            column_name = '[%.2f, %.2f]' % (feature_grids[i], feature_grids[i + 1])
        column_names.append(column_name)

    return column_names


def _prepare_pdp_count_data(feature, feature_type, data, feature_grids):
    """Calculate data point distribution

    Returns
    -------
    count_data: pandas DataFrame
        column x: bucket index,
        column count: number of data points fall in this bucket,
        column count_norm: normalized count number, notice that it is normalized
        by data.shape[0], just incase for onehot feature, not every data point has value
    """
    if feature_type == 'onehot':
        count_data = pd.DataFrame(data[feature_grids].sum(axis=0)).reset_index(drop=False).rename(columns={0: 'count'})
        count_data['x'] = range(count_data.shape[0])
    elif feature_type == 'binary':
        count_data = pd.DataFrame(data={'x': [0, 1], 'count': [data[data[feature] == 0].shape[0],
                                                               data[data[feature] == 1].shape[0]]})
    else:
        data_x = data.copy()
        vmin, vmax = data[feature].min(), data[feature].max()
        feature_grids = list(feature_grids)
        count_x = range(len(feature_grids) - 1)

        # append lower and upper bound to the grids
        # make sure all data points are included
        if feature_grids[0] > vmin:
            feature_grids = [vmin] + feature_grids
            count_x = [-1] + count_x
        if feature_grids[-1] < vmax:
            feature_grids = feature_grids + [vmax]
            count_x = count_x + [count_x[-1] + 1]

        data_x['x'] = pd.cut(x=data_x[feature].values, bins=feature_grids, labels=False, include_lowest=True)
        data_x['count'] = 1
        count_data_temp = data_x.groupby('x', as_index=False).agg(
            {'count': 'count'}).sort_values('x', ascending=True).reset_index(drop=True)
        count_data = pd.DataFrame(data={'x': range(len(feature_grids) - 1),
                                        'xticklabels': _pdp_count_dist_xticklabels(feature_grids=feature_grids)})
        count_data = count_data.merge(count_data_temp, on='x', how='left').fillna(0)
        count_data['x'] = count_x

    count_data['count_norm'] = count_data['count'] * 1.0 / data.shape[0]

    return count_data
