
import pandas as pd
import numpy as np
from pdpbox.utils import _get_string, _find_bucket


def _calc_ice_lines(feature_grid, data, model, model_features, n_classes, feature, feature_type,
                    predict_kwds, data_transformer, unit_test=False):
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

    # _data is returned for unit test
    if unit_test:
        return grid_results, _data
    else:
        return grid_results


def _calc_ice_lines_inter(feature_grids_combo, data, model, model_features, n_classes, feature_list,
                          predict_kwds, data_transformer, unit_test=False):
    """Apply predict function on a grid combo

    Returns
    -------
    Predicted result on this feature_grid
    """

    _data = data.copy()
    for idx in range(len(feature_list)):
        _data[feature_list[idx]] = feature_grids_combo[idx]

    if data_transformer is not None:
        _data = data_transformer(_data)

    if n_classes == 0:
        predict = model.predict
    else:
        predict = model.predict_proba

    preds = predict(_data[model_features], **predict_kwds)
    grid_result = _data[feature_list].copy()

    if n_classes == 0:
        grid_result['preds'] = preds
    elif n_classes == 2:
        grid_result['preds'] = preds[:, 1]
    else:
        for n_class in range(n_classes):
            grid_result['class_%d_preds' % n_class] = preds[:, n_class]

    # _data is returned for unit test
    if unit_test:
        return grid_result, _data
    else:
        return grid_result


def _pdp_count_dist_xticklabels(feature_grids):
    """Create bucket names based on feature grids"""
    column_names = []
    feature_grids_str = [_get_string(grid) for grid in feature_grids]

    # number of buckets: len(feature_grids) - 1
    for i in range(len(feature_grids_str) - 1):
        column_name = '[%s, %s)' % (feature_grids_str[i], feature_grids_str[i + 1])

        if i == len(feature_grids_str) - 2:
            column_name = '[%s, %s]' % (feature_grids_str[i], feature_grids_str[i + 1])
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
        count_x = list(np.arange(len(feature_grids) - 1))

        # append lower and upper bound to the grids
        # make sure all data points are included
        if feature_grids[0] > vmin:
            feature_grids = [vmin] + feature_grids
            count_x = [-1] + count_x
        if feature_grids[-1] < vmax:
            feature_grids = feature_grids + [vmax]
            count_x = count_x + [count_x[-1] + 1]

        data_x['x'] = data_x[feature].apply(lambda x: _find_bucket(x=x, feature_grids=feature_grids, endpoint=True))
        data_x = data_x[~data_x['x'].isnull()]
        data_x['count'] = 1
        count_data_temp = data_x.groupby('x', as_index=False).agg(
            {'count': 'count'}).sort_values('x', ascending=True).reset_index(drop=True)
        count_data_temp['x'] = count_data_temp['x'] - count_data_temp['x'].min()
        count_data = pd.DataFrame(data={'x': range(len(feature_grids) - 1),
                                        'xticklabels': _pdp_count_dist_xticklabels(feature_grids=feature_grids)})
        count_data = count_data.merge(count_data_temp, on='x', how='left').fillna(0)
        count_data['x'] = count_x

    count_data['count_norm'] = count_data['count'] * 1.0 / data.shape[0]

    return count_data
