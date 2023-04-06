import pandas as pd
import numpy as np
from sklearn.cluster import MiniBatchKMeans, KMeans
import copy
from .utils import _get_string, _find_bucket, _calc_preds, _make_list


def _calc_grid_results(
    model,
    data,
    grid_idx,
    n_classes,
    pred_func,
    from_model,
    predict_kwds,
    data_trans,
    chunk_size,
    unit_test=False,
):
    if data_trans is not None:
        data = data_trans(data)

    preds = _calc_preds(model, data, pred_func, from_model, predict_kwds, chunk_size)

    if n_classes == 0:
        grid_results = pd.DataFrame(preds, columns=[grid_idx])
    elif n_classes == 2:
        grid_results = pd.DataFrame(preds[:, 1], columns=[grid_idx])
    else:
        grid_results = []
        for n_class in range(n_classes):
            grid_result = pd.DataFrame(preds[:, n_class], columns=[grid_idx])
            grid_results.append(grid_result)

    # _data is returned for unit test
    if unit_test:
        return grid_results, data
    else:
        return grid_results


def _calc_ice_lines(
    model,
    data,
    feat,
    feat_type,
    feat_grid,
    grid_idx,
    n_classes,
    pred_func,
    from_model,
    predict_kwds,
    data_trans,
    chunk_size,
    unit_test=False,
):
    """Apply predict function on a feature_grid

    Returns
    -------
    Predicted result on this feature_grid
    """
    if feat_type == "onehot":
        # for onehot encoding feature, need to change all levels together
        data[feat] = 0
        data[feat_grid] = 1
    else:
        data[feat] = feat_grid

    return _calc_grid_results(
        model,
        data,
        grid_idx,
        n_classes,
        pred_func,
        from_model,
        predict_kwds,
        data_trans,
        chunk_size,
        unit_test,
    )


def _calc_ice_lines_inter(
    model,
    data,
    feats,
    feat_grid_combo,
    grid_idx,
    n_classes,
    pred_func,
    from_model,
    predict_kwds,
    data_trans,
    chunk_size,
    unit_test=False,
):
    """Apply predict function on a grid combo

    Returns
    -------
    Predicted result on this feature_grid
    """

    feat_list = _make_list(feats[0]) + _make_list(feats[1])
    for i, feat in enumerate(feat_list):
        data[feat] = feat_grid_combo[i]

    return _calc_grid_results(
        model,
        data,
        grid_idx,
        n_classes,
        pred_func,
        from_model,
        predict_kwds,
        data_trans,
        chunk_size,
        unit_test,
    )


def _cluster_ice_lines(ice_lines, feature_grids, plot_style):
    method = plot_style.clustering["method"]
    if method not in ["approx", "accurate"]:
        raise ValueError('cluster method: should be "approx" or "accurate".')

    n_centers = plot_style.clustering["n_centers"]
    if method == "approx":
        kmeans = MiniBatchKMeans(n_clusters=n_centers, random_state=0, verbose=0)
    else:
        kmeans = KMeans(n_clusters=n_centers, random_state=0)

    kmeans.fit(ice_lines[np.arange(len(feature_grids))])
    return pd.DataFrame(kmeans.cluster_centers_, columns=feature_grids)


def _sample_ice_lines(ice_lines, frac_to_plot):
    """Get sample ice lines to plot

    Notes
    -----
    If frac_to_plot==1, will plot all lines instead of sampling one line

    """

    if frac_to_plot < 1.0:
        ice_lines = ice_lines.sample(int(ice_lines.shape[0] * frac_to_plot))
    elif frac_to_plot > 1:
        ice_lines = ice_lines.sample(frac_to_plot)

    return ice_lines.reset_index(drop=True)


def _prepare_pdp_line_data(
    class_id,
    pdp_isolate_obj,
    plot_style,
):
    pdp_result = pdp_isolate_obj.results[class_id]
    pdp = copy.deepcopy(pdp_result.pdp)
    ice_lines = copy.deepcopy(pdp_result.ice_lines)
    feature_grids = pdp_isolate_obj.feature_grids
    grid_indices = np.arange(len(feature_grids))

    x = grid_indices
    if pdp_isolate_obj.feature_type == "numeric" and not plot_style.x_quantile:
        x = feature_grids

    if plot_style.center:
        pdp -= pdp[0]
        for i in np.arange(1, len(feature_grids)):
            ice_lines[i] -= ice_lines[0]
        ice_lines[0] = 0

    line_data = None
    if plot_style.plot_lines:
        if plot_style.clustering["on"]:
            line_data = _cluster_ice_lines(ice_lines, feature_grids, plot_style)
        else:
            line_data = _sample_ice_lines(ice_lines, plot_style.frac_to_plot)

    line_std = ice_lines[grid_indices].std().values

    return x, pdp, line_data, line_std


def _pdp_count_dist_xticklabels(feature_grids):
    """Create bucket names based on feature grids"""
    column_names = []
    feature_grids_str = [_get_string(grid) for grid in feature_grids]

    # number of buckets: len(feature_grids) - 1
    for i in range(len(feature_grids_str) - 1):
        column_name = "[%s, %s)" % (feature_grids_str[i], feature_grids_str[i + 1])

        if i == len(feature_grids_str) - 2:
            column_name = "[%s, %s]" % (feature_grids_str[i], feature_grids_str[i + 1])
        column_names.append(column_name)

    return column_names


def _prepare_pdp_count_data(
    feature,
    feature_type,
    data,
    num_grid_points,
    grid_type,
    percentile_range,
    grid_range,
    cust_grid_points,
):
    """Calculate data point distribution

    Returns
    -------
    count_data: pandas DataFrame
        column x: bucket index,
        column count: number of data points fall in this bucket,
        column count_norm: normalized count number, notice that it is normalized
        by data.shape[0], just incase for onehot feature, not every data point has value
    """

    prepared_results = _prepare_data_x(
        feature,
        feature_type,
        data,
        num_grid_points,
        grid_type,
        percentile_range,
        grid_range,
        cust_grid_points,
        show_percentile=True,
        show_outliers=False,
        endpoint=True,
    )
    data_x = prepared_results["data"]
    data_x["count"] = 1
    count_data = (
        data_x.groupby("x", as_index=False)
        .agg({"count": "count"})
        .sort_values("x", ascending=True)
    )
    count_data["count_norm"] = count_data["count"] * 1.0 / data.shape[0]
    prepared_results["count"] = count_data
    dist_data = data[feature]
    num_samples = 1000
    if len(dist_data) > num_samples:
        dist_data = dist_data.sample(num_samples, replace=False)
    prepared_results["dist"] = dist_data.values

    return prepared_results
