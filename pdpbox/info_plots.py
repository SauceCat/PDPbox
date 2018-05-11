
import pandas as pd

from .info_plot_utils import _target_plot, _info_plot_interact, _prepare_data_x, _actual_plot
from .other_utils import _check_feature, _check_percentile_range, _make_list, \
    _expand_default, _check_model, _check_dataset, _check_target, _check_grid_type, _check_classes


def target_plot(df, feature, feature_name, target, num_grid_points=10, grid_type='percentile',
                percentile_range=None, grid_range=None, cust_grid_points=None, show_percentile=False,
                show_outliers=False, figsize=None, ncols=2, plot_params=None):
    """Plot average target value across different feature values (feature grids)

    Parameters:
    -----------

    :param df: pandas DataFrame
        data set to investigate on, should contain at least
        the feature to investigate as well as the target
    :param feature: string or list
        feature or feature list to investigate
        for one-hot encoding features, feature list is required
    :param feature_name: string
        name of the feature, not necessary a column name
    :param target: string or list
        column name or column name list for target value
        for multi-class problem, a list of one-hot encoding target column
    :param num_grid_points: integer, optional, default=10
        number of grid points for numeric feature
    :param grid_type: string, optional, default='percentile'
        'percentile' or 'equal'
        type of grid points for numeric feature
    :param percentile_range: tuple or None, optional, default=None
        percentile range to investigate
        for numeric feature when grid_type='percentile'
    :param grid_range: tuple or None, optional, default=None
        value range to investigate
        for numeric feature when grid_type='equal'
    :param cust_grid_points: Series, 1d-array, list or None, optional, default=None
        customized list of grid points
        for numeric feature
    :param show_percentile: bool, optional, default=False
        whether to display the percentile buckets
        for numeric feature when grid_type='percentile'
    :param show_outliers: bool, optional, default=False
        whether to display the out of range buckets
        for numeric feature when percentile_range or grid_range is not None
    :param figsize: tuple or None, optional, default=None
        size of the figure, (width, height)
    :param ncols: integer, optional, default=2
        number subplot columns, used when it is multi-class problem
    :param plot_params: dict or None, optional, default=None
        parameters for the plot

    Return:
    -------

    :return axes: matplotlib Axes
        Returns the Axes object with the plot for further tweaking
    :return summary_df: pandas DataFrame
        Graph data in data frame format
    """

    # check inputs
    _check_dataset(df)
    feature_type = _check_feature(feature=feature, df=df)
    _ = _check_target(target=target, df=df)
    _check_grid_type(grid_type=grid_type)
    _check_percentile_range(percentile_range=percentile_range)

    # show_outliers should be only turned on when necessary
    if (percentile_range is None) and (grid_range is None) and (cust_grid_points is None):
        show_outliers = False

    # create feature grids and bar counts
    target = _make_list(target)
    useful_features = _make_list(feature) + target

    # map feature values to grid point buckets (x)
    data = df[useful_features].copy()
    data_x, display_columns, bound_lows, bound_ups, percentile_columns, percentile_bound_lows, percentile_bound_ups = \
        _prepare_data_x(feature=feature, feature_type=feature_type, data=data, num_grid_points=num_grid_points,
                        grid_type=grid_type, percentile_range=percentile_range, grid_range=grid_range,
                        cust_grid_points=cust_grid_points, show_percentile=show_percentile, show_outliers=show_outliers)

    # prepare data for bar plot
    data_x['fake_count'] = 1
    bar_data = data_x.groupby('x', as_index=False).agg({'fake_count': 'count'}).sort_values('x', ascending=True)

    # prepare summary data frame
    summary_df = pd.DataFrame(range(data_x['x'].min(), data_x['x'].max() + 1), columns=['x'])
    summary_df = summary_df.merge(bar_data.rename(columns={'fake_count': 'count'}), on='x', how='left').fillna(0)

    # prepare data for target lines
    # each target line contains 'x' and mean target value
    target_lines = []
    for target_idx in range(len(target)):
        target_line = data_x.groupby('x', as_index=False).agg(
            {target[target_idx]: 'mean'}).sort_values('x', ascending=True)
        target_lines.append(target_line)
        summary_df = summary_df.merge(target_line, on='x', how='outer').fillna(0)

    # map column names back
    summary_df['display_column'] = summary_df['x'].apply(lambda x: display_columns[int(x)])
    info_cols = ['x', 'display_column']
    if feature_type == 'numeric':
        summary_df['value_lower'] = summary_df['x'].apply(lambda x: bound_lows[int(x)])
        summary_df['value_upper'] = summary_df['x'].apply(lambda x: bound_ups[int(x)])
        info_cols += ['value_lower', 'value_upper']

    if len(percentile_columns) != 0:
        summary_df['percentile_column'] = summary_df['x'].apply(lambda x: percentile_columns[int(x)])
        summary_df['percentile_lower'] = summary_df['x'].apply(lambda x: percentile_bound_lows[int(x)])
        summary_df['percentile_upper'] = summary_df['x'].apply(lambda x: percentile_bound_ups[int(x)])
        info_cols += ['percentile_column', 'percentile_lower', 'percentile_upper']
    summary_df = summary_df[info_cols + ['count'] + target]

    # inner call target plot
    axes = _target_plot(
        feature_name=feature_name, display_columns=display_columns, percentile_columns=percentile_columns,
        target=target, bar_data=bar_data, target_lines=target_lines, figsize=figsize, ncols=ncols,
        plot_params=plot_params)

    return axes, summary_df


def q1(x):
    return x.quantile(0.25)


def q2(x):
    return x.quantile(0.5)


def q3(x):
    return x.quantile(0.75)


def actual_plot(model, X, feature, feature_name, num_grid_points=10, grid_type='percentile', percentile_range=None,
                grid_range=None, cust_grid_points=None, show_percentile=False, show_outliers=False,
                which_classes=None, predict_kwds={}, ncols=2, figsize=None, plot_params=None):

    # check inputs
    n_classes, classifier, predict = _check_model(model=model)
    _check_dataset(df=X)
    feature_type = _check_feature(feature=feature, df=X)
    _check_grid_type(grid_type=grid_type)
    _check_percentile_range(percentile_range=percentile_range)
    _check_classes(classes_list=which_classes, n_classes=n_classes)

    # show_outliers should be only turned on when necessary
    if (percentile_range is None) and (grid_range is None) and (cust_grid_points is None):
        show_outliers = False

    # make predictions
    # info_df only contains feature value and actual predictions
    prediction = predict(X, **predict_kwds)
    info_df = X[_make_list(feature)]
    actual_prediction_columns = ['actual_prediction']
    if n_classes == 0:
        info_df['actual_prediction'] = prediction
    elif n_classes == 2:
        info_df['actual_prediction'] = prediction[:, 1]
    else:
        plot_classes = range(n_classes)
        if which_classes is not None:
            plot_classes = sorted(which_classes)

        actual_prediction_columns = []
        for class_idx in plot_classes:
            info_df['actual_prediction_%d' % class_idx] = prediction[:, class_idx]
            actual_prediction_columns.append('actual_prediction_%d' % class_idx)

    info_df_x, display_columns, percentile_columns = _prepare_data_x(
        feature=feature, feature_type=feature_type, data=info_df, num_grid_points=num_grid_points, grid_type=grid_type,
        percentile_range=percentile_range, grid_range=grid_range, cust_grid_points=cust_grid_points,
        show_percentile=show_percentile, show_outliers=show_outliers)

    info_df_x['fake_count'] = 1
    bar_data = info_df_x.groupby('x', as_index=False).agg({'fake_count': 'count'}).sort_values('x', ascending=True)
    summary_df = pd.DataFrame(range(info_df_x['x'].min(), info_df_x['x'].max() + 1), columns=['x'])
    summary_df = summary_df.merge(bar_data.rename(columns={'fake_count': 'count'}), on='x', how='left').fillna(0)

    # prepare data for box lines
    # each box line contains 'x' and actual prediction q1, q2, q3
    box_lines = []
    actual_prediction_columns_qs = []
    for idx in range(len(actual_prediction_columns)):
        box_line = info_df_x.groupby('x', as_index=False).agg(
            {actual_prediction_columns[idx]: [q1, q2, q3]}).sort_values('x', ascending=True)
        box_line.columns = ['_'.join(col) if col[1] != '' else col[0] for col in box_line.columns]
        box_lines.append(box_line)
        actual_prediction_columns_qs += [actual_prediction_columns[idx] + '_%s' % q for q in ['q1', 'q2', 'q3']]
        summary_df = summary_df.merge(box_line, on='x', how='outer').fillna(0)

    summary_df['display_column'] = summary_df['x'].apply(lambda x: display_columns[int(x)])
    info_cols = ['x', 'display_column']
    if len(percentile_columns) != 0:
        summary_df['percentile_column'] = summary_df['x'].apply(lambda x: percentile_columns[int(x)])
        info_cols.append('percentile_column')
    summary_df = summary_df[info_cols + ['count'] + actual_prediction_columns_qs]

    axes = _actual_plot(plot_data=info_df_x, bar_data=bar_data, box_lines=box_lines,
                        actual_prediction_columns=actual_prediction_columns,
                        feature_name=feature_name, display_columns=display_columns,
                        percentile_columns=percentile_columns, figsize=figsize, ncols=ncols, plot_params=plot_params)
    return axes, summary_df


def target_plot_interact(df, features, feature_names, target, num_grid_points=None, grid_types=None,
                         percentile_ranges=None, grid_ranges=None, cust_grid_points=None, show_percentile=False,
                         show_outliers=False, figsize=None, ncols=2, plot_params=None):
    """Plot average target value across different feature value combinations (feature grid combinations)

    Parameters:
    -----------

    :param df: pandas DataFrame
        data set to investigate on, should contain at least
        the feature to investigate as well as the target
    :param features: list
        two features to investigate
    :param feature_names: list
        feature names
    :param target: string or list
        column name or column name list for target value
        for multi-class problem, a list of one-hot encoding target column
    :param num_grid_points: list, optional, default=None
        number of grid points for each feature
    :param grid_types: list, optional, default=None
        type of grid points for each feature
    :param percentile_ranges: list of tuple, optional, default=None
        percentile range to investigate for each feature
    :param grid_ranges: list of tuple, optional, default=None
        value range to investigate for each feature
    :param cust_grid_points: list of (Series, 1d-array, list), optional, default=None
        customized list of grid points for each feature
    :param show_percentile: bool, optional, default=False
        whether to display the percentile buckets for both feature
    :param show_outliers: bool, optional, default=False
        whether to display the out of range buckets for both features
    :param figsize: tuple or None, optional, default=None
        size of the figure, (width, height)
    :param ncols: integer, optional, default=2
        number subplot columns, used when it is multi-class problem
    :param plot_params: dict or None, optional, default=None
        parameters for the plot

    Notes:
    ------
    Parameters are consistent with the ones for function target_plot
    But for this function, you need to specify parameter value for both features
    in list format
    For example:
    percentile_ranges = [(0, 90), (5, 95)] means
    percentile_range = (0, 90) for feature 1
    percentile_range = (5, 95) for feature 2

    Return:
    -------

    :return axes: matplotlib Axes
        Returns the Axes object with the plot for further tweaking
    :return summary_df: pandas DataFrame
        Graph data in data frame format
    """

    # check inputs
    _check_dataset(df=df)
    feature_types = [_check_feature(feature=features[0], df=df), _check_feature(feature=features[1], df=df)]
    _ = _check_target(target=target, df=df)

    grid_types = _expand_default(grid_types, 'percentile')
    _check_grid_type(grid_type=grid_types[0])
    _check_grid_type(grid_type=grid_types[1])

    percentile_ranges = _expand_default(percentile_ranges, None)
    _check_percentile_range(percentile_range=percentile_ranges[0])
    _check_percentile_range(percentile_range=percentile_ranges[1])

    # expand some inputs
    num_grid_points = _expand_default(num_grid_points, 10)
    grid_ranges = _expand_default(grid_ranges, None)
    cust_grid_points = _expand_default(cust_grid_points, None)

    if not show_outliers:
        show_outliers = [False, False]
    else:
        show_outliers = [True, True]
        for i in range(2):
            if (percentile_ranges[i] is None) and (grid_ranges[i] is None) and (cust_grid_points[i] is None):
                show_outliers[i] = False

    # set up graph parameters
    if plot_params is None:
        plot_params = dict()

    # create feature grids and bar counts
    target = _make_list(target)
    useful_features = _make_list(features[0]) + _make_list(features[1]) + target

    # prepare data for bar plot
    data = df[useful_features].copy()

    # (data_x, display_columns, percentile_columns)
    results = []
    data_input = data
    for i in range(2):
        result = _prepare_data_x(
            feature=features[i], feature_type=feature_types[i], data=data_input,
            num_grid_points=num_grid_points[i], grid_type=grid_types[i], percentile_range=percentile_ranges[i],
            grid_range=grid_ranges[i], cust_grid_points=cust_grid_points[i],
            show_percentile=show_percentile, show_outliers=show_outliers[i])
        results.append(result)
        if i == 0:
            data_input = result[0].rename(columns={'x': 'x1'})
    data_x = results[1][0].rename(columns={'x': 'x2'})

    # prepare data for target interact plot
    data_x['fake_count'] = 1
    agg_dict = {}
    for t in target:
        agg_dict[t] = 'mean'
    agg_dict['fake_count'] = 'count'
    target_count_data = data_x.groupby(['x1', 'x2'], as_index=False).agg(agg_dict)

    # prepare summary data frame
    summary_df = target_count_data.rename(columns={'fake_count': 'count'})
    info_cols = ['x1', 'x2', 'display_column_1', 'display_column_2']
    for i in range(2):
        summary_df['display_column_%d' % (i + 1)] = summary_df['x%d' % (i + 1)].apply(lambda x: results[i][1][int(x)])
        if feature_types[i] == 'numeric':
            summary_df['value_lower_%d' % (i + 1)] = summary_df['x%d' % (i + 1)].apply(lambda x: results[i][2][int(x)])
            summary_df['value_upper_%d' % (i + 1)] = summary_df['x%d' % (i + 1)].apply(lambda x: results[i][3][int(x)])
            info_cols += ['value_lower_%d' % (i + 1), 'value_upper_%d' % (i + 1)]
        if len(results[i][4]) != 0:
            summary_df['percentile_column_%d' % (i + 1)] = summary_df['x%d' % (i + 1)].apply(lambda x: results[i][4][int(x)])
            summary_df['percentile_lower_%d' % (i + 1)] = summary_df['x%d' % (i + 1)].apply(lambda x: results[i][5][int(x)])
            summary_df['percentile_upper_%d' % (i + 1)] = summary_df['x%d' % (i + 1)].apply(lambda x: results[i][6][int(x)])
            info_cols += ['percentile_column_%d' % (i + 1), 'percentile_lower_%d' % (i + 1), 'percentile_upper_%d' % (i + 1)]
    summary_df = summary_df[info_cols + ['count'] + target]

    title = plot_params.get('title', 'Target plot for feature "%s"' % ' & '.join(feature_names))
    subtitle = plot_params.get('subtitle', 'Average target value through different feature value combinations.')

    # inner call target plot interact
    axes = _info_plot_interact(
        feature_names=feature_names, display_columns=[results[0][1], results[1][1]],
        percentile_columns=[results[0][4], results[1][4]], ys=target, plot_data=target_count_data,
        title=title, subtitle=subtitle, figsize=figsize, ncols=ncols, plot_params=plot_params)

    return axes, summary_df


def actual_plot_interact(model, X, features, feature_names, num_grid_points=None, grid_types=None,
                         percentile_ranges=None, grid_ranges=None, cust_grid_points=None, show_percentile=False,
                         show_outliers=False, which_classes=None, predict_kwds={}, ncols=2,
                         figsize=None, plot_params=None):
    # check model
    n_classes, classifier, predict = _check_model(model=model)

    # check input data set
    if type(X) != pd.core.frame.DataFrame:
        raise ValueError('X: only accept pandas DataFrame')

    num_grid_points = _expand_default(num_grid_points, 10)
    grid_types = _expand_default(grid_types, 'percentile')
    percentile_ranges = _expand_default(percentile_ranges, None)
    grid_ranges = _expand_default(grid_ranges, None)
    cust_grid_points = _expand_default(cust_grid_points, None)

    if not show_outliers:
        show_outliers = [False, False]
    else:
        show_outliers = [True, True]
        for i in range(2):
            if (percentile_ranges[i] is None) and (grid_ranges[i] is None) and (cust_grid_points[i] is None):
                show_outliers[i] = False

    # check features
    feature_types = [_check_feature(feature=features[0], df=X), _check_feature(feature=features[1], df=X)]

    # check percentile_range
    _check_percentile_range(percentile_range=percentile_ranges[0])
    _check_percentile_range(percentile_range=percentile_ranges[1])

    # prediction
    _X = X.copy()
    prediction = predict(_X, **predict_kwds)

    info_df = _X[_make_list(features[0]) + _make_list(features[1])]
    actual_prediction_columns = ['actual_prediction']
    if n_classes == 0:
        info_df['actual_prediction'] = prediction
    elif n_classes == 2:
        info_df['actual_prediction'] = prediction[:, 1]
    else:
        plot_classes = range(n_classes)
        if which_classes is not None:
            plot_classes = sorted(which_classes)

        actual_prediction_columns = []
        for class_idx in plot_classes:
            info_df['actual_prediction_%d' % class_idx] = prediction[:, class_idx]
            actual_prediction_columns.append('actual_prediction_%d' % class_idx)

    # (data_x, display_columns, percentile_columns)
    results = []
    data_input = info_df.copy()
    for i in range(2):
        result = _prepare_data_x(
            feature=features[i], feature_type=feature_types[i], data=data_input,
            num_grid_points=num_grid_points[i], grid_type=grid_types[i], percentile_range=percentile_ranges[i],
            grid_range=grid_ranges[i], cust_grid_points=cust_grid_points[i],
            show_percentile=show_percentile, show_outliers=show_outliers[i])
        results.append(result)
        if i == 0:
            data_input = result[0].rename(columns={'x': 'x1'})

    data_x = results[1][0].rename(columns={'x': 'x2'})
    data_x['fake_count'] = 1

    agg_dict = {}
    actual_prediction_columns_qs = []
    for idx in range(len(actual_prediction_columns)):
        agg_dict[actual_prediction_columns[idx]] = [q1, q2, q3]
        actual_prediction_columns_qs += [actual_prediction_columns[idx] + '_%s' % q for q in ['q1', 'q2', 'q3']]
    agg_dict['fake_count'] = 'count'
    actual_plot_data = data_x.groupby(['x1', 'x2'], as_index=False).agg(agg_dict)
    actual_plot_data.columns = ['_'.join(col) if col[1] != '' else col[0] for col in actual_plot_data.columns]
    actual_plot_data = actual_plot_data.rename(columns={'fake_count_count': 'fake_count'})
    summary_df = actual_plot_data.rename(columns={'fake_count': 'count'})

    info_cols = ['x1', 'x2', 'display_column_1', 'display_column_2']
    for i in range(2):
        summary_df['display_column_%d' % (i + 1)] = summary_df['x%d' % (i + 1)].apply(lambda x: results[i][1][int(x)])
        if len(results[i][2]) != 0:
            summary_df['percentile_column_%d' % (i + 1)] = summary_df['x%d' % (i + 1)].apply(lambda x: results[i][2][int(x)])
            info_cols.append('percentile_column_%d' % (i + 1))
    summary_df = summary_df[info_cols + ['count'] + actual_prediction_columns_qs]

    title = plot_params.get('title', 'Actual predictions plot for %s' % ' & '.join(feature_names))
    subtitle = plot_params.get('subtitle', 'Distribution of actual prediction through feature grids.')

    axes = _info_plot_interact(
        feature_names=feature_names, display_columns=[results[0][1], results[1][1]],
        percentile_columns=[results[0][2], results[1][2]], ys=actual_prediction_columns,
        plot_data=actual_plot_data, title=title, subtitle=subtitle, figsize=figsize,
        ncols=ncols, plot_params=plot_params)

    return axes, summary_df
