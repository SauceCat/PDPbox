from __future__ import absolute_import

import numpy as np
import pandas as pd
import psutil


def _check_feature(feature, df):
    """Make sure feature exists and infer feature type

    Feature types
    -------------
    1. binary
    2. onehot
    3. numeric
    """
    if type(feature) == str:
        if feature not in df.columns.values:
            raise ValueError('feature does not exist: %s' % feature)
        if sorted(list(np.unique(df[feature]))) == [0, 1]:
            feature_type = 'binary'
        else:
            feature_type = 'numeric'
    elif type(feature) == list:
        if len(feature) < 2:
            raise ValueError('one-hot encoding feature should contain more than 1 element')
        if not set(feature) < set(df.columns.values):
            raise ValueError('feature does not exist: %s' % str(feature))
        feature_type = 'onehot'
    else:
        raise ValueError('feature: please pass a string or a list (for onehot encoding feature)')

    return feature_type


def _check_percentile_range(percentile_range):
    """Make sure percentile range is valid"""
    if percentile_range is not None:
        if len(percentile_range) != 2:
            raise ValueError('percentile_range: should contain 2 elements')
        if np.max(percentile_range) > 100 or np.min(percentile_range) < 0:
            raise ValueError('percentile_range: should be between 0 and 100')


def _check_target(target, df):
    if type(target) == str:
        if target not in df.columns.values:
            raise ValueError('target does not exist: %s' % target)
        if sorted(list(np.unique(df[target]))) == [0, 1]:
            target_type = 'binary'
        else:
            target_type = 'regression'
    elif type(target) == list:
        if len(target) < 2:
            raise ValueError('multi-class target should contain more than 1 element')
        if not set(target) < set(df.columns.values):
            raise ValueError('target does not exist: %s' % (str(target)))
        for target_idx in range(len(target)):
            if sorted(list(np.unique(df[target[target_idx]]))) != [0, 1]:
                raise ValueError('multi-class targets should be one-hot encoded: %s' % (str(target[target_idx])))
        target_type = 'multi-class'
    else:
        raise ValueError('target: please pass a string or a list (for multi-class targets)')

    return target_type


def _check_dataset(df):
    """Make sure input dataset is pandas DataFrame"""
    if type(df) != pd.core.frame.DataFrame:
        raise ValueError('only accept pandas DataFrame')


def _make_list(x):
    """Make list when it is necessary"""
    if type(x) == list:
        return x
    return [x]


def _expand_default(x, default):
    if x is None:
        return [default] * 2
    return x


def _check_model(model):
    """Check model input, return class information and predict function"""
    try:
        n_classes = len(model.classes_)
        predict = model.predict_proba
    except:
        n_classes = 0
        predict = model.predict

    return n_classes, predict


def _check_grid_type(grid_type):
    """Make sure grid type is percentile or equal"""
    if grid_type not in ['percentile', 'equal']:
        raise ValueError('grid_type should be "percentile" or "equal".')


def _check_classes(classes_list, n_classes):
    if len(classes_list) > 0 and n_classes > 2:
        if np.min(classes_list) < 0:
            raise ValueError('class index should be >= 0.')
        if np.max(classes_list) > n_classes - 1:
            raise ValueError('class index should be < n_classes.')


def _check_info_plot_params(df, feature, grid_type, percentile_range, grid_range,
                            cust_grid_points, show_outliers):
    _check_dataset(df=df)
    feature_type = _check_feature(feature=feature, df=df)
    _check_grid_type(grid_type=grid_type)
    _check_percentile_range(percentile_range=percentile_range)

    # show_outliers should be only turned on when necessary
    if (percentile_range is None) and (grid_range is None) and (cust_grid_points is None):
        show_outliers = False
    return feature_type, show_outliers


def _check_info_plot_interact_params(num_grid_points, grid_types, percentile_ranges, grid_ranges, cust_grid_points,
                                     show_outliers, plot_params, features, df):
    _check_dataset(df=df)
    num_grid_points = _expand_default(num_grid_points, 10)
    grid_types = _expand_default(grid_types, 'percentile')
    _check_grid_type(grid_type=grid_types[0])
    _check_grid_type(grid_type=grid_types[1])

    percentile_ranges = _expand_default(percentile_ranges, None)
    _check_percentile_range(percentile_range=percentile_ranges[0])
    _check_percentile_range(percentile_range=percentile_ranges[1])

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

    # check features
    feature_types = [_check_feature(feature=features[0], df=df), _check_feature(feature=features[1], df=df)]

    return {
        'num_grid_points': num_grid_points,
        'grid_types': grid_types,
        'percentile_ranges': percentile_ranges,
        'grid_ranges': grid_ranges,
        'cust_grid_points': cust_grid_points,
        'show_outliers': show_outliers,
        'plot_params': plot_params,
        'feature_types': feature_types
    }


def _check_memory_limit(memory_limit):
    """Make sure memory limit is between 0 and 1"""
    if memory_limit <= 0 or memory_limit >= 1:
        raise ValueError('memory_limit: should be (0, 1)')


def _check_frac_to_plot(frac_to_plot):
    """Make sure frac_to_plot is between 0 and 1 if it is float"""
    if type(frac_to_plot) == float:
        if (frac_to_plot <= 0.0) or (frac_to_plot > 1.0):
            raise ValueError('frac_to_plot: should in range(0, 1) when it is a float')


def _plot_title(title, subtitle, title_ax, plot_params):
    """Add plot title."""

    title_params = {'fontname': plot_params.get('font_family', 'Arial'), 'x': 0, 'va': 'top', 'ha': 'left'}
    title_fontsize = plot_params.get('title_fontsize', 15)
    subtitle_fontsize = plot_params.get('subtitle_fontsize', 12)

    title_ax.set_facecolor('white')
    title_ax.text(y=0.7, s=title, fontsize=title_fontsize, **title_params)
    title_ax.text(y=0.5, s=subtitle, fontsize=subtitle_fontsize, color='grey', **title_params)
    title_ax.axis('off')


def _calc_memory_usage(df, total_units, n_jobs, memory_limit):
    """Calculate n_jobs to use"""
    unit_memory = df.memory_usage(deep=True).sum()
    free_memory = psutil.virtual_memory()[1] * memory_limit
    num_units = int(np.floor(free_memory / unit_memory))
    true_n_jobs = np.min([num_units, n_jobs, total_units])

    return true_n_jobs


def _axes_modify(font_family, ax, top=False, right=False, grid=False):
    """Modify matplotlib Axes

    Parameters
    ----------
    top: bool, default=False
        xticks location=top
    right: bool, default=False
        yticks, location=right
    grid: bool, default=False
        whether it is for grid plot
    """

    ax.set_facecolor('white')
    ax.tick_params(axis='both', which='major', labelsize=10, labelcolor='#424242', colors='#9E9E9E')

    for tick in ax.get_xticklabels():
        tick.set_fontname(font_family)
    for tick in ax.get_yticklabels():
        tick.set_fontname(font_family)

    ax.set_frame_on(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    if top:
        ax.get_xaxis().tick_top()
    if right:
        ax.get_yaxis().tick_right()
    if not grid:
        ax.grid(True, 'major', 'x', ls='--', lw=.5, c='k', alpha=.3)
        ax.grid(True, 'major', 'y', ls='--', lw=.5, c='k', alpha=.3)


def _modify_legend_ax(ax, font_family):
    """Modify legend like Axes"""
    ax.set_frame_on(False)

    for tick in ax.get_xticklabels():
        tick.set_fontname(font_family)
    for tick in ax.get_yticklabels():
        tick.set_fontname(font_family)

    ax.set_facecolor('white')
    ax.set_xticks([])
    ax.set_yticks([])


def _get_grids(feature_values, num_grid_points, grid_type, percentile_range, grid_range):
    """Calculate grid points for numeric feature

    Parameters
    ----------
    feature_values: Series, 1d-array, or list.
        values to calculate grid points
    num_grid_points: integer
        number of grid points for numeric feature
    grid_type: string
        'percentile' or 'equal',
        type of grid points for numeric feature
    percentile_range: tuple or None
        percentile range to investigate,
        for numeric feature when grid_type='percentile'
    grid_range: tuple or None
        value range to investigate,
        for numeric feature when grid_type='equal'

    Returns
    -------
    feature_grids: 1d-array
        calculated grid points
    percentile_info: 1d-array or []
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
        value_grids = np.percentile(feature_values, percentile_grids)

        grids_df = pd.DataFrame()
        grids_df['percentile_grids'] = [round(v, 2) for v in percentile_grids]
        grids_df['value_grids'] = value_grids
        grids_df = grids_df.groupby(['value_grids'], as_index=False).agg(
            {'percentile_grids': lambda v: str(tuple(v)).replace(',)', ')')}).sort_values('value_grids', ascending=True)

        feature_grids, percentile_info = grids_df['value_grids'].values, grids_df['percentile_grids'].values
    else:
        if grid_range is not None:
            value_grids = np.linspace(np.min(grid_range), np.max(grid_range), num_grid_points)
        else:
            value_grids = np.linspace(np.min(feature_values), np.max(feature_values), num_grid_points)
        feature_grids, percentile_info = value_grids, []

    return feature_grids, percentile_info


def _sample_data(ice_lines, frac_to_plot):
    """Get sample ice lines to plot"""

    if frac_to_plot < 1.:
        ice_plot_data = ice_lines.sample(int(ice_lines.shape[0] * frac_to_plot))
    elif frac_to_plot > 1:
        ice_plot_data = ice_lines.sample(frac_to_plot)
    else:
        ice_plot_data = ice_lines.copy()

    ice_plot_data = ice_plot_data.reset_index(drop=True)
    return ice_plot_data


