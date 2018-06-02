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
    """Check and return target type

    target types
    -------------
    1. binary
    2. multi-class
    3. regression
    """

    if type(target) == str:
        if target not in df.columns.values:
            raise ValueError('target does not exist: %s' % target)
        if sorted(list(np.unique(df[target]))) == [0, 1]:
            target_type = 'binary'
        else:
            target_type = 'regression'
    elif type(target) == list:
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
    """Create a list of default values"""
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
    """Makre sure classes list is valid"""
    if len(classes_list) > 0 and n_classes > 2:
        if np.min(classes_list) < 0:
            raise ValueError('class index should be >= 0.')
        if np.max(classes_list) > n_classes - 1:
            raise ValueError('class index should be < n_classes.')


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


def _get_grid_combos(feature_grids, feature_types):
    """Calculate grid combinations of two grid lists"""

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


def _find_onehot_actual(x):
    """Map one-hot value to one-hot name"""
    try:
        value = list(x).index(1)
    except:
        value = np.nan
    return value


def _find_bucket(x, feature_grids, endpoint):
    """Find bucket that x falls in"""
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
    """Create bucket names based on feature grids"""
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
    """Create bucket names based on percentile info"""
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


def _calc_figsize(num_charts, ncols, title_height, unit_figsize):
    """Calculate figure size"""
    if num_charts > 1:
        nrows = int(np.ceil(num_charts * 1.0 / ncols))
        ncols = np.min([num_charts, ncols])
        width = np.min([unit_figsize[0] * ncols, 15])
        height = np.min([width * 1.0 / ncols, unit_figsize[1]]) * nrows
    else:
        width, height, nrows = unit_figsize[0], unit_figsize[1] + title_height, 1

    return width, height, nrows, ncols


