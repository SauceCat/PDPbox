
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

import copy
from pdp_plot_utils import _axis_modify
from pdp_calc_utils import _get_grids, _find_bucket, _make_bucket_column_names, _find_onehot_actual, _find_closest


def _actual_plot_title(feature_name, ax, figsize, multi_flag, which_class, plot_params):
    """
    Plot actual plot title

    :param feature_name: feature name
    :param ax: axes to plot on
    :param figsize: figure size
    :param multi_flag: whether it is a subplot of a multi-classes plot
    :param which_class: which class to plot
    :param plot_params: values of plot parameters
    """
    font_family = 'Arial'
    title = 'Actual predictions plot for %s' % feature_name
    subtitle = 'Each point is clustered to the closest grid point.'

    title_fontsize = 15
    subtitle_fontsize = 12

    if plot_params is not None:
        if 'font_family' in plot_params.keys():
            font_family = plot_params['font_family']
        if 'title' in plot_params.keys():
            title = plot_params['title']
        if 'title_fontsize' in plot_params.keys():
            title_fontsize = plot_params['title_fontsize']
        if 'subtitle_fontsize' in plot_params.keys():
            subtitle_fontsize = plot_params['subtitle_fontsize']

    ax.set_facecolor('white')
    if multi_flag:
        ax.text(0, 0.7, title, va="top", ha="left", fontsize=title_fontsize, fontname=font_family)
        ax.text(0, 0.45, "For Class %d" % which_class, va="top", ha="left", fontsize=subtitle_fontsize,
                fontname=font_family)
        ax.text(0, 0.25, subtitle, va="top", ha="left", fontsize=subtitle_fontsize, fontname=font_family, color='grey')
    else:
        ax.text(0, 0.7, title, va="top", ha="left", fontsize=title_fontsize, fontname=font_family)
        ax.text(0, 0.4, subtitle, va="top", ha="left", fontsize=subtitle_fontsize, fontname=font_family, color='grey')
    ax.axis('off')


def _actual_plot(pdp_isolate_out, feature_name, figwidth, plot_params, outer):
    """
    Plot actual prediction distribution

    :param pdp_isolate_out: instance of pdp_isolate_obj
        a calculated pdp_isolate_obj instance
    :param feature_name: feature name
    :param figwidth: figure width
    :param plot_params: values of plot parameters
    :param outer: outer GridSpec
    """

    try:
        import seaborn as sns
    except:
        raise RuntimeError('seaborn is necessary for the actual plot.')

    if outer is None:
        plt.figure(figsize=(figwidth, figwidth / 1.6))
        gs = GridSpec(2, 1)
        ax1 = plt.subplot(gs[0])
        ax2 = plt.subplot(gs[1], sharex=ax1)
    else:
        inner = GridSpecFromSubplotSpec(2, 1, subplot_spec=outer, wspace=0, hspace=0)
        ax1 = plt.subplot(inner[0])
        ax2 = plt.subplot(inner[1], sharex=ax1)

    font_family = 'Arial'
    boxcolor = '#66C2D7'
    linecolor = '#1A4E5D'
    barcolor = '#5BB573'
    xticks_rotation = 0

    if plot_params is not None:
        if 'font_family' in plot_params.keys():
            font_family = plot_params['font_family']
        if 'boxcolor' in plot_params.keys():
            boxcolor = plot_params['boxcolor']
        if 'linecolor' in plot_params.keys():
            linecolor = plot_params['linecolor']
        if 'barcolor' in plot_params.keys():
            barcolor = plot_params['barcolor']
        if 'xticks_rotation' in plot_params.keys():
            xticks_rotation = plot_params['xticks_rotation']

    _axis_modify(font_family, ax1)
    _axis_modify(font_family, ax2)

    df = copy.deepcopy(pdp_isolate_out.ice_lines)
    actual_columns = pdp_isolate_out.actual_columns
    feature_grids = pdp_isolate_out.feature_grids
    df = df[actual_columns + ['actual_preds']]

    if pdp_isolate_out.feature_type == 'binary':
        df['x'] = df[actual_columns[0]]
    elif pdp_isolate_out.feature_type == 'onehot':
        df['x'] = df[actual_columns].apply(lambda x: _find_onehot_actual(x), axis=1)
        df = df[df['x'].isnull() == False].reset_index(drop=True)
    else:
        df = df[(df[actual_columns[0]] >= feature_grids[0])
                & (df[actual_columns[0]] <= feature_grids[-1])].reset_index(drop=True)
        df['x'] = df[actual_columns[0]].apply(lambda x: _find_closest(x, feature_grids))

    pred_median_gp = df.groupby('x', as_index=False).agg({'actual_preds': 'median'})
    pred_count_gp = df.groupby('x', as_index=False).agg({'actual_preds': 'count'})
    pred_mean_gp = df.groupby('x', as_index=False).agg({'actual_preds': 'mean'}).rename(columns={'actual_preds': 'preds_mean'})
    pred_std_gp = df.groupby('x', as_index=False).agg({'actual_preds': 'std'}).rename(columns={'actual_preds': 'preds_std'})

    pred_outlier_gp = pred_mean_gp.merge(pred_std_gp, on='x', how='left')
    pred_outlier_gp['outlier_upper'] = pred_outlier_gp['preds_mean'] + 3 * pred_outlier_gp['preds_std']
    pred_outlier_gp['outlier_lower'] = pred_outlier_gp['preds_mean'] - 3 * pred_outlier_gp['preds_std']

    # draw boxplot: prediction distribution
    boxwith = np.min([0.5, 0.5 / (10.0 / len(feature_grids))])
    sns.boxplot(x=df['x'], y=df['actual_preds'], width=boxwith, ax=ax1, color=boxcolor, linewidth=1, saturation=1)
    sns.pointplot(x=pred_median_gp['x'], y=pred_median_gp['actual_preds'], ax=ax1, color=linecolor)
    ax1.set_xlabel('')
    ax1.set_ylabel('actual_preds')
    ax1.set_ylim(pred_outlier_gp['outlier_lower'].min(), pred_outlier_gp['outlier_upper'].max())

    # draw bar plot
    rects = ax2.bar(pred_count_gp['x'], pred_count_gp['actual_preds'], width=boxwith, color=barcolor, alpha=0.5)
    ax2.set_xlabel(feature_name)
    ax2.set_ylabel('count')
    plt.xticks(range(len(feature_grids)), pdp_isolate_out.feature_grids, rotation=xticks_rotation)

    _autolabel(rects, ax2, barcolor)


def _autolabel(rects, ax, bar_color):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        bbox_props = {'facecolor': 'white', 'edgecolor': bar_color, 'boxstyle': "square,pad=0.5"}
        ax.text(rect.get_x() + rect.get_width() / 2., height, '%d' % int(height),
                ha='center', va='center', bbox=bbox_props, color=bar_color)


def _prepare_data_x(feature, feature_type, data, num_grid_points, grid_type, percentile_range,
                    grid_range, cust_grid_points):
    display_columns = []
    data_x = data.copy()
    if feature_type == 'binary':
        feature_grids = display_columns = np.array([0, 1])
        data_x['x'] = data_x[feature]
    if feature_type == 'numeric':
        percentile_info = None
        if cust_grid_points is None:
            feature_grids, percentile_info = _get_grids(data_x[feature], num_grid_points, grid_type, percentile_range, grid_range)
        else:
            feature_grids = np.array(sorted(cust_grid_points))

        data_x = data_x[(data_x[feature] >= feature_grids[0])
                        & (data_x[feature] <= feature_grids[-1])].reset_index(drop=True)
        data_x['x'] = data_x[feature].apply(lambda x: _find_bucket(x, feature_grids))
        display_columns = _make_bucket_column_names(feature_grids, percentile_info)

    if feature_type == 'onehot':
        feature_grids = display_columns = np.array(feature)
        data_x['x'] = data_x[feature].apply(lambda x: _find_onehot_actual(x), axis=1)
        data_x = data_x[~data_x['x'].isnull()].reset_index(drop=True)

    return data_x, display_columns


def _target_plot_title(feature_name, ax, plot_params):
    """
    Draw target plot title

    :param feature_name: feature name
    :param ax: axes to plot on
    :param plot_params: values of plot parameters
    """

    if plot_params is None:
        plot_params = dict()

    font_family = plot_params.get('font_family', 'Arial')
    title = plot_params.get('title', 'Target plot for feature "%s"' % feature_name)
    subtitle = plot_params.get('subtitle', 'Average target values through feature grids.')
    title_fontsize = plot_params.get('title_fontsize', 15)
    subtitle_fontsize = plot_params.get('subtitle_fontsize', 12)

    ax.set_facecolor('white')
    ax.text(0, 0.7, title, va="top", ha="left", fontsize=title_fontsize, fontname=font_family)
    ax.text(0, 0.4, subtitle, va="top", ha="left", fontsize=subtitle_fontsize, fontname=font_family, color='grey')
    ax.axis('off')


def _target_plot(feature_name, display_columns, target, bar_data, target_lines, figsize, plot_params):
    """
    Plot target distribution through feature grids

    :param feature_name: feature name
    :param display_columns: column names to display
    :param target: target columns
    :param bar_data: bar counts data
    :param target_lines: target lines data
    :param figsize: figure size
    :param plot_params: values of plot parameters
    """

    width, height = 16, 7
    if figsize is not None:
        width, height = figsize

    if plot_params is None:
        plot_params = dict()

    font_family = plot_params.get('font_family', 'Arial')
    line_color = plot_params.get('line_color', '#1A4E5D')
    bar_color = plot_params.get('bar_color', '#5BB573')
    line_width = plot_params.get('line_width', 1)
    xticks_rotation = plot_params.get('xticks_rotation', 0)

    # graph title
    plt.figure(figsize=(width, width / 6.7))
    title_ax = plt.subplot(111)
    _target_plot_title(feature_name=feature_name, ax=title_ax, plot_params=plot_params)

    # bar plot
    plt.figure(figsize=(width, height))
    bar_ax = plt.subplot(111)
    box_width = np.min([0.5, 0.5 / (10.0 / len(display_columns))])

    rects = bar_ax.bar(bar_data['x'], bar_data['fake_count'], width=box_width, color=bar_color, alpha=0.5)
    bar_ax.set_xlabel(feature_name)
    bar_ax.set_ylabel('count')

    plt.xticks(range(len(display_columns)), display_columns, rotation=xticks_rotation)
    plt.xlim(-0.5, len(display_columns) - 0.5)
    _autolabel(rects, bar_ax, bar_color)
    _axis_modify(font_family, bar_ax)

    # target lines
    line_ax = bar_ax.twinx()
    if len(target_lines) == 1:
        target_line = target_lines[0].sort_values('x', ascending=True).set_index('x')
        line_ax.plot(target_line.index.values, target_line[target], linewidth=line_width, c=line_color, marker='o')
        for idx in target_line.index.values:
            bbox_props = {'facecolor': line_color, 'edgecolor': 'none', 'boxstyle': "square,pad=0.5"}
            line_ax.text(idx, target_line.loc[idx, target], '%.3f' % target_line.loc[idx, target],
                         ha="center", va="bottom", size=10, bbox=bbox_props, color='#ffffff')
    else:
        line_colors = plt.get_cmap('tab20')(range(20))
        for target_idx in range(len(target)):
            line_color = line_colors[target_idx]
            target_line = target_lines[target_idx].sort_values('x', ascending=True).set_index('x')
            line_ax.plot(target_line.index.values, target_line[target[target_idx]], linewidth=line_width,
                         c=line_color, marker='o', label=target[target_idx])
            for idx in target_line.index.values:
                bbox_props = {'facecolor': line_color, 'edgecolor': 'none', 'boxstyle': "square,pad=0.5"}
                line_ax.text(idx, target_line.loc[idx, target[target_idx]],
                             '%.3f' % target_line.loc[idx, target[target_idx]],
                             ha="center", va="top", size=10, bbox=bbox_props, color='#ffffff')
            plt.legend(loc="upper left", ncol=5, bbox_to_anchor=(0, 1.2), frameon=False)

    _axis_modify(font_family, line_ax)
    line_ax.get_yaxis().tick_right()
    line_ax.grid(False)
    line_ax.set_ylabel('target_avg')

