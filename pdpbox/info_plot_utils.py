
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1 import make_axes_locatable

import copy
from .pdp_plot_utils import _axes_modify
from .pdp_calc_utils import (_get_grids, _find_bucket, _make_bucket_column_names, _find_onehot_actual,
                             _find_closest, _make_bucket_column_names_percentile)


def _actual_plot_title(feature_name, ax, plot_params):
    font_family = 'Arial'
    title = 'Actual predictions plot for %s' % feature_name
    # subtitle = 'Each point is clustered to the closest grid point.'

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
    ax.text(0, 0.7, title, va="top", ha="left", fontsize=title_fontsize, fontname=font_family)
    # ax.text(0, 0.4, subtitle, va="top", ha="left", fontsize=subtitle_fontsize, fontname=font_family, color='grey')
    ax.axis('off')


def _draw_boxplot(box_data, box_ax, plot_params):
    font_family = plot_params.get('font_family', 'Arial')
    xs = sorted(box_data['x'].unique())
    ys = []
    for x in xs:
        ys.append(box_data[box_data['x'] == x]['y'].values)

    box_ax.boxplot(ys, positions=xs, showfliers=False)
    _axes_modify(font_family=font_family, ax=box_ax)


def _draw_box_bar(bar_data, bar_ax, box_data, box_ax,
                  feature_name, display_columns, percentile_columns, plot_params):

    font_family = plot_params.get('font_family', 'Arial')
    xticks_rotation = plot_params.get('xticks_rotation', 0)

    _draw_boxplot(box_data=box_data, box_ax=box_ax, plot_params=plot_params)
    box_ax.set_ylabel('actual_prediction')

    _draw_barplot(bar_data=bar_data, bar_ax=bar_ax, display_columns=display_columns, plot_params=plot_params)

    # bar plot
    bar_ax.set_xlabel(feature_name)
    bar_ax.set_ylabel('count')

    bar_ax.set_xticks(range(len(display_columns)))
    bar_ax.set_xticklabels(display_columns, rotation=xticks_rotation)
    bar_ax.set_xlim(-0.5, len(display_columns) - 0.5)

    # display percentile
    if len(percentile_columns) > 0:
        percentile_ax = box_ax.twiny()
        percentile_ax.set_xticks(box_ax.get_xticks())
        percentile_ax.set_xbound(box_ax.get_xbound())
        percentile_ax.set_xticklabels(percentile_columns, rotation=xticks_rotation)
        percentile_ax.set_xlabel('percentile buckets')
        _axes_modify(font_family=font_family, ax=percentile_ax, top=True)


def _actual_plot_new(plot_data, actual_prediction_columns, feature_name, display_columns, percentile_columns,
                     figsize, ncols, plot_params):
    # set up graph parameters
    width, height = 16, 7
    nrows = 1

    if plot_params is None:
        plot_params = dict()

    font_family = plot_params.get('font_family', 'Arial')

    if len(actual_prediction_columns) > 1:
        nrows = int(np.ceil(len(actual_prediction_columns) * 1.0 / ncols))
        ncols = np.min([len(actual_prediction_columns), ncols])
        width = np.min([8 * len(actual_prediction_columns), 16])
        height = width * 1.0 / ncols * nrows

    if figsize is not None:
        width, height = figsize

    if plot_params is None:
        plot_params = dict()

    # Axes for title
    plt.figure(figsize=(width, 2))
    title_ax = plt.subplot(111)
    _actual_plot_title(feature_name=feature_name, ax=title_ax, plot_params=plot_params)

    plot_data['fake_count'] = 1
    bar_data = plot_data.groupby('x', as_index=False).agg({'fake_count': 'count'}).sort_values('x', ascending=True)

    if len(actual_prediction_columns) == 1:
        plt.figure(figsize=(width, height))
        gs = GridSpec(2, 1)
        box_ax = plt.subplot(gs[0])
        bar_ax = plt.subplot(gs[1], sharex=box_ax)

        box_data = plot_data[['x', actual_prediction_columns[0]]].rename(columns={actual_prediction_columns[0]: 'y'})
        _draw_box_bar(bar_data=bar_data, bar_ax=bar_ax, box_data=box_data, box_ax=box_ax,
                      feature_name=feature_name, display_columns=display_columns,
                      percentile_columns=percentile_columns, plot_params=plot_params)
    else:
        plt.figure(figsize=(width, height))
        outer = GridSpec(nrows, ncols, wspace=0.2, hspace=0.2)

        box_ax = []
        bar_ax = []
        for idx in range(len(actual_prediction_columns)):
            inner = GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[idx], wspace=0, hspace=0)
            inner_box_ax = plt.subplot(inner[0])
            inner_bar_ax = plt.subplot(inner[1], sharex=inner_box_ax)

            inner_box_data = plot_data[['x', actual_prediction_columns[idx]]].rename(
                columns={actual_prediction_columns[0]: 'y'})
            _draw_box_bar(bar_data=bar_data, bar_ax=inner_bar_ax, box_data=inner_box_data, box_ax=inner_box_ax,
                          feature_name=feature_name, display_columns=display_columns,
                          percentile_columns=percentile_columns, plot_params=plot_params)

            subplot_title = 'target_%s' % actual_prediction_columns[idx].split('_')[-1]
            if len(percentile_columns) > 0:
                subplot_title += '\n\n\n'
            inner_box_ax.set_title(subplot_title, fontdict={'fontsize': 12, 'fontname': font_family})

            box_ax.append(inner_box_ax)
            bar_ax.append(inner_bar_ax)


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

    _axes_modify(font_family, ax1)
    _axes_modify(font_family, ax2)

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
    for rect in rects:
        height = rect.get_height()
        bbox_props = {'facecolor': 'white', 'edgecolor': bar_color, 'boxstyle': "square,pad=0.5"}
        ax.text(rect.get_x() + rect.get_width() / 2., height, '%d' % int(height),
                ha='center', va='center', bbox=bbox_props, color=bar_color)


def _prepare_data_x(feature, feature_type, data, num_grid_points, grid_type, percentile_range,
                    grid_range, cust_grid_points, show_percentile, show_outliers):
    """map feature values to grid points

    Parameters:
    -----------

    :param feature: string or list
        feature or feature list to investigate
        for one-hot encoding features, feature list is required
    :param feature_type: string
        'binary', 'numeric' or 'onehot'
        feature type
    :param data: pandas DataFrame
        data set to investigate on
        only contains useful columns
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
    :param cust_grid_points: Series, 1d-array, list or None
        customized list of grid points
        for numeric feature
    :param show_percentile: bool
        whether to display the percentile buckets
        for numeric feature when grid_type='percentile'
    :param show_outliers: bool
        whether to display the out of range buckets
        for numeric feature when percentile_range or grid_range is not None

    Returns:
    --------

    :return data_x: pandas DataFrame
        data with 'x' column, indicating the mapped grid point
    :return display_columns: list
        list of xticklabels
    :return percentile_columns: list
        list of xticklabels in percentile format
    """

    display_columns = []
    percentile_columns = []
    data_x = data.copy()

    if feature_type == 'binary':
        feature_grids = np.array([0, 1])
        display_columns = ['%s_0' % feature, '%s_1' % feature]
        data_x['x'] = data_x[feature]
    if feature_type == 'numeric':
        percentile_info = None
        if cust_grid_points is None:
            feature_grids, percentile_info = _get_grids(
                x=data_x[feature].values, num_grid_points=num_grid_points, grid_type=grid_type,
                percentile_range=percentile_range, grid_range=grid_range)
        else:
            feature_grids = np.array(sorted(cust_grid_points))

        if not show_outliers:
            data_x = data_x[(data_x[feature] >= feature_grids[0])
                            & (data_x[feature] <= feature_grids[-1])].reset_index(drop=True)

        # map feature value into value buckets
        data_x['x'] = data_x[feature].apply(lambda x: _find_bucket(x=x, feature_grids=feature_grids))
        uni_xs = sorted(data_x['x'].unique())

        # create bucket names
        display_columns = _make_bucket_column_names(feature_grids=feature_grids)
        display_columns = np.array(display_columns)[range(uni_xs[0], uni_xs[-1]+1)]

        # create percentile bucket names
        if show_percentile and grid_type == 'percentile':
            percentile_columns = _make_bucket_column_names_percentile(percentile_info=percentile_info)
            percentile_columns = np.array(percentile_columns)[range(uni_xs[0], uni_xs[-1]+1)]

        # adjust results
        data_x['x'] = data_x['x'] - data_x['x'].min()

    if feature_type == 'onehot':
        feature_grids = display_columns = np.array(feature)
        data_x['x'] = data_x[feature].apply(lambda x: _find_onehot_actual(x=x), axis=1)
        data_x = data_x[~data_x['x'].isnull()].reset_index(drop=True)

    data_x['x'] = data_x['x'].map(int)

    return data_x, display_columns, percentile_columns


def _target_plot_title(feature_name, ax, plot_params):
    font_family = plot_params.get('font_family', 'Arial')
    title = plot_params.get('title', 'Target plot for feature "%s"' % feature_name)
    subtitle = plot_params.get('subtitle', 'Average target values through feature grids.')
    title_fontsize = plot_params.get('title_fontsize', 15)
    subtitle_fontsize = plot_params.get('subtitle_fontsize', 12)

    ax.set_facecolor('white')
    ax.text(0, 0.7, title, va="top", ha="left", fontsize=title_fontsize, fontname=font_family)
    ax.text(0, 0.5, subtitle, va="top", ha="left", fontsize=subtitle_fontsize, fontname=font_family, color='grey')
    ax.axis('off')


def _draw_barplot(bar_data, bar_ax, display_columns, plot_params):

    font_family = plot_params.get('font_family', 'Arial')
    bar_color = plot_params.get('bar_color', '#5BB573')
    bar_width = plot_params.get('bar_width', np.min([0.4, 0.4 / (10.0 / len(display_columns))]))

    # add value label for bar plot
    rects = bar_ax.bar(x=bar_data['x'], height=bar_data['fake_count'], width=bar_width, color=bar_color, alpha=0.5)
    _autolabel(rects=rects, ax=bar_ax, bar_color=bar_color)
    _axes_modify(font_family=font_family, ax=bar_ax)


def _draw_lineplot(line_data, line_ax, line_color, plot_params):

    font_family = plot_params.get('font_family', 'Arial')
    line_width = plot_params.get('line_width', 1)

    line_ax.plot(line_data['x'], line_data['y'], linewidth=line_width, c=line_color, marker='o')
    for idx in line_data.index.values:
        bbox_props = {'facecolor': line_color, 'edgecolor': 'none', 'boxstyle': "square,pad=0.5"}
        line_ax.text(line_data.loc[idx, 'x'], line_data.loc[idx, 'y'],
                     '%.3f' % line_data.loc[idx, 'y'],
                     ha="center", va="top", size=10, bbox=bbox_props, color='#ffffff')

    _axes_modify(font_family=font_family, ax=line_ax)


def _draw_bar_line(bar_data, bar_ax, line_data, line_ax, line_color,
                   feature_name, display_columns, percentile_columns, plot_params):

    font_family = plot_params.get('font_family', 'Arial')
    xticks_rotation = plot_params.get('xticks_rotation', 0)

    _draw_barplot(bar_data=bar_data, bar_ax=bar_ax, display_columns=display_columns, plot_params=plot_params)

    # bar plot
    bar_ax.set_xlabel(feature_name)
    bar_ax.set_ylabel('count')

    bar_ax.set_xticks(range(len(display_columns)))
    bar_ax.set_xticklabels(display_columns, rotation=xticks_rotation)
    bar_ax.set_xlim(-0.5, len(display_columns) - 0.5)

    # display percentile
    if len(percentile_columns) > 0:
        percentile_ax = bar_ax.twiny()
        percentile_ax.set_xticks(bar_ax.get_xticks())
        percentile_ax.set_xbound(bar_ax.get_xbound())
        percentile_ax.set_xticklabels(percentile_columns, rotation=xticks_rotation)
        percentile_ax.set_xlabel('percentile buckets')
        _axes_modify(font_family=font_family, ax=percentile_ax, top=True)

    _draw_lineplot(line_data=line_data, line_ax=line_ax, line_color=line_color, plot_params=plot_params)
    line_ax.get_yaxis().tick_right()
    line_ax.grid(False)
    line_ax.set_ylabel('target_avg')


def _target_plot(feature_name, display_columns, percentile_columns, target, bar_data,
                 target_lines, figsize, ncols, plot_params):
    """inner call for function target_plot

    Parameters:
    -----------

    :param feature_name: string
        name of the feature, not necessary a column name
    :param display_columns: list
        list of xticklabels
    :param percentile_columns: list
        list of xticklabels in percentile format
    :param target: list
        list of target columns to investigate
        if it is not multi-classes, the list would only contain 1 element
    :param bar_data: pandas DataFrame
        data for bar plot
    :param target_lines: list of pandas DataFrame
        data for target lines
    :param figsize: tuple or None
        size of the figure, (width, height)
    :param ncols: integer, optional, default=2
        number subplot columns, used when it is multi-class problem
    :param plot_params: dict or None
        parameters for the plot

    Returns:
    --------

    :return: list of matplotlib Axes
        [axes for title, axes for bar, axes for line]
    """

    # set up graph parameters
    width, height = 16, 7
    nrows = 1

    if len(target) > 1:
        nrows = int(np.ceil(len(target) * 1.0 / ncols))
        ncols = np.min([len(target), ncols])
        width = np.min([8 * len(target), 16])
        height = width * 1.0 / ncols * nrows

    if figsize is not None:
        width, height = figsize

    if plot_params is None:
        plot_params = dict()

    font_family = plot_params.get('font_family', 'Arial')
    line_color = plot_params.get('line_color', '#1A4E5D')
    line_colors_cmap = plot_params.get('line_colors_cmap', 'tab20')
    line_colors = plot_params.get('line_colors', plt.get_cmap(line_colors_cmap)(range(20)))

    # Axes for title
    plt.figure(figsize=(width, 2))
    title_ax = plt.subplot(111)
    _target_plot_title(feature_name=feature_name, ax=title_ax, plot_params=plot_params)

    if len(target) == 1:
        plt.figure(figsize=(width, height))
        bar_ax = plt.subplot(111)
        line_ax = bar_ax.twinx()

        line_data = target_lines[0].rename(columns={target[0]: 'y'}).sort_values('x', ascending=True)
        _draw_bar_line(bar_data=bar_data, bar_ax=bar_ax, line_data=line_data, line_ax=line_ax, line_color=line_color,
                       feature_name=feature_name, display_columns=display_columns,
                       percentile_columns=percentile_columns, plot_params=plot_params)

    else:
        _, plot_axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(width, height), sharex='all', sharey='all')
        plot_axes = plot_axes.flatten()
        bar_ax = []
        line_ax = []

        # get max average target value
        ys = []
        for target_idx in range(len(target)):
            ys += list(target_lines[target_idx][target[target_idx]].values)
        y_max = np.max(ys)

        for target_idx in range(len(target)):
            inner_line_color = line_colors[target_idx % len(line_colors)]
            inner_bar_ax = plot_axes[target_idx]
            inner_line_ax = inner_bar_ax.twinx()

            line_data = target_lines[target_idx].rename(
                columns={target[target_idx]: 'y'}).sort_values('x', ascending=True)

            _draw_bar_line(bar_data=bar_data, bar_ax=inner_bar_ax, line_data=line_data, line_ax=inner_line_ax,
                           line_color=inner_line_color, feature_name=feature_name, display_columns=display_columns,
                           percentile_columns=percentile_columns, plot_params=plot_params)

            subplot_title = target[target_idx]
            if len(percentile_columns) > 0:
                subplot_title += '\n\n\n'
            plot_axes[target_idx].set_title(subplot_title, fontdict={'fontsize': 12, 'fontname': font_family})

            inner_line_ax.set_ylim(0., y_max)
            bar_ax.append(inner_bar_ax)
            line_ax.append(inner_line_ax)

        if len(plot_axes) > len(target):
            for idx in range(len(target), len(plot_axes)):
                plot_axes[idx].axis('off')

    return [title_ax, bar_ax, line_ax]


def _modify_legend_ax(ax, font_family):
    for d in ['top', 'bottom', 'right', 'left']:
        ax.spines[d].set_visible(False)

    for tick in ax.get_xticklabels():
        tick.set_fontname(font_family)
    for tick in ax.get_yticklabels():
        tick.set_fontname(font_family)

    ax.set_facecolor('white')
    ax.set_xticks([])
    ax.set_yticks([])


def _plot_legend_colorbar(value_min, value_max, colorbar_ax, cmap, font_family, height="30%", width="100%"):
    norm = mpl.colors.Normalize(vmin=float(value_min), vmax=float(value_max))

    # color bar
    cax = inset_axes(colorbar_ax, height=height, width=width, loc=10)
    cb = mpl.colorbar.ColorbarBase(cax, cmap=plt.get_cmap(cmap), norm=norm, orientation='horizontal')
    cb.outline.set_linewidth(0)
    cb.set_ticks([])

    width_float = float(width.replace('%', '')) / 100
    cb.ax.text((1. - width_float) / 2, 0.5, round(value_min, 3), fontdict={'color': '#424242', 'fontsize': 10},
               transform=colorbar_ax.transAxes, horizontalalignment='left', verticalalignment='center')

    cb.ax.text(1. - (1. - width_float) / 2, 0.5, round(value_max, 3), fontdict={'color': '#ffffff', 'fontsize': 10},
               transform=colorbar_ax.transAxes, horizontalalignment='right', verticalalignment='center')
    _modify_legend_ax(colorbar_ax, font_family)


def _plot_legend_circles(count_min, count_max, circle_ax, cmap, font_family):
    # circle size
    circle_ax.scatter([0.75, 2], [1] * 2, s=[500, 1500],
                      edgecolors=plt.get_cmap(cmap)(1.), color='white')
    circle_ax.set_xlim(0., 3)

    circle_ax.text(0.75, 1., count_min, fontdict={'color': '#424242', 'fontsize': 10},
                   horizontalalignment='center', verticalalignment='center')
    circle_ax.text(1.275, 1., '-->', fontdict={'color': '#424242', 'fontsize': 10, 'weight': 'bold'},
                   horizontalalignment='center', verticalalignment='center')
    circle_ax.text(2, 1., count_max, fontdict={'color': '#424242', 'fontsize': 10},
                   horizontalalignment='center', verticalalignment='center')
    _modify_legend_ax(circle_ax, font_family)


def _plot_interact(target_count_data, _target, plot_ax, feature_names, display_columns, percentile_columns,
                   marker_sizes, cmap, line_width, xticks_rotation, font_family):
    plot_ax.set_xticks(range(len(display_columns[0])))
    plot_ax.set_xticklabels(display_columns[0], rotation=xticks_rotation)
    plot_ax.set_yticks(range(len(display_columns[1])))
    plot_ax.set_yticklabels(display_columns[1])
    plot_ax.set_xlim(-0.5, len(display_columns[0]) - 0.5)
    plot_ax.set_ylim(-0.5, len(display_columns[1]) - 0.5)

    # display percentile
    if len(percentile_columns[0]) > 0:
        percentile_ax = plot_ax.twiny()
        percentile_ax.set_xticks(plot_ax.get_xticks())
        percentile_ax.set_xbound(plot_ax.get_xbound())
        percentile_ax.set_xticklabels(percentile_columns[0], rotation=xticks_rotation)
        percentile_ax.set_xlabel('percentile buckets')
        _axes_modify(font_family=font_family, ax=percentile_ax, top=True)

    # display percentile
    if len(percentile_columns[1]) > 0:
        percentile_ay = plot_ax.twinx()
        percentile_ay.set_yticks(plot_ax.get_yticks())
        percentile_ay.set_ybound(plot_ax.get_ybound())
        percentile_ay.set_yticklabels(percentile_columns[1])
        percentile_ay.set_ylabel('percentile buckets')
        _axes_modify(font_family=font_family, ax=percentile_ay, right=True)

    value_min, value_max = target_count_data[_target].min(), target_count_data[_target].max()
    value_min_pts = target_count_data[target_count_data[_target] == value_min].reset_index(drop=True)
    value_max_pts = target_count_data[target_count_data[_target] == value_max].reset_index(drop=True)
    colors = [plt.get_cmap(cmap)(float(v - value_min) / (value_max - value_min))
              for v in target_count_data[_target].values]

    plot_ax.scatter(target_count_data['x1'].values, target_count_data['x2'].values,
                    s=marker_sizes, c=colors, linewidth=line_width, edgecolors=plt.get_cmap(cmap)(1.))
    plot_ax.set_xlabel(feature_names[0])
    plot_ax.set_ylabel(feature_names[1])

    # add annotation
    for pts in [value_min_pts, value_max_pts]:
        for i in range(pts.shape[0]):
            pt = pts.iloc[i]
            anno_text = '%.3f\n count: %d' % (pt[_target], pt['fake_count'])
            anno_x, anno_y = pt['x1'], pt['x2']
            text_x, text_y = anno_x + 0.2, anno_y + 0.2
            plot_ax.annotate(anno_text, xy=(anno_x, anno_y), xytext=(text_x, text_y),
                             arrowprops=dict(color=plt.get_cmap(cmap)(1.), arrowstyle="->"),
                             color=plt.get_cmap(cmap)(1.))

    _axes_modify(font_family=font_family, ax=plot_ax)
    return value_min, value_max


def _target_plot_interact(feature_names, display_columns, percentile_columns, target, target_count_data,
                          figsize, ncols, plot_params):
    """inner call for function target_plot_interact

    :param feature_names: list
        feature names
    :param display_columns: list
        xticklabels for both features
    :param percentile_columns: list
        xticklabels in percentile format for both features
    :param target: list
        list of target columns to investigate
        if it is not multi-classes, the list would only contain 1 element
    :param target_count_data: pandas DataFrame
        data for the graph
    :param figsize: tuple or None
        size of the figure, (width, height)
    :param ncols: integer, optional, default=2
        number subplot columns, used when it is multi-class problem
    :param plot_params: dict or None
        parameters for the plot

    :return: list of matplotlib Axes
        [axes for title, axes for value]
    """

    nrows = int(np.ceil(len(target) * 1.0 / ncols))
    ncols = np.min([len(target), ncols])
    width = np.min([8 * len(target), 16])
    height = width * 1.0 / ncols * nrows

    if figsize is not None:
        width, height = figsize

    # set up graph parameters
    if plot_params is None:
        plot_params = dict()

    font_family = plot_params.get('font_family', 'Arial')
    cmap = plot_params.get('cmap', 'Blues')
    cmaps = plot_params.get('cmaps', ['Blues', 'Greens', 'Oranges', 'Reds', 'Purples', 'Greys'])
    line_width = plot_params.get('line_width', 1)
    xticks_rotation = plot_params.get('xticks_rotation', 0)
    marker_size_min = plot_params.get('marker_size_min', 50)
    marker_size_max = plot_params.get('marker_size_max', 1500)

    # plot title
    plt.figure(figsize=(width, 2))
    title_ax = plt.subplot(111)
    _target_plot_title(feature_name=' & '.join(feature_names), ax=title_ax, plot_params=plot_params)

    # draw value plots and legend
    count_min, count_max = target_count_data['fake_count'].min(), target_count_data['fake_count'].max()
    marker_sizes = []
    for count in target_count_data['fake_count'].values:
        size = float(count - count_min) / (count_max - count_min) * (marker_size_max - marker_size_min) + marker_size_min
        marker_sizes.append(size)
    legend_width = np.max([8, np.min([width, 12])])

    if len(target) == 1:
        _, value_ax = plt.subplots(nrows=1, ncols=1, figsize=(width, height))
        value_min, value_max = _plot_interact(
            target_count_data=target_count_data, _target=target[0], plot_ax=value_ax,
            feature_names=feature_names, display_columns=display_columns, percentile_columns=percentile_columns,
            marker_sizes=marker_sizes, cmap=cmap, line_width=line_width, xticks_rotation=xticks_rotation,
            font_family=font_family)

        # draw legend
        _, legend_ax = plt.subplots(1, 2, gridspec_kw={'width_ratios': [2, 1]}, figsize=(legend_width, 1))
        _plot_legend_colorbar(value_min=value_min, value_max=value_max, colorbar_ax=legend_ax[0],
                              cmap=cmap, font_family=font_family)
        _plot_legend_circles(count_min=count_min, count_max=count_max, circle_ax=legend_ax[1],
                             cmap=cmap, font_family=font_family)

    else:
        # value_ax = fig.axes
        plt.figure(figsize=(width / 3, 1))
        legend_ax = plt.subplot(111)
        _plot_legend_circles(count_min=count_min, count_max=count_max, circle_ax=legend_ax,
                             cmap=cmap, font_family=font_family)

        _, value_ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(width, height), sharex='all', sharey='all')
        value_ax = value_ax.flatten()

        for target_idx in range(len(target)):
            cmap_idx = cmaps[target_idx % len(cmaps)]

            value_min, value_max = _plot_interact(
                target_count_data=target_count_data, _target=target[target_idx], plot_ax=value_ax[target_idx],
                feature_names=feature_names, display_columns=display_columns, percentile_columns=percentile_columns,
                marker_sizes=marker_sizes, cmap=cmap_idx, line_width=line_width,
                xticks_rotation=xticks_rotation, font_family=font_family)

            colorbar_ax = make_axes_locatable(value_ax[target_idx]).append_axes('bottom', size='5%', pad=0.5)
            _plot_legend_colorbar(value_min=value_min, value_max=value_max, colorbar_ax=colorbar_ax,
                                  cmap=cmap_idx, font_family=font_family, height="90%", width="80%")

            subplot_title = target[target_idx]
            if len(percentile_columns[1]) > 0:
                subplot_title += '\n\n\n'
            value_ax[target_idx].set_title(subplot_title, fontdict={'fontsize': 12, 'fontname': font_family})

        if len(value_ax) > len(target):
            for idx in range(len(target), len(value_ax)):
                value_ax[idx].axis('off')

    return [title_ax, value_ax]


