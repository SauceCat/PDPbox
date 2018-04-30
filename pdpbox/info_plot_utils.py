
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from mpl_toolkits.axes_grid.inset_locator import inset_axes

import copy
from pdp_plot_utils import _axes_modify
from other_utils import _make_list
from pdp_calc_utils import (_get_grids, _find_bucket, _make_bucket_column_names, _find_onehot_actual,
                            _find_closest, _make_bucket_column_names_percentile)


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
    """add annotation for bar plot

    Parameters:
    -----------

    :param rects: list of matplotlib Axes
        list of axes returned by bar plot
    :param ax: matplotlib Axes
        axes to plot on
    :param bar_color: string
        color for the bars
    """

    for rect in rects:
        height = rect.get_height()
        bbox_props = {'facecolor': 'white', 'edgecolor': bar_color, 'boxstyle': "square,pad=0.5"}
        ax.text(rect.get_x() + rect.get_width() / 2., height, '%d' % int(height),
                ha='center', va='center', bbox=bbox_props, color=bar_color)


def _prepare_data_x(feature, feature_type, data, num_grid_points, grid_type, percentile_range,
                    grid_range, cust_grid_points, show_percentile):
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
        feature_grids = display_columns = np.array([0, 1])
        data_x['x'] = data_x[feature]
    if feature_type == 'numeric':
        percentile_info = None
        if cust_grid_points is None:
            feature_grids, percentile_info = _get_grids(
                x=data_x[feature].values, num_grid_points=num_grid_points, grid_type=grid_type,
                percentile_range=percentile_range, grid_range=grid_range)
        else:
            feature_grids = np.array(sorted(cust_grid_points))

        data_x = data_x[(data_x[feature] >= feature_grids[0])
                        & (data_x[feature] <= feature_grids[-1])].reset_index(drop=True)

        # map feature value into value buckets
        data_x['x'] = data_x[feature].apply(lambda x: _find_bucket(x=x, feature_grids=feature_grids))

        # create bucket names
        display_columns = _make_bucket_column_names(feature_grids=feature_grids)

        # create percentile bucket names
        if show_percentile and grid_type == 'percentile':
            percentile_columns = _make_bucket_column_names_percentile(percentile_info=percentile_info)

    if feature_type == 'onehot':
        feature_grids = display_columns = np.array(feature)
        data_x['x'] = data_x[feature].apply(lambda x: _find_onehot_actual(x=x), axis=1)
        data_x = data_x[~data_x['x'].isnull()].reset_index(drop=True)

    return data_x, display_columns, percentile_columns


def _target_plot_title(feature_name, ax, plot_params):
    """title for target plot

    Parameters:
    -----------

    :param feature_name: string
        name of the feature, not necessary a column name
    :param ax: matplotlib Axes
        axes to plot on
    :param plot_params: dict
        parameters for the plot
    """

    font_family = plot_params.get('font_family', 'Arial')
    title = plot_params.get('title', 'Target plot for feature "%s"' % feature_name)
    subtitle = plot_params.get('subtitle', 'Average target values through feature grids.')
    title_fontsize = plot_params.get('title_fontsize', 15)
    subtitle_fontsize = plot_params.get('subtitle_fontsize', 12)

    ax.set_facecolor('white')
    ax.text(0, 0.7, title, va="top", ha="left", fontsize=title_fontsize, fontname=font_family)
    ax.text(0, 0.5, subtitle, va="top", ha="left", fontsize=subtitle_fontsize, fontname=font_family, color='grey')
    ax.axis('off')


def _target_plot_title_ax(feature_name, ax, plot_params, percentile_columns):
    """title for target plot (when axes is provided at every beginning)

    Parameters:
    -----------

    :param feature_name: string
        name of the feature, not necessary a column name
    :param ax: matplotlib Axes
        axes to plot on
    :param plot_params: dict
        parameters for the plot
    :param percentile_columns: list
        list of xticklabels in percentile format
    """

    font_family = plot_params.get('font_family', 'Arial')
    title = plot_params.get('title', 'Target plot for feature "%s"' % feature_name) + '\n'
    title_fontsize = plot_params.get('title_fontsize', 12)

    if len(percentile_columns) > 0:
        title += '\n\n'
    ax.set_title(title, fontdict={'fontsize': title_fontsize, 'fontname': font_family})


def _target_plot(feature_name, display_columns, percentile_columns, target, bar_data,
                 target_lines, figsize, plot_params):
    """inner call for function target_plot

    Parameters:
    -----------

    :param feature_name: string
        name of the feature, not necessary a column name
    :param display_columns: list
        list of xticklabels
    :param percentile_columns: list
        list of xticklabels in percentile format
    :param target: string or list
        column name or column name list for target value
        for multi-class problem, a list of one-hot encoding target column
    :param bar_data: pandas DataFrame
        data for bar plot
    :param target_lines: list of pandas DataFrame
        data for target lines
    :param figsize: tuple or None
        size of the figure, (width, height)
    :param plot_params: dict or None
        parameters for the plot

    Returns:
    --------

    :return: matplotlib Axes or list of matplotlib Axes
        if Axes is provided, return provided Axes
        if not, return [axes for title, axes for bar, axes for line]
    """

    # set up graph parameters
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

    # Axes for title
    plt.figure(figsize=(width, 2))
    title_ax = plt.subplot(111)
    _target_plot_title(feature_name=feature_name, ax=title_ax, plot_params=plot_params)

    # Axes for bar and lines
    plt.figure(figsize=(width, height))
    bar_ax = plt.subplot(111)
    line_ax = bar_ax.twinx()

    # bar plot
    box_width = np.min([0.5, 0.5 / (10.0 / len(display_columns))])
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

    # add value label for bar plot
    rects = bar_ax.bar(x=bar_data['x'], height=bar_data['fake_count'], width=box_width, color=bar_color, alpha=0.5)
    _autolabel(rects=rects, ax=bar_ax, bar_color=bar_color)
    _axes_modify(font_family=font_family, ax=bar_ax)

    # target lines
    if len(target_lines) == 1:
        target_line = target_lines[0].sort_values('x', ascending=True).set_index('x')
        line_ax.plot(target_line.index.values, target_line[target].values,
                     linewidth=line_width, c=line_color, marker='o')
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
            line_ax.legend(loc="upper left", ncol=5, bbox_to_anchor=(0, 1.2), frameon=False)

    _axes_modify(font_family=font_family, ax=line_ax)

    line_ax.get_yaxis().tick_right()
    line_ax.grid(False)
    line_ax.set_ylabel('target_avg')

    return [title_ax, bar_ax, line_ax]


def _plot_legend(value_min, value_max, count_min, count_max, legend_axes, cmap, font_family):
    norm = mpl.colors.Normalize(vmin=float(value_min), vmax=float(value_max))

    # colorbar
    cax = inset_axes(legend_axes[0], height="20%", width="100%", loc=10)
    cb = mpl.colorbar.ColorbarBase(cax, cmap=plt.get_cmap(cmap), norm=norm, orientation='horizontal')
    legend_axes[0].set_xlabel('Average target value')
    cb.outline.set_linewidth(0)
    cb.set_ticks([])

    legend_axes[0].set_xticks(range(2))
    legend_axes[0].set_xticklabels([round(value_min, 3), round(value_max, 3)])

    # circle size
    legend_axes[1].scatter([-0.5, 0.75, 2], [1] * 3, s=[100, 500, 1000],
                           edgecolors=plt.get_cmap(cmap)(1.), color='white')
    legend_axes[1].set_xlim(-1, 3)
    legend_axes[1].set_xlabel('Counts')

    legend_axes[1].set_xticks([-1, 3])
    legend_axes[1].set_xticklabels([count_min, count_max])

    for i in range(2):
        for d in ['top', 'right', 'left']:
            legend_axes[i].spines[d].set_visible(False)
        for tick in legend_axes[i].get_xticklabels():
            tick.set_fontname(font_family)
        for tick in legend_axes[i].get_yticklabels():
            tick.set_fontname(font_family)

        legend_axes[i].set_facecolor('white')
        legend_axes[i].tick_params(axis='both', which='major', labelsize=10, labelcolor='#424242', colors='#9E9E9E')
        legend_axes[i].set_yticks([])


def _target_plot_interact(feature_names, display_columns, percentile_columns, target, target_count_data,
                          figsize, ncols, plot_params):

    target = _make_list(target)
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
    line_width = plot_params.get('line_width', 1)
    xticks_rotation = plot_params.get('xticks_rotation', 0)
    line_color = plt.get_cmap(cmap)(1.0)

    # draw title and values
    plt.figure(figsize=(width, 2))
    title_ax = plt.subplot(111)

    _, value_ax = plt.subplots(nrows=nrows, ncols=ncols, sharex='col', sharey='row', figsize=(width, height))

    # draw legend
    legend_width = np.max([8, np.min([width, 12])])
    _, legend_axes = plt.subplots(1, 2, gridspec_kw={'width_ratios': [2, 1]}, figsize=(legend_width, 1))

    _target_plot_title(feature_name=' & '.join(feature_names), ax=title_ax, plot_params=plot_params)

    marker_size_min, marker_size_max = 100, 1000
    count_min, count_max = target_count_data['fake_count'].min(), target_count_data['fake_count'].max()

    marker_sizes = []
    for count in target_count_data['fake_count'].values:
        size = (count - count_min) / (count_max - count_min) * (marker_size_max - marker_size_min) + marker_size_min
        marker_sizes.append(size)

    plot_ax = []
    # recreate target value range here
    value_min, value_max = target_count_data[target[0]].min(), target_count_data[target[0]].max()
    value_min_pts = target_count_data[target_count_data[target[0]] == value_min].reset_index(drop=True)
    value_max_pts = target_count_data[target_count_data[target[0]] == value_max].reset_index(drop=True)

    if len(target) == 1:
        plot_ax.append(value_ax)
    else:
        plot_ax = value_ax.flatten()
        for idx in range(1, len(target)):
            if target_count_data[target[idx]].min() < value_min:
                value_min = target_count_data[target[idx]].min()
                value_min_pts = target_count_data[target_count_data[target[idx]] == value_min].reset_index(drop=True)
            if target_count_data[target[idx]].max() > value_max:
                value_max = target_count_data[target[idx]].max()
                value_max_pts = target_count_data[target_count_data[target[idx]] == value_max].reset_index(drop=True)
    target_value_range = [value_min, value_max]

    _plot_legend(value_min, value_max, count_min, count_max, legend_axes, cmap, font_family)

    for idx in range(len(plot_ax)):
        plot_ax[idx].set_xticks(range(len(display_columns[0])))
        plot_ax[idx].set_xticklabels(display_columns[0], rotation=xticks_rotation)
        plot_ax[idx].set_yticks(range(len(display_columns[1])))
        plot_ax[idx].set_yticklabels(display_columns[1])
        plot_ax[idx].set_xlim(-0.5, len(display_columns[0]) - 0.5)
        plot_ax[idx].set_ylim(-0.5, len(display_columns[1]) - 0.5)

        # display percentile
        if len(percentile_columns[0]) > 0:
            percentile_ax = plot_ax[idx].twiny()
            percentile_ax.set_xticks(plot_ax[idx].get_xticks())
            percentile_ax.set_xbound(plot_ax[idx].get_xbound())
            percentile_ax.set_xticklabels(percentile_columns[0], rotation=xticks_rotation)
            percentile_ax.set_xlabel('percentile buckets')
            _axes_modify(font_family=font_family, ax=percentile_ax, top=True)

        # display percentile
        if len(percentile_columns[1]) > 0:
            percentile_ay = plot_ax[idx].twinx()
            percentile_ay.set_yticks(plot_ax[idx].get_yticks())
            percentile_ay.set_ybound(plot_ax[idx].get_ybound())
            percentile_ay.set_yticklabels(percentile_columns[1])
            percentile_ay.set_ylabel('percentile buckets')
            _axes_modify(font_family=font_family, ax=percentile_ay, right=True)

        colors = [plt.get_cmap(cmap)(float(v - target_value_range[0]) / (target_value_range[1] - target_value_range[0]))
                  for v in target_count_data[target[idx]].values]
        plot_ax[idx].scatter(target_count_data['x1'].values, target_count_data['x2'].values,
                             s=marker_sizes, c=colors, linewidth=line_width, edgecolors=line_color)
        plot_ax[idx].set_xlabel(feature_names[0])
        plot_ax[idx].set_ylabel(feature_names[1])
        if len(target) > 1:
            if len(percentile_columns[0]) > 0:
                plot_ax[idx].set_title(target[idx] + '\n\n', fontsize=12)
            else:
                plot_ax[idx].set_title(target[idx], fontsize=12)

        # add annotation
        for i in range(value_min_pts.shape[0]):
            pt = value_min_pts.iloc[i]
            anno_text = '%.3f\ncount: %d' %(value_min, pt['fake_count'])
            anno_x, anno_y = pt['x1'], pt['x2']
            text_x, text_y = anno_x + 0.2, anno_y + 0.2
            plot_ax[idx].annotate(anno_text, xy=(anno_x, anno_y), xytext=(text_x, text_y),
                                  arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=6))

        for i in range(value_max_pts.shape[0]):
            pt = value_max_pts.iloc[i]
            anno_text = '%.3f\ncount: %d' %(value_max, pt['fake_count'])
            anno_x, anno_y = pt['x1'], pt['x2']
            text_x, text_y = anno_x + 0.2, anno_y + 0.2
            plot_ax[idx].annotate(anno_text, xy=(anno_x, anno_y), xytext=(text_x, text_y),
                                  arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=6))

        _axes_modify(font_family=font_family, ax=plot_ax[idx])

    return [title_ax, value_ax]


