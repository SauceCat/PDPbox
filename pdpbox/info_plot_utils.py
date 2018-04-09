
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

import copy
import pdp_plot_utils, pdp_calc_utils


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

    pdp_plot_utils._axis_modify(font_family, ax1)
    pdp_plot_utils._axis_modify(font_family, ax2)

    df = copy.deepcopy(pdp_isolate_out.ice_lines)
    actual_columns = pdp_isolate_out.actual_columns
    feature_grids = pdp_isolate_out.feature_grids
    df = df[actual_columns + ['actual_preds']]

    if pdp_isolate_out.feature_type == 'binary':
        df['x'] = df[actual_columns[0]]
    elif pdp_isolate_out.feature_type == 'onehot':
        df['x'] = df[actual_columns].apply(lambda x: pdp_calc_utils._find_onehot_actual(x), axis=1)
        df = df[df['x'].isnull() == False].reset_index(drop=True)
    else:
        df = df[(df[actual_columns[0]] >= feature_grids[0])
                & (df[actual_columns[0]] <= feature_grids[-1])].reset_index(drop=True)
        df['x'] = df[actual_columns[0]].apply(lambda x: pdp_calc_utils._find_closest(x, feature_grids))

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


def _autolabel(rects, ax, barcolor):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        bbox_props = {'facecolor': 'white', 'edgecolor': barcolor, 'boxstyle': "square,pad=0.5"}
        ax.text(rect.get_x() + rect.get_width() / 2., height, '%d' % int(height),
                ha='center', va='center', bbox=bbox_props, color=barcolor, weight='bold')


def _target_plot_title(feature_name, ax, figsize, plot_params):
    """
    Draw target plot title

    :param feature_name: feature name
    :param ax: axes to plot on
    :param figsize: figure size
    :param plot_params: values of plot parameters
    """

    font_family = 'Arial'
    title = 'Real target plot for %s' % feature_name
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
    ax.text(0, 0.7, title, va="top", ha="left", fontsize=title_fontsize, fontname=font_family)
    ax.text(0, 0.4, subtitle, va="top", ha="left", fontsize=subtitle_fontsize, fontname=font_family, color='grey')
    ax.axis('off')


def _target_plot(feature_name, feature_grids, target, bar_counts_gp, target_lines, figsize, plot_params):
    """
    Plot target distribution through feature grids

    :param feature_name: feature name
    :param feature_grids: feature grids
    :param target: target columns
    :param bar_counts_gp: bar counts data
    :param target_lines: target lines data
    :param figsize: figure size
    :param plot_params: values of plot parameters
    """

    if figsize is None:
        figwidth = 16
        figheight = 7
    else:
        figwidth = figsize[0]
        figheight = figsize[1]

    font_family = 'Arial'
    linecolor = '#1A4E5D'
    barcolor = '#5BB573'
    linewidth = 2
    xticks_rotation = 0

    if plot_params is not None:
        if 'font_family' in plot_params.keys():
            font_family = plot_params['font_family']
        if 'linecolor' in plot_params.keys():
            linecolor = plot_params['linecolor']
        if 'barcolor' in plot_params.keys():
            barcolor = plot_params['barcolor']
        if 'linewidth' in plot_params.keys():
            linewidth = plot_params['linewidth']
        if 'xticks_rotation' in plot_params.keys():
            xticks_rotation = plot_params['xticks_rotation']

    plt.figure(figsize=(figwidth, figwidth / 6.7))
    ax1 = plt.subplot(111)
    _target_plot_title(feature_name=feature_name, ax=ax1, figsize=figsize, plot_params=plot_params)

    boxwith = np.min([0.5, 0.5 / (10.0 / len(feature_grids))])
    plt.figure(figsize=(figwidth, figheight))
    ax1 = plt.subplot(111)
    rects = ax1.bar(bar_counts_gp['x'], bar_counts_gp['fake_count'], width=boxwith, color=barcolor, alpha=0.5)
    ax1.set_xlabel(feature_name)
    ax1.set_ylabel('count')
    plt.xticks(range(len(feature_grids)), feature_grids, rotation=xticks_rotation)
    _autolabel(rects, ax1, barcolor)

    ax2 = ax1.twinx()
    if len(target_lines) == 1:
        target_line = target_lines[0]
        ax2.plot(target_line['x'], target_line[target], linewidth=linewidth, c=linecolor, marker='o')
        for idx in range(target_line.shape[0]):
            bbox_props = {'facecolor': linecolor, 'edgecolor': 'none', 'boxstyle': "square,pad=0.5"}
            ax2.text(idx, target_line.iloc[idx][target], '%.3f' % (round(target_line.iloc[idx][target], 3)),
                     ha="center", va="bottom", size=10, bbox=bbox_props, color='#ffffff', weight='bold')
    else:
        linecolors = plt.get_cmap('tab10')(range(10))
        for target_idx in range(len(target)):
            linecolor = linecolors[target_idx]
            target_line = target_lines[target_idx]
            ax2.plot(target_line['x'], target_line[target[target_idx]], linewidth=linewidth, c=linecolor, marker='o',
                     label=target[target_idx])
            for idx in range(target_line.shape[0]):
                bbox_props = {'facecolor': linecolor, 'edgecolor': 'none', 'boxstyle': "square,pad=0.5"}
                ax2.text(idx, target_line.iloc[idx][target[target_idx]],
                         '%.3f' % (round(target_line.iloc[idx][target[target_idx]], 3)),
                         ha="center", va="top", size=10, bbox=bbox_props, color='#ffffff', weight='bold')
            plt.legend()

    pdp_plot_utils._axis_modify(font_family, ax2)
    ax2.get_yaxis().tick_right()
    ax2.grid(False)
    ax2.set_ylabel('target_avg')

    pdp_plot_utils._axis_modify(font_family, ax1)