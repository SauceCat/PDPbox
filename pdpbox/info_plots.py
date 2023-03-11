from .info_plot_utils import (
    _target_plot,
    _target_plot_plotly,
    _info_plot_interact,
    _actual_plot,
    _actual_plot_plotly,
    _prepare_info_plot_interact_data,
    _prepare_info_plot_data,
    _check_info_plot_interact_params,
    _check_info_plot_params,
    _prepare_plot_params,
    _prepare_actual_plot_data,
    _info_plot_interact_plotly,
)
from .utils import _make_list, _check_target, _q1, _q2, _q3


def target_plot(
    df,
    feature,
    feature_name,
    target,
    num_grid_points=10,
    grid_type="percentile",
    percentile_range=None,
    grid_range=None,
    cust_grid_points=None,
    show_percentile=False,
    show_outliers=False,
    endpoint=True,
    figsize=None,
    dpi=300,
    ncols=2,
    plot_params=None,
    engine="plotly",
    template="plotly_white",
):
    """
    Plot average target value by different feature values (feature grids)
    For binary or one-hot encoded features, it is very intuitive.

    Parameters
    ----------
    df: pandas DataFrame
        data set to investigate on, should contain at least
        the feature to investigate as well as the target
    feature: string or list
        feature or feature list to investigate,
        for one-hot encoding features, feature list is required
    feature_name: string
        name of the feature, not necessary a column name
    target: string or list
        column name or column name list for target value
        for multi-class problem, a list of one-hot encoding target column
    num_grid_points: integer, optional, default=10
        number of grid points for numeric feature
    grid_type: string, optional, default='percentile'
        'percentile' or 'equal'
        type of grid points for numeric feature
    percentile_range: tuple or None, optional, default=None
        percentile range to investigate
        for numeric feature when grid_type='percentile'
    grid_range: tuple or None, optional, default=None
        value range to investigate
        for numeric feature when grid_type='equal'
    cust_grid_points: Series, 1d-array, list or None, optional, default=None
        customized list of grid points
        for numeric feature
    show_percentile: bool, optional, default=False
        whether to display the percentile buckets
        for numeric feature when grid_type='percentile'
    show_outliers: bool, optional, default=False
        whether to display the out of range buckets
        for numeric feature when percentile_range or grid_range is not None
    endpoint: bool, optional, default=True
        If True, stop is the last grid point
        Otherwise, it is not included
    figsize: tuple or None, optional, default=None
        size of the figure, (width, height)
    ncols: integer, optional, default=2
        number subplot columns, used when it is multi-class problem
    plot_params: dict or None, optional, default=None
        parameters for the plot, check styles.infoPlotStyle for more details
    engine: str, optinal, default=plotly
        visualization engine, can be plotly or matplotlib
    template: str, optional, default='plotly_white'
        plotly template

    Returns
    -------
    fig: matplotlib Figure
    axes: a dictionary of matplotlib Axes
        Returns the Axes objects for further tweaking
    summary_df: pandas DataFrame
        Graph data in data frame format

    Examples
    --------

    Quick start with target_plot

    .. highlight:: python
    .. code-block:: python

        from pdpbox import info_plots, get_dataset

        test_titanic = get_dataset.titanic()
        titanic_data = test_titanic['data']
        titanic_target = test_titanic['target']
        fig, axes, summary_df = info_plots.target_plot(
            df=titanic_data, feature='Sex', feature_name='Sex', target=titanic_target)


    With One-hot encoding features

    .. highlight:: python
    .. code-block:: python

        fig, axes, summary_df = info_plots.target_plot(
            df=titanic_data, feature=['Embarked_C', 'Embarked_Q', 'Embarked_S'],
            feature_name='Embarked', target=titanic_target)


    With numeric features

    .. highlight:: python
    .. code-block:: python

        fig, axes, summary_df = info_plots.target_plot(
            df=titanic_data, feature='Fare', feature_name='Fare',
            target=titanic_target, show_percentile=True)


    With multi-class

    .. highlight:: python
    .. code-block:: python

        from pdpbox import info_plots, get_dataset

        test_otto = get_dataset.otto()
        otto_data = test_otto['data']
        otto_target = test_otto['target']
        fig, axes, summary_df = info_plots.target_plot(
            df=otto_data, feature='feat_67', feature_name='feat_67',
            target=['target_0', 'target_2', 'target_5', 'target_8'])

    """

    _check_target(target, df)
    feature_type, show_outliers = _check_info_plot_params(
        df,
        feature,
        grid_type,
        percentile_range,
        grid_range,
        cust_grid_points,
        show_outliers,
    )

    target = _make_list(target)
    useful_features = _make_list(feature) + target

    # map feature values to grid point buckets (x)
    data = df[useful_features]
    (
        data_x,
        bar_data,
        summary_df,
        display_columns,
        percentile_columns,
    ) = _prepare_info_plot_data(
        feature,
        feature_type,
        data,
        num_grid_points,
        grid_type,
        percentile_range,
        grid_range,
        cust_grid_points,
        show_percentile,
        show_outliers,
        endpoint,
    )

    # prepare data for target lines
    # each target line contains 'x' and mean target value
    target_lines = []
    for t in target:
        target_line = (
            data_x.groupby("x", as_index=False)
            .agg({t: "mean"})
            .sort_values("x", ascending=True)
        )
        target_lines.append(target_line)
        summary_df = summary_df.merge(target_line, on="x", how="outer")

    plot_params = _prepare_plot_params(
        plot_params,
        ncols,
        display_columns,
        percentile_columns,
        figsize,
        dpi,
        template,
        engine,
    )

    if engine == "matplotlib":
        fig, axes = _target_plot(
            feature_name, target, bar_data, target_lines, plot_params
        )
    else:
        fig = _target_plot_plotly(
            feature_name, target, bar_data, target_lines, plot_params
        )
        axes = None

    return fig, axes, summary_df


def actual_plot(
    model,
    X,
    feature,
    feature_name,
    pred_func=None,
    n_classes=None,
    num_grid_points=10,
    grid_type="percentile",
    percentile_range=None,
    grid_range=None,
    cust_grid_points=None,
    show_percentile=False,
    show_outliers=False,
    endpoint=True,
    which_classes=None,
    predict_kwds={},
    ncols=2,
    figsize=None,
    dpi=300,
    plot_params=None,
    engine="plotly",
    template="plotly_white",
):
    """Plot prediction distribution across different feature values (feature grid)

    Parameters
    ----------

    model: a fitted model
    X: pandas DataFrame
        data set on which the model is trained
    feature: string or list
        feature or feature list to investigate
        for one-hot encoding features, feature list is required
    feature_name: string
        name of the feature, not necessary a column name
    pred_func: callable, optional, default=None
        function to get model prediction on X.
        if it's not provided, we assume the model is a sklearn model.
        we use `model.predict_proba` to get classification prediction,
        and `model.predict` for regression prediction.
        if the provided model doesn't follow the same pattern as a sklearn model,
        you need to provide the predict function,
        the function must be sth like: `func(model, X, predict_kwds)`.
    n_classes: integer, optional, default=None
        required when we can't infer number of classes from `model.classes_`,
        `n_classes` should be set as 0 when it is a regression problem
    num_grid_points: integer, optional, default=10
        number of grid points for numeric feature
    grid_type: string, optional, default='percentile'
        'percentile' or 'equal',
        type of grid points for numeric feature
    percentile_range: tuple or None, optional, default=None
        percentile range to investigate,
        for numeric feature when grid_type='percentile'
    grid_range: tuple or None, optional, default=None
        value range to investigate,
        for numeric feature when grid_type='equal'
    cust_grid_points: Series, 1d-array, list or None, optional, default=None
        customized list of grid points for numeric feature
    show_percentile: bool, optional, default=False
        whether to display the percentile buckets,
        for numeric feature when grid_type='percentile'
    show_outliers: bool, optional, default=False
        whether to display the out of range buckets
        for numeric feature when percentile_range or grid_range is not None
    endpoint: bool, optional
        If True, stop is the last grid point, default=True
        Otherwise, it is not included
    which_classes: list, optional, default=None
        which classes to plot, only use when it is a multi-class problem
    predict_kwds: dict, default={}
        keywords to be passed to the model's predict function
    figsize: tuple or None, optional, default=None
        size of the figure, (width, height)
    ncols: integer, optional, default=2
        number subplot columns, used when it is multi-class problem
    plot_params: dict or None, optional, default=None
        parameters for the plot
    engine: str, optinal, default=plotly
        visualization engine, can be plotly or matplotlib
    template: str, optional, default='plotly_white'
        plotly template

    Returns
    -------
    fig: matplotlib Figure
    axes: a dictionary of matplotlib Axes
        Returns the Axes objects for further tweaking
    summary_df: pandas DataFrame
        Graph data in data frame format

    Examples
    --------

    Quick start with actual_plot

    .. highlight:: python
    .. code-block:: python

        from pdpbox import info_plots, get_dataset

        test_titanic = get_dataset.titanic()
        titanic_data = test_titanic['data']
        titanic_features = test_titanic['features']
        titanic_target = test_titanic['target']
        titanic_model = test_titanic['xgb_model']
        fig, axes, summary_df = info_plots.actual_plot(
            model=titanic_model, X=titanic_data[titanic_features],
            feature='Sex', feature_name='Sex')


    With One-hot encoding features

    .. highlight:: python
    .. code-block:: python

        fig, axes, summary_df = info_plots.actual_plot(
            model=titanic_model, X=titanic_data[titanic_features],
            feature=['Embarked_C', 'Embarked_Q', 'Embarked_S'], feature_name='Embarked')


    With numeric features

    .. highlight:: python
    .. code-block:: python

        fig, axes, summary_df = info_plots.actual_plot(
            model=titanic_model, X=titanic_data[titanic_features],
            feature='Fare', feature_name='Fare')


    With multi-class

    .. highlight:: python
    .. code-block:: python

        from pdpbox import info_plots, get_dataset

        test_otto = get_dataset.otto()
        otto_data = test_otto['data']
        otto_model = test_otto['rf_model']
        otto_features = test_otto['features']
        otto_target = test_otto['target']

        fig, axes, summary_df = info_plots.actual_plot(
            model=otto_model, X=otto_data[otto_features],
            feature='feat_67', feature_name='feat_67', which_classes=[1, 2, 3])

    """

    feature_type, show_outliers = _check_info_plot_params(
        X,
        feature,
        grid_type,
        percentile_range,
        grid_range,
        cust_grid_points,
        show_outliers,
    )

    info_features = _make_list(feature)
    info_df, pred_cols = _prepare_actual_plot_data(
        model, X, n_classes, pred_func, predict_kwds, info_features, which_classes
    )

    (
        info_df_x,
        bar_data,
        summary_df,
        display_columns,
        percentile_columns,
    ) = _prepare_info_plot_data(
        feature,
        feature_type,
        info_df,
        num_grid_points,
        grid_type,
        percentile_range,
        grid_range,
        cust_grid_points,
        show_percentile,
        show_outliers,
        endpoint,
    )

    # prepare data for box lines
    # each box line contains 'x' and actual prediction q1, q2, q3
    box_lines = []
    pred_cols_qs = []
    for p in pred_cols:
        box_line = (
            info_df_x.groupby("x", as_index=False)
            .agg({p: [_q1, _q2, _q3]})
            .sort_values("x", ascending=True)
        )
        box_line.columns = [
            "".join(col) if col[1] != "" else col[0] for col in box_line.columns
        ]
        box_lines.append(box_line)
        pred_cols_qs += [p + "_%s" % q for q in ["q1", "q2", "q3"]]
        summary_df = summary_df.merge(box_line, on="x", how="outer").fillna(0)

    plot_params = _prepare_plot_params(
        plot_params,
        ncols,
        display_columns,
        percentile_columns,
        figsize,
        dpi,
        template,
        engine,
    )

    if engine == "matplotlib":
        fig, axes = _actual_plot(
            feature_name,
            pred_cols,
            info_df_x,
            bar_data,
            box_lines,
            plot_params,
        )
    else:
        fig = _actual_plot_plotly(
            feature_name,
            pred_cols,
            info_df_x,
            bar_data,
            box_lines,
            plot_params,
        )
        axes = None
    return fig, axes, summary_df


def target_plot_interact(
    df,
    features,
    feature_names,
    target,
    num_grid_points=None,
    grid_types=None,
    percentile_ranges=None,
    grid_ranges=None,
    cust_grid_points=None,
    show_percentile=False,
    show_outliers=False,
    endpoint=True,
    figsize=None,
    dpi=300,
    ncols=2,
    annotate=False,
    plot_params=None,
    engine="plotly",
    template="plotly_white",
):
    """Plot average target value across different feature value combinations (feature grid combinations)

    Parameters
    ----------

    df: pandas DataFrame
        data set to investigate on, should contain at least
        the feature to investigate as well as the target
    features: list
        two features to investigate
    feature_names: list
        feature names
    target: string or list
        column name or column name list for target value
        for multi-class problem, a list of one-hot encoding target column
    num_grid_points: list, optional, default=None
        number of grid points for each feature
    grid_types: list, optional, default=None
        type of grid points for each feature
    percentile_ranges: list of tuple, optional, default=None
        percentile range to investigate for each feature
    grid_ranges: list of tuple, optional, default=None
        value range to investigate for each feature
    cust_grid_points: list of (Series, 1d-array, list), optional, default=None
        customized list of grid points for each feature
    show_percentile: bool, optional, default=False
        whether to display the percentile buckets for both feature
    show_outliers: bool, optional, default=False
        whether to display the out of range buckets for both features
    endpoint: bool, optional
        If True, stop is the last grid point, default=True
        Otherwise, it is not included
    figsize: tuple or None, optional, default=None
        size of the figure, (width, height)
    ncols: integer, optional, default=2
        number subplot columns, used when it is multi-class problem
    annotate: bool, default=False
        whether to annotate the points
    plot_params: dict or None, optional, default=None
        parameters for the plot
    engine: str, optinal, default=plotly
        visualization engine, can be plotly or matplotlib
    template: str, optional, default='plotly_white'
        plotly template

    Notes
    -----

    - Parameters are consistent with the ones for function target_plot
    - But for this function, you need to specify parameter value for both features in list format
    - For example:
        - percentile_ranges = [(0, 90), (5, 95)] means
        - percentile_range = (0, 90) for feature 1
        - percentile_range = (5, 95) for feature 2

    Returns
    -------
    fig: matplotlib Figure
    axes: a dictionary of matplotlib Axes
        Returns the Axes objects for further tweaking
    summary_df: pandas DataFrame
        Graph data in data frame format

    Examples
    --------

    Quick start with target_plot_interact

    .. highlight:: python
    .. code-block:: python

        from pdpbox import info_plots, get_dataset

        test_titanic = get_dataset.titanic()
        titanic_data = test_titanic['data']
        titanic_target = test_titanic['target']

        fig, axes, summary_df = info_plots.target_plot_interact(
            df=titanic_data, features=['Sex', ['Embarked_C', 'Embarked_Q', 'Embarked_S']],
            feature_names=['Sex', 'Embarked'], target=titanic_target)

    """

    _check_target(target, df)
    target = _make_list(target)
    useful_features = _make_list(features[0]) + _make_list(features[1]) + target

    return plot_interact(
        df[useful_features],
        features,
        feature_names,
        target,
        num_grid_points,
        grid_types,
        percentile_ranges,
        grid_ranges,
        cust_grid_points,
        show_percentile,
        show_outliers,
        endpoint,
        figsize,
        dpi,
        ncols,
        annotate,
        plot_params,
        engine,
        template,
        "target_interact",
    )


def actual_plot_interact(
    model,
    X,
    features,
    feature_names,
    pred_func=None,
    n_classes=None,
    num_grid_points=None,
    grid_types=None,
    percentile_ranges=None,
    grid_ranges=None,
    cust_grid_points=None,
    show_percentile=False,
    show_outliers=False,
    endpoint=True,
    which_classes=None,
    predict_kwds={},
    figsize=None,
    dpi=300,
    ncols=2,
    annotate=False,
    plot_params=None,
    engine="plotly",
    template="plotly_white",
):
    """Plot prediction distribution across different feature value combinations (feature grid combinations)

    Parameters
    ----------

    model: a fitted sklearn model
    X: pandas DataFrame
        data set to investigate on, should contain at least
        the feature to investigate as well as the target
    features: list
        two features to investigate
    feature_names: list
        feature names
    num_grid_points: list, optional, default=None
        number of grid points for each feature
    grid_types: list, optional, default=None
        type of grid points for each feature
    percentile_ranges: list of tuple, optional, default=None
        percentile range to investigate for each feature
    grid_ranges: list of tuple, optional, default=None
        value range to investigate for each feature
    cust_grid_points: list of (Series, 1d-array, list), optional, default=None
        customized list of grid points for each feature
    show_percentile: bool, optional, default=False
        whether to display the percentile buckets for both feature
    show_outliers: bool, optional, default=False
        whether to display the out of range buckets for both features
    endpoint: bool, optional
        If True, stop is the last grid point, default=True
        Otherwise, it is not included
    which_classes: list, optional, default=None
        which classes to plot, only use when it is a multi-class problem
    predict_kwds: dict, default={}
        keywords to be passed to the model's predict function
    figsize: tuple or None, optional, default=None
        size of the figure, (width, height)
    ncols: integer, optional, default=2
        number subplot columns, used when it is multi-class problem
    annotate: bool, default=False
        whether to annotate the points
    plot_params: dict or None, optional, default=None
        parameters for the plot

    Returns
    -------
    fig: matplotlib Figure
    axes: a dictionary of matplotlib Axes
        Returns the Axes objects for further tweaking
    summary_df: pandas DataFrame
        Graph data in data frame format

    Notes
    -----

    - Parameters are consistent with the ones for function actual_plot
    - But for this function, you need to specify parameter value for both features in list format
    - For example:
        - percentile_ranges = [(0, 90), (5, 95)] means
        - percentile_range = (0, 90) for feature 1
        - percentile_range = (5, 95) for feature 2

    Examples
    --------

    Quick start with actual_plot_interact

    .. highlight:: python
    .. code-block:: python

        from pdpbox import info_plots, get_dataset

        test_titanic = get_dataset.titanic()
        titanic_data = test_titanic['data']
        titanic_features = test_titanic['features']
        titanic_target = test_titanic['target']
        titanic_model = test_titanic['xgb_model']

        fig, axes, summary_df = info_plots.actual_plot_interact(
            model=titanic_model, X=titanic_data[titanic_features],
            features=['Fare', ['Embarked_C', 'Embarked_Q', 'Embarked_S']],
            feature_names=['Fare', 'Embarked'])

    """

    # check model
    info_features = _make_list(features[0]) + _make_list(features[1])
    info_df, pred_cols = _prepare_actual_plot_data(
        model, X, n_classes, pred_func, predict_kwds, info_features, which_classes
    )

    return plot_interact(
        info_df,
        features,
        feature_names,
        pred_cols,
        num_grid_points,
        grid_types,
        percentile_ranges,
        grid_ranges,
        cust_grid_points,
        show_percentile,
        show_outliers,
        endpoint,
        figsize,
        dpi,
        ncols,
        annotate,
        plot_params,
        engine,
        template,
        "actual_interact",
    )


def plot_interact(
    df,
    features,
    feature_names,
    target,
    num_grid_points,
    grid_types,
    percentile_ranges,
    grid_ranges,
    cust_grid_points,
    show_percentile,
    show_outliers,
    endpoint,
    figsize,
    dpi,
    ncols,
    annotate,
    plot_params,
    engine,
    template,
    plot_type="target_interact",
):

    check_results = _check_info_plot_interact_params(
        df,
        features,
        grid_types,
        percentile_ranges,
        grid_ranges,
        num_grid_points,
        cust_grid_points,
        show_outliers,
    )
    (
        plot_data,
        target_plot_data,
        summary_df,
        display_columns,
        percentile_columns,
    ) = _prepare_info_plot_interact_data(
        features=features,
        target=target,
        data=df,
        show_percentile=show_percentile,
        endpoint=endpoint,
        **check_results
    )

    plot_params = _prepare_plot_params(
        plot_params,
        ncols,
        display_columns,
        percentile_columns,
        figsize,
        dpi,
        template,
        engine,
    )
    plot_params["annotate"] = annotate

    if engine == "matplotlib":
        fig, axes = _info_plot_interact(
            feature_names,
            target,
            plot_data,
            plot_params,
            plot_type,
        )
    else:
        fig = _info_plot_interact_plotly(
            feature_names,
            target,
            plot_data,
            plot_params,
            plot_type,
        )
        axes = None

    return fig, axes, summary_df
