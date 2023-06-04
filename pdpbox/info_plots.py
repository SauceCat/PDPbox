from .info_plot_utils import (
    TargetPlotEngine,
    PredictPlotEngine,
    InteractInfoPlotEngine,
)
from .utils import (
    _make_list,
    _check_target,
    _q1,
    _q2,
    _q3,
    FeatureInfo,
    _check_dataset,
    _check_model,
    _calc_preds,
    _check_classes,
    _expand_params_for_interact,
)

import pandas as pd
from itertools import product


class _BaseInfoPlot:
    def __init__(self, plot_type):
        self.plot_type = plot_type
        self.plot_engines = {
            "target": TargetPlotEngine,
            "predict": PredictPlotEngine,
            "interact_target": InteractInfoPlotEngine,
            "interact_predict": InteractInfoPlotEngine,
        }

    def prepare_target_plot(self, df, target):
        """
        Prepare for a target plot.

        - Assign `self.target` with a list of target columns.
        - Assign `self.n_classes` according to target type.
        - Assign `self.df` with a DataFrame containing feature and target columns.
        """
        target_type = _check_target(target, df)
        self.target = _make_list(target)

        if target_type == "binary":
            self.n_classes = 2
        elif target_type == "multi-class":
            self.n_classes = len(self.target)
        else:
            self.n_classes = 0
        use_cols = self.feature_cols + self.target
        self.df = df[use_cols]

    def prepare_predict_plot(
        self, model, df, pred_func, n_classes, predict_kwds, chunk_size
    ):
        """
        Prepare for a prediction plot.

        - Update `self.n_classes` according to `model` and `n_classes`.
        - Assign `self.target` with a list of pred columns.
        - Assign `self.df` with a DataFrame containing feature and prediction columns.
        """
        self.n_classes, pred_func, from_model = _check_model(
            model, n_classes, pred_func
        )
        preds = _calc_preds(model, df, pred_func, from_model, predict_kwds, chunk_size)

        df = df[self.feature_cols]
        pred_cols = ["pred"]
        if self.n_classes == 0:
            df["pred"] = preds
        elif self.n_classes == 2:
            df["pred"] = preds[:, 1]
        else:
            pred_cols = [f"pred_{class_idx}" for class_idx in range(self.n_classes)]
            for idx, col in enumerate(pred_cols):
                df[col] = preds[:, idx]

        self.target = pred_cols
        self.df = df

    def _plot(
        self,
        which_classes=None,
        show_percentile=False,
        figsize=None,
        dpi=300,
        ncols=2,
        plot_params=None,
        engine="plotly",
        template="plotly_white",
        **kwargs,
    ):
        if plot_params is None:
            plot_params = {}

        plot_params.update(
            {
                "ncols": ncols,
                "figsize": figsize,
                "dpi": dpi,
                "template": template,
                "engine": engine,
                "show_percentile": show_percentile,
            }
        )
        plot_params.update(kwargs)

        which_classes = _check_classes(which_classes, self.n_classes)
        plot_engine = self.plot_engines[self.plot_type](
            self, which_classes, plot_params
        )
        fig, axes = plot_engine.plot()
        return fig, axes, self.summary_df


class _InfoPlot(_BaseInfoPlot):
    def __init__(
        self,
        df,
        feature,
        feature_name,
        target=None,
        model=None,
        model_features=None,
        pred_func=None,
        n_classes=None,
        predict_kwds=None,
        chunk_size=-1,
        plot_type="target",
        **kwargs,
    ):
        super().__init__(plot_type)
        _check_dataset(df, model_features)
        self.plot_type = plot_type
        self.feature_info = FeatureInfo(feature, feature_name, df, **kwargs)
        self.feature_cols = _make_list(self.feature_info.col_name)
        if self.plot_type == "target":
            self.prepare_target_plot(df, target)
        else:
            self.prepare_predict_plot(
                model,
                df[model_features],
                pred_func,
                n_classes,
                predict_kwds,
                chunk_size,
            )
        self.prepare_feature()
        self.agg_target()

    def prepare_feature(self):
        """
        Prepare the feature information.

        - Update `self.df` with feature bucket information.
        - Assign `self.count_df` with the count of each feature bucket.
        - Assign `self.summary_df` with the summary statistics for each feature bucket.
        """
        self.df, self.count_df, self.summary_df = self.feature_info.prepare(self.df)

    def agg_target(self):
        """
        Aggregates the target or prediction variable based on the plot type specified
        in `self.plot_type`.

        Depending on `self.plot_type`, this function employs different strategies to
        aggregate:

        - If `self.plot_type` is 'target', it calculates the mean of the target
        variable for each feature bucket 'x'.
        - If `self.plot_type` is 'predict', it calculates the first quartile, median
        (second quartile), and third quartile of the prediction variable for each
        feature bucket 'x'.

        The results of these aggregation operations are then stored in
        `self.target_lines`.

        Additionally, this function updates `self.summary_df` with the aggregate
        results to provide a summarized view of the target or prediction variable
        across different feature buckets.
        """
        if self.plot_type == "target":

            def _agg_target(t):
                return (
                    self.df.groupby("x", as_index=False)
                    .agg({t: "mean"})
                    .sort_values("x", ascending=True)
                )

        else:

            def _agg_target(t):
                target_line = (
                    self.df.groupby("x", as_index=False)
                    .agg({t: [_q1, _q2, _q3]})
                    .sort_values("x", ascending=True)
                )
                target_line.columns = [
                    "".join(col) if col[1] != "" else col[0]
                    for col in target_line.columns
                ]
                return target_line

        self.target_lines = [_agg_target(t) for t in self.target]
        for target_line in self.target_lines:
            self.summary_df = self.summary_df.merge(target_line, on="x", how="outer")

    def plot(
        self,
        which_classes=None,
        show_percentile=False,
        figsize=None,
        dpi=300,
        ncols=2,
        plot_params=None,
        engine="plotly",
        template="plotly_white",
    ):
        """
        The plot function for `TargetPlot` and `PredictPlot`.

        Parameters
        ----------
        which_classes : list of int, optional
            List of class indices to plot. If None, all classes will be plotted.
            Default is None.
        show_percentile : bool, optional
            If True, percentiles are shown in the plot. Default is False.
        figsize : tuple or None, optional
            The figure size for matplotlib or plotly figure. If None, the default
            figure size is used. Default is None.
        dpi : int, optional
            The resolution of the plot, measured in dots per inch. Only applicable when
            `engine` is 'matplotlib'. Default is 300.
        ncols : int, optional
            The number of columns of subplots in the figure. Default is 2.
        plot_params : dict or None, optional
            Custom plot parameters that control the style and aesthetics of the plot.
            Default is None.
        engine : {'matplotlib', 'plotly'}, optional
            The plotting engine to use. Default is plotly.
        template : str, optional
            The template to use for plotly plots. Only applicable when `engine` is
            'plotly'. Reference: https://plotly.com/python/templates/ Default is
            plotly_white.

        Returns
        -------
        matplotlib.figure.Figure or plotly.graph_objects.Figure
            A Matplotlib or Plotly figure object depending on the plot engine being
            used.
        dict of matplotlib.axes.Axes or None
            A dictionary of Matplotlib axes objects. The keys are the names of the
            axes. The values are the axes objects. If `engine` is 'ploltly', it is
            None.
        pd.DataFrame
            A DataFrame that contains the summary statistics of target (for target
            plot) or predict (for predict plot) values for each feature bucket.
        """
        return self._plot(
            which_classes,
            show_percentile,
            figsize,
            dpi,
            ncols,
            plot_params,
            engine,
            template,
            num_bins=self.feature_info.num_bins,
        )


class TargetPlot(_InfoPlot):
    """
    Generates plots displaying the average values of target variables
    across distinct groups (or buckets) of a single feature.

    These plots provide insights into how the target's average values change with
    respect to the different groups of a chosen feature. This understanding is
    crucial for comprehensive feature analysis and facilitates the interpretation
    of model predictions.

    Attributes
    ----------
    df : pd.DataFrame
        A processed DataFrame that includes feature and target (for target plot) or
        predict (for predict plot) columns, feature buckets, along with the count
        of samples within each bucket.
    feature_info : :class:`FeatureInfo`
        An instance of the `FeatureInfo` class.
    feature_cols : list of str
        List of feature columns.
    target : list of int
        List of target indices. For binary and regression problems, the list will
        be just [0]. For multi-class targets, the list is the class indices.
    n_classes : int
        The number of classes inferred from the target columns.
    plot_type : str
        The type of the plot to be generated.
    plot_engines : dict
        A dictionary that maps plot types to their plotting engines.
    count_df : pd.DataFrame
        A DataFrame that contains the count as well as the normalized count
        (percentage) of samples within each feature bucket.
    summary_df : pd.DataFrame
        A DataFrame that contains the summary statistics of target (for target
        plot) or predict (for predict plot) values for each feature bucket.
    target_lines : list of pd.DataFrame
        A list of DataFrames, each DataFrame includes aggregate metrics across
        feature buckets for a target (for target plot) or predict (for predict
        plot) variable. For binary and regression problems, the list will contain a
        single DataFrame. For multi-class targets, the list will contain a
        DataFrame for each class.

    Methods
    -------
    plot(**kwargs) :
        Generates the plot.
    """

    def __init__(
        self,
        df,
        feature,
        feature_name,
        target,
        cust_grid_points=None,
        grid_type="percentile",
        num_grid_points=10,
        percentile_range=None,
        grid_range=None,
        show_outliers=False,
        endpoint=True,
    ):
        """
        Initialize a `TargetPlot` instance.

        Parameters
        ----------
        df : pd.DataFrame
            A DataFrame that at least contains the feature(s) and target columns.
        feature : str or list of str
            The column name(s) of the chosen feature. It is a list of str when the
            chosen feature is one-hot encoded.
        feature_name : str
            A custom name for the chosen feature.
        target : str or list of str
            The target column or columns (when it is multi-class).
        cust_grid_points : array-like or list of arrays, optional
            Custom grid points for the feature. For interact plot, it can also be a
            list of two arrays, indicating the grid points for each feature. Default is
            None.
        grid_type : {'percentile', 'equal'}, optional
            The grid type. Only applicable for numeric feature. Default is percentile.
        num_grid_points : int or list of int, optional
            The number of grid points to use. Only applicable for numeric feature. For
            interact plot, it can also be a list of two integers, indicating the number
            of grid points for each feature. Default is 10.
        percentile_range : tuple, optional
            A tuple of two values indicating the range of percentiles to use. Only
            applicable for numeric feature and when `grid_type` is 'percentile'. If it
            is None, will use all samples. Default is None.
        grid_range : tuple, optional
            A tuple of two values indicating the range of grid values to use. Only
            applicable for numeric feature. If it is None, will use all samples.
            Default is None.
        show_outliers : bool or list of bool, optional
            Whether to show outliers in the plot. Only applicable for numeric feature.
            For interact plot, it can also be a list of two booleans, indicating
            whether to show outliers for each feature. Default is False.
        endpoint : bool, optional
            Whether to include the endpoint of the range. Default is True.
        """
        super().__init__(
            df,
            feature,
            feature_name,
            target,
            plot_type="target",
            cust_grid_points=cust_grid_points,
            grid_type=grid_type,
            num_grid_points=num_grid_points,
            percentile_range=percentile_range,
            grid_range=grid_range,
            show_outliers=show_outliers,
            endpoint=endpoint,
        )


class PredictPlot(_InfoPlot):
    """
    Generates box plots depicting the statistical distribution of prediction values
    across distinct groups (or buckets) of a single feature.

    The box plots illustrate the distribution of prediction values, with emphasis
    on the first quartile, median (second quartile), and third quartile, across
    different feature buckets. This visualization provides insights into the
    variation of predicted values with respect to different groups of the chosen
    feature. Such understanding is essential for comprehensive feature analysis
    and aids in interpreting model predictions.

    Attributes
    ----------
    df : pd.DataFrame
        A processed DataFrame that includes feature and target (for target plot) or
        predict (for predict plot) columns, feature buckets, along with the count
        of samples within each bucket.
    feature_info : :class:`FeatureInfo`
        An instance of the `FeatureInfo` class.
    feature_cols : list of str
        List of feature columns.
    target : list of int
        List of target indices. For binary and regression problems, the list will
        be just [0]. For multi-class targets, the list is the class indices.
    n_classes : int
        The number of classes provided, or inferred from the model when it is not
        provided.
    plot_type : str
        The type of the plot to be generated.
    plot_engines : dict
        A dictionary that maps plot types to their plotting engines.
    count_df : pd.DataFrame
        A DataFrame that contains the count as well as the normalized count
        (percentage) of samples within each feature bucket.
    summary_df : pd.DataFrame
        A DataFrame that contains the summary statistics of target (for target
        plot) or predict (for predict plot) values for each feature bucket.
    target_lines : list of pd.DataFrame
        A list of DataFrames, each DataFrame includes aggregate metrics across
        feature buckets for a target (for target plot) or predict (for predict
        plot) variable. For binary and regression problems, the list will contain a
        single DataFrame. For multi-class targets, the list will contain a
        DataFrame for each class.

    Methods
    -------
    plot(**kwargs) :
        Generates the plot.
    """

    def __init__(
        self,
        df,
        feature,
        feature_name,
        model,
        model_features,
        pred_func=None,
        n_classes=None,
        predict_kwds=None,
        chunk_size=-1,
        cust_grid_points=None,
        grid_type="percentile",
        num_grid_points=10,
        percentile_range=None,
        grid_range=None,
        show_outliers=False,
        endpoint=True,
    ):
        """
        Initialize a `PredictPlot` instance.

        Parameters
        ----------
        df : pd.DataFrame
            A DataFrame that at least contains columns specified by `model_features`.
        feature : str or list of str
            The column name(s) of the chosen feature. It is a list of str when the
            chosen feature is one-hot encoded.
        feature_name : str
            A custom name for the chosen feature.
        model : object
            A trained model object. The model should have a `predict` or
            `predict_proba` method. Otherwise a custom prediction function should be
            provided through `pred_func`.
        model_features : list of str
            A list of features used in model prediction.
        pred_func : callable, optional
            A custom prediction function. If not provided, `predict` or `predict_proba`
            method of `model` is used to generate the predictions. Default is None.
        n_classes : int, optional
            Number of classes. If it is None, will infer from `model.n_classes_`.
            Please set it as 0 for regression. Default is None.
        predict_kwds : dict, optional
            Additional keyword arguments to pass to the `model`'s predict function.
            Default is None.
        chunk_size : int, optional
            The number of samples to predict at each iteration. -1 means all samples at
            once. Default is -1.
        cust_grid_points : array-like or list of arrays, optional
            Custom grid points for the feature. For interact plot, it can also be a
            list of two arrays, indicating the grid points for each feature. Default is
            None.
        grid_type : {'percentile', 'equal'}, optional
            The grid type. Only applicable for numeric feature. Default is percentile.
        num_grid_points : int or list of int, optional
            The number of grid points to use. Only applicable for numeric feature. For
            interact plot, it can also be a list of two integers, indicating the number
            of grid points for each feature. Default is 10.
        percentile_range : tuple, optional
            A tuple of two values indicating the range of percentiles to use. Only
            applicable for numeric feature and when `grid_type` is 'percentile'. If it
            is None, will use all samples. Default is None.
        grid_range : tuple, optional
            A tuple of two values indicating the range of grid values to use. Only
            applicable for numeric feature. If it is None, will use all samples.
            Default is None.
        show_outliers : bool or list of bool, optional
            Whether to show outliers in the plot. Only applicable for numeric feature.
            For interact plot, it can also be a list of two booleans, indicating
            whether to show outliers for each feature. Default is False.
        endpoint : bool, optional
            Whether to include the endpoint of the range. Default is True.
        """
        super().__init__(
            df,
            feature,
            feature_name,
            None,
            model,
            model_features,
            pred_func,
            n_classes,
            predict_kwds,
            chunk_size,
            plot_type="predict",
            cust_grid_points=cust_grid_points,
            grid_type=grid_type,
            num_grid_points=num_grid_points,
            percentile_range=percentile_range,
            grid_range=grid_range,
            show_outliers=show_outliers,
            endpoint=endpoint,
        )


class _InteractInfoPlot(_BaseInfoPlot):
    def __init__(
        self,
        df,
        features,
        feature_names,
        target=None,
        model=None,
        model_features=None,
        pred_func=None,
        n_classes=None,
        predict_kwds=None,
        chunk_size=-1,
        plot_type="interact_target",
        **kwargs,
    ):
        super().__init__(plot_type)
        _check_dataset(df, model_features)
        params = _expand_params_for_interact(kwargs)
        self.plot_type = plot_type
        self.feature_infos = [
            FeatureInfo(
                features[i],
                feature_names[i],
                df,
                params["cust_grid_points"][i],
                params["grid_types"][i],
                params["num_grid_points"][i],
                params["percentile_ranges"][i],
                params["grid_ranges"][i],
                params["show_outliers"][i],
                params["endpoints"][i],
            )
            for i in range(2)
        ]
        self.feature_cols = _make_list(self.feature_infos[0].col_name) + _make_list(
            self.feature_infos[1].col_name
        )
        if self.plot_type == "interact_target":
            self.prepare_target_plot(df, target)
        else:
            self.prepare_predict_plot(
                model,
                df[model_features],
                pred_func,
                n_classes,
                predict_kwds,
                chunk_size,
            )
        self.prepare_feature()
        self.agg_target()

    def prepare_feature(self):
        for i, feature_info in enumerate(self.feature_infos, start=1):
            self.df, _, _ = feature_info.prepare(self.df)
            self.df = self.df.rename(columns={"x": f"x{i}"})

    def agg_target(self):
        """
        Aggregates the target or prediction variable according to the chosen plot
        type.

        This method creates a DataFrame that contains aggregate measures of the target
        variable across different feature buckets (distinct 'x1' and 'x2'
        combinations). The aggregation strategy depends on the selected plot type:

        - If the plot type is "interact_target", the target variable is aggregated by
        calculating the mean for each unique pair of feature values ('x1' and 'x2').
        - If the plot type is not "interact_target", the prediction variable is
        aggregated by calculating the median for each unique pair of feature values
        ('x1' and 'x2').

        It also determines the count of instances for each unique pair of feature
        values. The resulting DataFrame is saved in `self.summary_df`, while
        `self.plot_df` stores the aggregated results for each unique pair of feature
        values.

        This method also updates feature values in `self.summary_df` with their
        corresponding display and percentile values from the `FeatureInfo` objects.
        """
        self.df["count"] = 1
        target_agg_func = "mean" if self.plot_type == "interact_target" else _q2
        agg_dict = {target: target_agg_func for target in self.target}
        agg_dict["count"] = "count"
        plot_df = self.df.groupby(["x1", "x2"], as_index=False).agg(agg_dict)

        x1_vals = list(range(self.df["x1"].min(), self.df["x1"].max() + 1))
        x2_vals = list(range(self.df["x2"].min(), self.df["x2"].max() + 1))
        summary_df = pd.DataFrame(list(product(x1_vals, x2_vals)), columns=["x1", "x2"])
        summary_df = summary_df.merge(plot_df, on=["x1", "x2"], how="left").fillna(0)
        summary_df[["x1", "x2"]] = summary_df[["x1", "x2"]].astype(int)

        info_cols = ["x1", "x2", "value_1", "value_2"]
        for i, feature_info in enumerate(self.feature_infos, start=1):
            summary_df[f"value_{i}"] = summary_df[f"x{i}"].apply(
                lambda x: feature_info.display_columns[x]
            )
            if len(feature_info.percentile_columns):
                summary_df[f"percentile_{i}"] = summary_df[f"x{i}"].apply(
                    lambda x: feature_info.percentile_columns[x]
                )
                info_cols.append(f"percentile_{i}")

        self.summary_df = summary_df[info_cols + ["count"] + self.target]
        self.plot_df = plot_df

    def plot(
        self,
        which_classes=None,
        show_percentile=False,
        figsize=None,
        dpi=300,
        ncols=2,
        annotate=False,
        plot_params=None,
        engine="plotly",
        template="plotly_white",
    ):
        """
        Generate the interaction plot.

        Parameters
        ----------
        which_classes : list of int, optional
            List of class indices to plot. If None, all classes will be plotted.
            Default is None.
        show_percentile : bool, optional
            If True, percentiles are shown in the plot. Default is False.
        figsize : tuple or None, optional
            The figure size for matplotlib or plotly figure. If None, the default
            figure size is used. Default is None.
        dpi : int, optional
            The resolution of the plot, measured in dots per inch. Only applicable when
            `engine` is 'matplotlib'. Default is 300.
        ncols : int, optional
            The number of columns of subplots in the figure. Default is 2.
        annotate : bool, optional
            If it is True, the circles on the plot will be annotated with detailed
            information. Default is False.
        plot_params : dict or None, optional
            Custom plot parameters that control the style and aesthetics of the plot.
            Default is None.
        engine : {'matplotlib', 'plotly'}, optional
            The plotting engine to use. Default is plotly.
        template : str, optional
            The template to use for plotly plots. Only applicable when `engine` is
            'plotly'. Reference: https://plotly.com/python/templates/ Default is
            plotly_white.

        Returns
        -------
        matplotlib.figure.Figure or plotly.graph_objects.Figure
            A Matplotlib or Plotly figure object depending on the plot engine being
            used.
        dict of matplotlib.axes.Axes or None
            A dictionary of Matplotlib axes objects. The keys are the names of the
            axes. The values are the axes objects. If `engine` is 'ploltly', it is
            None.
        pd.DataFrame
            A DataFrame that contains the summary statistics of target (for target
            plot) or predict (for predict plot) values for each feature bucket.
        """
        return self._plot(
            which_classes,
            show_percentile,
            figsize,
            dpi,
            ncols,
            plot_params,
            engine,
            template,
            annotate=annotate,
        )


class InteractTargetPlot(_InteractInfoPlot):
    """
    `TargetPlot` for interaction between two features.

    Attributes
    ----------
    df : pd.DataFrame
        A processed DataFrame that includes feature and target (for target plot) or
        predict (for predict plot) columns, feature buckets, along with the count
        of samples within each bucket.
    feature_infos : list of :class:`FeatureInfo`
        A list of `FeatureInfo` objects.
    feature_cols : list of str
        List of feature columns.
    target : list of int
        List of target indices. For binary and regression problems, the list will
        be just [0]. For multi-class targets, the list is the class indices.
    n_classes : int
        The number of classes inferred from the target columns.
    plot_type : str
        The type of the plot to be generated.
    plot_engines : dict
        A dictionary that maps plot types to their plotting engines.
    summary_df : pd.DataFrame
        A DataFrame that contains the summary statistics of target (for target
        plot) or predict (for predict plot) values for each feature bucket.
    plot_df : pd.DataFrame
        A DataFrame that contains the aggregated target (for target plot) or
        predict (for predict plot) values for each unique pair of feature values.

    Methods
    -------
    plot(**kwargs) :
        Generates the plot.
    """

    def __init__(
        self,
        df,
        features,
        feature_names,
        target,
        num_grid_points=10,
        grid_types="percentile",
        percentile_ranges=None,
        grid_ranges=None,
        cust_grid_points=None,
        show_outliers=False,
        endpoints=True,
    ):
        """
        Initialize an `InteractTargetPlot` instance.

        Parameters
        ----------
        df : pd.DataFrame
            A DataFrame that at least contains the feature(s) and target columns.
        features : list
            List of column name(s) for the 2 chosen features. The length of the list
            should be strictly 2.
        feature_names : list
            List of custom names for the 2 chosen features. The length of the list
            should be strictly 2.
        target : str or list of str
            The target column or columns (when it is multi-class).
        num_grid_points : int or list of int, optional
            The number of grid points to use. Only applicable for numeric feature. For
            interact plot, it can also be a list of two integers, indicating the number
            of grid points for each feature. Default is 10.
        grid_types : str or list of str, optional
            Same as `grid_type`, but could be a list of two strings, indicating the
            grid type for each feature. Default is percentile.
        percentile_ranges : tuple or a list of tuples, optional
            Same as `percentile_range`, but could be a list of two tuples, indicating
            the percentile range for each feature. Default is None.
        grid_ranges : tuple or list of tuples, optional
            Same as `grid_range`, but could be a list of two tuples, indicating the
            grid range for each feature. Default is None.
        cust_grid_points : array-like or list of arrays, optional
            Custom grid points for the feature. For interact plot, it can also be a
            list of two arrays, indicating the grid points for each feature. Default is
            None.
        show_outliers : bool or list of bool, optional
            Whether to show outliers in the plot. Only applicable for numeric feature.
            For interact plot, it can also be a list of two booleans, indicating
            whether to show outliers for each feature. Default is False.
        endpoints : bool or list of bool, optional
            Same as `endpoint`, but could be a list of two booleans, indicating whether
            to include the endpoint for each feature. Default is True.
        """
        super().__init__(
            df,
            features,
            feature_names,
            target,
            plot_type="interact_target",
            num_grid_points=num_grid_points,
            grid_types=grid_types,
            percentile_ranges=percentile_ranges,
            grid_ranges=grid_ranges,
            cust_grid_points=cust_grid_points,
            show_outliers=show_outliers,
            endpoints=endpoints,
        )


class InteractPredictPlot(_InteractInfoPlot):
    """
    `PredictPlot` for interaction between two features.

    Attributes
    ----------
    df : pd.DataFrame
        A processed DataFrame that includes feature and target (for target plot) or
        predict (for predict plot) columns, feature buckets, along with the count
        of samples within each bucket.
    feature_infos : list of :class:`FeatureInfo`
        A list of `FeatureInfo` objects.
    feature_cols : list of str
        List of feature columns.
    target : list of int
        List of target indices. For binary and regression problems, the list will
        be just [0]. For multi-class targets, the list is the class indices.
    n_classes : int
        The number of classes provided, or inferred from the model when it is not
        provided.
    plot_type : str
        The type of the plot to be generated.
    plot_engines : dict
        A dictionary that maps plot types to their plotting engines.
    summary_df : pd.DataFrame
        A DataFrame that contains the summary statistics of target (for target
        plot) or predict (for predict plot) values for each feature bucket.
    plot_df : pd.DataFrame
        A DataFrame that contains the aggregated target (for target plot) or
        predict (for predict plot) values for each unique pair of feature values.

    Methods
    -------
    plot(**kwargs) :
        Generates the plot.
    """

    def __init__(
        self,
        df,
        features,
        feature_names,
        model,
        model_features,
        pred_func=None,
        n_classes=None,
        predict_kwds=None,
        chunk_size=-1,
        num_grid_points=10,
        grid_types="percentile",
        percentile_ranges=None,
        grid_ranges=None,
        cust_grid_points=None,
        show_outliers=False,
        endpoints=True,
    ):
        """
        Initialize an `InteractPredictPlot` instance.

        Parameters
        ----------
        df : pd.DataFrame
            A DataFrame that at least contains columns specified by `model_features`.
        features : list
            List of column name(s) for the 2 chosen features. The length of the list
            should be strictly 2.
        feature_names : list
            List of custom names for the 2 chosen features. The length of the list
            should be strictly 2.
        model : object
            A trained model object. The model should have a `predict` or
            `predict_proba` method. Otherwise a custom prediction function should be
            provided through `pred_func`.
        model_features : list of str
            A list of features used in model prediction.
        pred_func : callable, optional
            A custom prediction function. If not provided, `predict` or `predict_proba`
            method of `model` is used to generate the predictions. Default is None.
        n_classes : int, optional
            Number of classes. If it is None, will infer from `model.n_classes_`.
            Please set it as 0 for regression. Default is None.
        predict_kwds : dict, optional
            Additional keyword arguments to pass to the `model`'s predict function.
            Default is None.
        chunk_size : int, optional
            The number of samples to predict at each iteration. -1 means all samples at
            once. Default is -1.
        num_grid_points : int or list of int, optional
            The number of grid points to use. Only applicable for numeric feature. For
            interact plot, it can also be a list of two integers, indicating the number
            of grid points for each feature. Default is 10.
        grid_types : str or list of str, optional
            Same as `grid_type`, but could be a list of two strings, indicating the
            grid type for each feature. Default is percentile.
        percentile_ranges : tuple or a list of tuples, optional
            Same as `percentile_range`, but could be a list of two tuples, indicating
            the percentile range for each feature. Default is None.
        grid_ranges : tuple or list of tuples, optional
            Same as `grid_range`, but could be a list of two tuples, indicating the
            grid range for each feature. Default is None.
        cust_grid_points : array-like or list of arrays, optional
            Custom grid points for the feature. For interact plot, it can also be a
            list of two arrays, indicating the grid points for each feature. Default is
            None.
        show_outliers : bool or list of bool, optional
            Whether to show outliers in the plot. Only applicable for numeric feature.
            For interact plot, it can also be a list of two booleans, indicating
            whether to show outliers for each feature. Default is False.
        endpoints : bool or list of bool, optional
            Same as `endpoint`, but could be a list of two booleans, indicating whether
            to include the endpoint for each feature. Default is True.
        """
        super().__init__(
            df,
            features,
            feature_names,
            None,
            model,
            model_features,
            pred_func,
            n_classes,
            predict_kwds,
            chunk_size,
            plot_type="interact_predict",
            num_grid_points=num_grid_points,
            grid_types=grid_types,
            percentile_ranges=percentile_ranges,
            grid_ranges=grid_ranges,
            cust_grid_points=cust_grid_points,
            show_outliers=show_outliers,
            endpoints=endpoints,
        )
