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

    def prepare_target(self, df, target):
        """
        Prepare the target data for plotting.

        Parameters
        ----------
        df : pandas.DataFrame
            The input DataFrame containing the feature and target columns.
        target : str or list of str
            The target column(s) to be used for plotting.
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

    def prepare_predict(
        self, model, df, pred_func, n_classes, predict_kwds, chunk_size
    ):
        """
        Prepare the prediction data for plotting.

        Parameters
        ----------
        model : object
            The trained model to be used for generating predictions.
        df : pandas.DataFrame
            The input DataFrame containing the feature columns.
        pred_func : callable
            The prediction function to be used with the model.
        n_classes : int
            The number of classes in the target variable.
        predict_kwds : dict
            The additional keyword arguments to be passed to the prediction function.
        chunk_size : int
            The number of samples to process at a time when calculating predictions.
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
    """
    Base class for creating information plots.

    Attributes
    ----------
    plot_type : str
        Type of plot to be generated, such as "target" or "predict".
    feature_info : FeatureInfo
        An instance of the FeatureInfo class, which contains information about the feature.
    feature_cols : list
        List of feature column names.
    target : list
        List of target column names.
    n_classes : int
        Number of classes in the target variable.
    df : pd.DataFrame
        The input data frame with selected columns.
    count_df : pd.DataFrame
        Data frame containing the count of observations for each bin.
    summary_df : pd.DataFrame
        Data frame containing the summary statistics for each bin.
    target_lines : list
        List of aggregated target lines for each target.
    """
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
        predict_kwds={},
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
            self.prepare_target(df, target)
        else:
            self.prepare_predict(
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
        self.df, self.count_df, self.summary_df = self.feature_info.prepare(self.df)

    def agg_target(self):
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
        Create the plot for the given data.

        Parameters
        ----------
        which_classes : list of int, optional
            The classes to be plotted, by default all classes will be plotted.
        show_percentile : bool, optional
            Whether to display percentiles on the plot, by default False.
        figsize : tuple of int, optional
            The size of the figure, by default None.
        dpi : int, optional
            The resolution of the plot, by default 300. Only used when engine='matplotlib'.
        ncols : int, optional
            The number of columns in the plot, by default 2.
        plot_params : dict, optional
            Additional plot parameters, by default None.
        engine : str, optional
            The plotting engine to use, either 'plotly' or 'matplotlib', by default 'plotly'.
        template : str, optional
            The plot template to use, by default 'plotly_white'. Only used when engine='plotly'.

        Returns
        -------
        tuple
            A tuple containing the figure, axes, and summary DataFrame.
            When engine='plotly', the axes will be None.
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
    Class for creating target plots based on input data and specified parameters.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame containing the data for plotting.
    feature : str
        Name of the feature column used in the plot.
    feature_name : str
        A user-defined name for the feature column.
    target : str or list of str
        The target column(s) used for the y-axis in the plot.
    cust_grid_points : array-like, optional
        Custom grid points to use for binning the data, by default None.
    grid_type : {'percentile', 'equal'}, optional
        The type of grid to use for binning the data. Default is 'percentile'.
    num_grid_points : int, optional
        The number of grid points to use for binning the data. Default is 10.
    percentile_range : tuple, optional
        Range of percentiles to include in the plot, by default None.
    grid_range : tuple, optional
        Range of values to include in the plot, by default None.
    show_outliers : bool, optional
        Whether to include outlier points in the plot, by default False.
    endpoint : bool, optional
        Whether to include the endpoint of the grid range, by default True.

    Attributes
    ----------
    See attributes of the _InfoPlot class.
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
    Class for creating prediction plots based on input data and specified parameters.

    Parameters
    ----------
    df : pd.DataFrame
        Input data frame at least containing the features for model prediction.
    feature : str
        Name of the feature column used in the plot.
    feature_name : str
        A user-defined name for the feature column.
    model : object
        A trained model object that can be used to generate predictions.
    model_features : list
        List of features used in the model.
    pred_func : callable, optional
        Prediction function to be used with the model, by default None.
    n_classes : int, optional
        Number of classes in the target variable, by default None.
    predict_kwds : dict, optional
        Keyword arguments to pass to the prediction function, by default {}.
    chunk_size : int, optional
        Size of data chunks when generating predictions, by default -1.
    cust_grid_points : array-like, optional
        Custom grid points to use for binning the data, by default None.
    grid_type : {'percentile', 'equal'}, optional
        The type of grid to use for binning the data. Default is 'percentile'.
    num_grid_points : int, optional
        The number of grid points to use for binning the data. Default is 10.
    percentile_range : tuple, optional
        Range of percentiles to include in the plot, by default None.
    grid_range : tuple, optional
        Range of values to include in the plot, by default None.
    show_outliers : bool, optional
        Whether to include outlier points in the plot, by default False.
    endpoint : bool, optional
        Whether to include the endpoint of the grid range, by default True.

    Attributes
    ----------
    See attributes of the _InfoPlot class.
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
        predict_kwds={},
        chunk_size=-1,
        cust_grid_points=None,
        grid_type="percentile",
        num_grid_points=10,
        percentile_range=None,
        grid_range=None,
        show_outliers=False,
        endpoint=True,
    ):
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
    """
    Base class for creating interactive information plots.

    Attributes
    ----------
    plot_type : str
        Type of plot to generate.
    feature_infos : list
        List of FeatureInfo objects for the features in the plot.
    feature_cols : list
        List of feature column names.
    df : pd.DataFrame
        Data frame containing the prepared data.
    target : list
        List of target column names.
    summary_df : pd.DataFrame
        Data frame containing the summary information.
    plot_df : pd.DataFrame
        Data frame containing the plot data.
    """
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
        predict_kwds={},
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
            self.prepare_target(df, target)
        else:
            self.prepare_predict(
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
        Generates the interactive plot with the specified options.

        Parameters
        ----------
        which_classes : list, optional
            List of class labels to include in the plot, by default None.
        show_percentile : bool, optional
            Whether to show percentile information in the plot, by default False.
        figsize : tuple, optional
            Tuple containing the figure width and height in inches, by default None.
        dpi : int, optional
            Dots per inch of the generated plot, by default 300. Only applicable when engine is "matplotlib".
        ncols : int, optional
            Number of columns for the plot layout, by default 2.
        annotate : bool, optional
            Whether to annotate plot points with additional information, by default False.
        plot_params : dict, optional
            Additional plot parameters, by default None.
        engine : str, optional
            Plotting engine to use, by default "plotly".
        template : str, optional
            Plotly template to use for the plot, by default "plotly_white". Only applicable when engine is "plotly".

        Returns
        -------
        tuple
            A tuple containing the figure, axes, and summary DataFrame.
            When engine='plotly', the axes will be None.
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


class InterectTargetPlot(_InteractInfoPlot):
    """
    Class for visualizing the relationship between two features and their aggregated target values in an interactive 2D plot.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe containing the features and target.
    features : list
        List of two feature names (or column names) to be plotted.
    feature_names : list
        List of two human-readable feature names to be displayed on the plot.
    target : str
        Target column name.
    num_grid_points : int, optional, default=10
        Number of grid points to be used for both features.
    grid_types : str or list, optional, default="percentile"
        Grid type for both features, either 'percentile' or 'equal', or a list containing the grid type for each feature.
    percentile_ranges : tuple or list, optional, default=None
        Tuple containing the minimum and maximum percentiles for both features, or a list of tuples for each feature.
    grid_ranges : tuple or list, optional, default=None
        Tuple containing the minimum and maximum grid values for both features, or a list of tuples for each feature.
    cust_grid_points : list or list of lists, optional, default=None
        Custom grid points for both features, or a list of custom grid points for each feature.
    show_outliers : bool, optional, default=False
        Whether to show outlier data points.
    endpoints : bool, optional, default=True
        Whether to include endpoints in the grid.

    Attributes
    ----------
    See attributes of the _InteractInfoPlot class.
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


class InterectPredictPlot(_InteractInfoPlot):
    """
    Class for visualizing the relationship between two features and their aggregated predictions from a model in an interactive 2D plot.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe containing the features and target.
    features : list
        List of two feature names (or column names) to be plotted.
    feature_names : list
        List of two human-readable feature names to be displayed on the plot.
    model : object
        Trained model object.
    model_features : list
        List of feature column names to be used for prediction.
    pred_func : callable, optional, default=None
        Custom prediction function. If not provided, the model's default predict or predict_proba method will be used.
    n_classes : int, optional, default=None
        Number of classes in the target variable. If not provided, will be inferred from the model or data.
    predict_kwds : dict, optional, default={}
        Additional keyword arguments to be passed to the prediction function.
    chunk_size : int, optional, default=-1
        Size of chunks for processing large datasets. If -1, the entire dataset will be processed at once.
    num_grid_points : int, optional, default=10
        Number of grid points to be used for both features.
    grid_types : str or list, optional, default="percentile"
        Grid type for both features, either 'percentile' or 'equal', or a list containing the grid type for each feature.
    percentile_ranges : tuple or list, optional, default=None
        Tuple containing the minimum and maximum percentiles for both features, or a list of tuples for each feature.
    grid_ranges : tuple or list, optional, default=None
        Tuple containing the minimum and maximum grid values for both features, or a list of tuples for each feature.
    cust_grid_points : list or list of lists, optional, default=None
        Custom grid points for both features, or a list of custom grid points for each feature.
    show_outliers : bool, optional, default=False
        Whether to show outlier data points.
    endpoints : bool, optional, default=True
        Whether to include endpoints in the grid.

    Attributes
    ----------
    See attributes of the _InteractInfoPlot class.
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
        predict_kwds={},
        chunk_size=-1,
        num_grid_points=10,
        grid_types="percentile",
        percentile_ranges=None,
        grid_ranges=None,
        cust_grid_points=None,
        show_outliers=False,
        endpoints=True,
    ):
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
