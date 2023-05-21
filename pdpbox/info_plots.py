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

        Assign `self.target` with a list of target column names.
        Assign `self.n_classes` according to target type.
        Assign `self.df` with a DataFrame containing feature and target columns.
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
        Prepare for a predict plot.

        Update `self.n_classes` according to `model` and `n_classes`.
        Assign `self.target` with a list of pred column names.
        Assign `self.df` with a DataFrame containing feature and pred columns.
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

        Update `self.df` with feature bucket information.
        Assign `self.count_df` with the count of each feature bucket.
        Assign `self.summary_df` with the summary statistics for each feature bucket.
        """
        self.df, self.count_df, self.summary_df = self.feature_info.prepare(self.df)

    def agg_target(self):
        """
        Aggregate the target variable according to `self.plot_type`.

        If `self.plot_type` is `'target'`, the target variable is aggregated by 
        taking the mean of the target for each feature bucket `x`.
        If `self.plot_type` is `'predict'`, the target variable is aggregated by 
        taking the first quartile, median (second quartile), and third quartile 
        of the target for each feature bucket `x`.
        The aggregated results are stored in `self.target_lines`.
        `self.summary_df` is updated with the aggregated results.
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
            The plotting engine to use. Default is 'plotly'.
        template : str, optional
            The template to use for plotly plots. Only applicable when `engine` is
            'plotly'. Default is 'plotly_white'. Reference:
            https://plotly.com/python/templates/

        Returns
        -------
        matplotlib.figure.Figure or plotly.graph_objects.Figure : 
            A Matplotlib or Plotly figure object depending on the plot engine being
            used.
        dict of matplotlib.axes.Axes or None : 
            A dictionary of Matplotlib axes objects. The keys are the names of the
            axes. The values are the axes objects. If `engine` is 'ploltly', it is
            None.
        pd.DataFrame : 
            A Pandas DataFrame containing the summary statistics for each feature
            bucket.
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

    Attributes
    ----------
    df
    target
    n_classes : only applicable for `PredictPlot`
    plot_type : str
        The type of the plot to be generated. For this class, it's `'target'`.
    plot_engines : :ref:`param-plot_engines`
    feature_info : :ref:`param-feature_info`
    feature_cols : list of str
        List of feature column names.
    count_df : :ref:`param-count_df`
    summary_df : :ref:`param-summary_df`
    target_lines : :ref:`param-target_lines`
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
        Parameters
        ----------
        df : pd.DataFrame
            Input data frame at least containing the feature and target columns.
        feature : :ref:`param-feature`
        feature_name : :ref:`param-feature_name`
        target : str or list of str
            The target column(s) used for the y-axis in the plot.
        cust_grid_points : :ref:`param-cust_grid_points`
        grid_type : :ref:`param-grid_type`
        num_grid_points : :ref:`param-num_grid_points`
        percentile_range : :ref:`param-percentile_range`
        grid_range : :ref:`param-grid_range`
        show_outliers : :ref:`param-show_outliers`
        endpoint : :ref:`param-endpoint`
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
    Class for creating prediction plots based on input data and specified parameters.

    Attributes
    ----------
    df
    plot_type : str
        The type of the plot to be generated. For this class, it's `'predict'`.
    plot_engines : :ref:`param-plot_engines`
    feature_info : :ref:`param-feature_info`
    feature_cols : list of str
        List of feature column names.
    n_classes : :ref:`param-n_classes`
    target : only applicable for `TargetPlot`
    count_df : :ref:`param-count_df`
    summary_df : :ref:`param-summary_df`
    target_lines : :ref:`param-target_lines`
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
        """
        Parameters
        ----------
        df : pd.DataFrame
            Input data frame at least containing the feature and target columns.
        feature : :ref:`param-feature`
        feature_name : :ref:`param-feature_name`
        model : :ref:`param-model`
        model_features : :ref:`param-model_features`
        pred_func : :ref:`param-pred_func`
        n_classes : :ref:`param-n_classes`
        predict_kwds : :ref:`param-predict_kwds`
        chunk_size : :ref:`param-chunk_size`
        cust_grid_points : :ref:`param-cust_grid_points`
        grid_type : :ref:`param-grid_type`
        num_grid_points : :ref:`param-num_grid_points`
        percentile_range : :ref:`param-percentile_range`
        grid_range : :ref:`param-grid_range`
        show_outliers : :ref:`param-show_outliers`
        endpoint : :ref:`param-endpoint`
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
        Aggregate the target variable based on the interaction of two features.

        This function creates a DataFrame that contains the mean value of the target variable
        (for each unique pair of feature values) if the plot type is "interact_target". Otherwise, 
        it calculates the median value of the target variable. It also counts the number of instances
        for each unique pair of feature values. The result is stored in `self.summary_df` and `self.plot_df`.
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
        which_classes : :ref:`param-which_classes`
        show_percentile : :ref:`param-show_percentile`
        figsize : :ref:`param-figsize`
        dpi : :ref:`param-dpi`
        ncols : :ref:`param-ncols`
        annotate : :ref:`param-annotate`
        plot_params : :ref:`param-plot_params`
        engine : :ref:`param-engine`
        template : :ref:`param-template`

        Returns
        -------
        object :
            plotly or matplotlib figure object
        matplotlib axes object or None :
            None when `engine` is `"plotly"`
        pandas.DataFrame :
            summary_df
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

    Attributes
    ----------
    df
    target
    plot_type : str
        The type of the plot to be generated. For this class, it's `'interact_target'`.
    plot_engines : :ref:`param-plot_engines`
    feature_infos : :ref:`param-feature_infos`
    feature_cols : list of str
        List of feature column names.
    n_classes : only applicable for `InteractPredictPlot`
    summary_df : :ref:`param-summary_df`
    plot_df : :ref:`param-plot_df`
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
        Parameters
        ----------
        df : pd.DataFrame
            The input DataFrame.
        features : :ref:`param-features`
        feature_names : :ref:`param-feature_names`
        target : str or list of str
            The target column(s) used for the y-axis in the plot.
        num_grid_points : :ref:`param-num_grid_points`
        grid_types : :ref:`param-grid_types`
        percentile_ranges : :ref:`param-percentile_ranges`
        grid_ranges : :ref:`param-grid_ranges`
        cust_grid_points : :ref:`param-cust_grid_points`
        show_outliers : :ref:`param-show_outliers`
        endpoint : :ref:`param-endpoint`
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


class InterectPredictPlot(_InteractInfoPlot):
    """
    Class for visualizing the relationship between two features and their aggregated predictions from a model in an interactive 2D plot.

    Attributes
    ----------
    df
    target : only applicable for `InteractTargetPlot`
    plot_type : str
        The type of the plot to be generated. For this class, it's `'interact_predict'`.
    plot_engines : :ref:`param-plot_engines`
    feature_infos : :ref:`param-feature_infos`
    feature_cols : list of str
        List of feature column names.
    n_classes : only applicable for `InteractPredictPlot`
    summary_df : :ref:`param-summary_df`
    plot_df : :ref:`param-plot_df`
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
        """
        Parameters
        ----------
        df : pd.DataFrame
            The input DataFrame.
        features : :ref:`param-features`
        feature_names : :ref:`param-feature_names`
        model : :ref:`param-model`
        model_features : :ref:`param-model_features`
        pred_func : :ref:`param-pred_func`
        n_classes : :ref:`param-n_classes`
        predict_kwds : :ref:`param-predict_kwds`
        chunk_size : :ref:`param-chunk_size`
        num_grid_points : :ref:`param-num_grid_points`
        grid_types : :ref:`param-grid_types`
        percentile_ranges : :ref:`param-percentile_ranges`
        grid_ranges : :ref:`param-grid_ranges`
        cust_grid_points : :ref:`param-cust_grid_points`
        show_outliers : :ref:`param-show_outliers`
        endpoint : :ref:`param-endpoint`
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
