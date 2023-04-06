from .info_plot_utils import (
    _target_plot,
    _target_plot_plotly,
    _info_plot_interact,
    _predict_plot,
    _predict_plot_plotly,
    _prepare_info_plot_interact_data,
    _check_info_plot_interact_params,
    _info_plot_interact_plotly,
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
)


class InfoPlot:
    def __init__(
        self,
        df,
        feature,
        feature_name,
        num_grid_points=10,
        grid_type="percentile",
        percentile_range=None,
        grid_range=None,
        cust_grid_points=None,
        show_outliers=False,
        endpoint=True,
    ):
        _check_dataset(df)
        self.feature_info = FeatureInfo(
            feature,
            feature_name,
            df,
            cust_grid_points,
            grid_type,
            num_grid_points,
            percentile_range,
            grid_range,
            show_outliers,
            endpoint,
        )

    def prepare_feature(self, df):
        use_cols = _make_list(self.feature_info.col_name) + self.target
        self.df, self.count_df, self.summary_df = self.feature_info.prepare(
            df[use_cols]
        )

    def _plot(
        self,
        plot_func,
        show_percentile=False,
        figsize=None,
        dpi=300,
        ncols=2,
        plot_params=None,
        engine="plotly",
        template="plotly_white",
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
                "num_bins": self.feature_info.num_bins,
            }
        )
        fig, axes = plot_func(self, plot_params)

        return fig, axes, self.summary_df


class TargetPlot(InfoPlot):
    def __init__(
        self,
        df,
        feature,
        feature_name,
        target,
        num_grid_points=10,
        grid_type="percentile",
        percentile_range=None,
        grid_range=None,
        cust_grid_points=None,
        show_outliers=False,
        endpoint=True,
    ):
        super().__init__(
            df,
            feature,
            feature_name,
            num_grid_points,
            grid_type,
            percentile_range,
            grid_range,
            cust_grid_points,
            show_outliers,
            endpoint,
        )
        self.plot_type = "target"
        self.prepare_target(df, target)
        self.prepare_feature(df)
        self.agg_target()

    def prepare_target(self, df, target):
        _check_target(target, df)
        self.target = _make_list(target)

    def agg_target(self):
        def _agg_target(t):
            # prepare data for target lines
            # each target line contains 'x' and mean target value
            return (
                self.df.groupby("x", as_index=False)
                .agg({t: "mean"})
                .sort_values("x", ascending=True)
            )

        self.target_lines = [_agg_target(t) for t in self.target]
        for target_line in self.target_lines:
            self.summary_df = self.summary_df.merge(target_line, on="x", how="outer")

    def plot(
        self,
        show_percentile=False,
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
        """
        plot_func = _target_plot if engine == "matplotlib" else _target_plot_plotly
        return self._plot(
            plot_func,
            show_percentile,
            figsize,
            dpi,
            ncols,
            plot_params,
            engine,
            template,
        )


class PredictPlot(InfoPlot):
    def __init__(
        self,
        model,
        df,
        feature,
        feature_name,
        pred_func=None,
        n_classes=None,
        num_grid_points=10,
        grid_type="percentile",
        percentile_range=None,
        grid_range=None,
        cust_grid_points=None,
        show_outliers=False,
        endpoint=True,
        predict_kwds={},
        chunk_size=-1,
    ):
        super().__init__(
            df,
            feature,
            feature_name,
            num_grid_points,
            grid_type,
            percentile_range,
            grid_range,
            cust_grid_points,
            show_outliers,
            endpoint,
        )
        self.plot_type = "predict"
        df = self.prepare_target(
            model, df, pred_func, n_classes, predict_kwds, chunk_size
        )
        self.prepare_feature(df)
        self.agg_target()

    def prepare_target(self, model, df, pred_func, n_classes, predict_kwds, chunk_size):
        self.n_classes, pred_func, from_model = _check_model(
            model, n_classes, pred_func
        )

        preds = _calc_preds(model, df, pred_func, from_model, predict_kwds, chunk_size)

        df = df[_make_list(self.feature_info.col_name)]
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
        return df

    def agg_target(self):
        def _agg_target(t):
            # prepare data for box lines
            # each box line contains 'x' and actual prediction q1, q2, q3
            target_line = (
                self.df.groupby("x", as_index=False)
                .agg({t: [_q1, _q2, _q3]})
                .sort_values("x", ascending=True)
            )
            target_line.columns = [
                "".join(col) if col[1] != "" else col[0] for col in target_line.columns
            ]
            return target_line

        self.target_lines = [_agg_target(t) for t in self.target]
        for target_line in self.target_lines:
            self.summary_df = self.summary_df.merge(target_line, on="x", how="outer")

    def plot(
        self,
        show_percentile=False,
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
        """
        plot_func = _predict_plot if engine == "matplotlib" else _predict_plot_plotly
        return self._plot(
            plot_func,
            show_percentile,
            figsize,
            dpi,
            ncols,
            plot_params,
            engine,
            template,
        )


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
    """Plot average target value across different feature value combinations (feature grid combinations)"""

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
    chunk_size=-1,
    figsize=None,
    dpi=300,
    ncols=2,
    annotate=False,
    plot_params=None,
    engine="plotly",
    template="plotly_white",
):
    """Plot prediction distribution across different feature value combinations (feature grid combinations)"""

    # check model
    info_features = _make_list(features[0]) + _make_list(features[1])
    info_df, pred_cols = _prepare_actual_plot_data(
        model,
        X,
        n_classes,
        pred_func,
        predict_kwds,
        info_features,
        which_classes,
        chunk_size,
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
        **check_results,
    )

    if plot_params is None:
        plot_params = {}

    plot_params.update(
        {
            "ncols": ncols,
            "figsize": figsize,
            "dpi": dpi,
            "template": template,
            "engine": engine,
            "annotate": annotate,
            "display_columns": display_columns,
            "percentile_columns": percentile_columns,
            "show_percentile": show_percentile,
        }
    )

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
