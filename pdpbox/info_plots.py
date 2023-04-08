from .info_plot_utils import (
    _target_plot,
    _target_plot_plotly,
    _predict_plot,
    _predict_plot_plotly,
    _interact_info_plot,
    _interact_info_plot_plotly,
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

import numpy as np
import pandas as pd
from itertools import product


class _InfoPlot:
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
        which_classes=None,
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
        if which_classes is None:
            which_classes = list(np.arange(len(self.target)))
        else:
            _check_classes(which_classes, self.n_classes)
        fig, axes = plot_func(self, which_classes, plot_params)

        return fig, axes, self.summary_df


class TargetPlot(_InfoPlot):
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
        self.n_classes = len(self.target)

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
        Plot average target value by different feature values (feature grids)
        For binary or one-hot encoded features, it is very intuitive.
        """
        plot_func = _target_plot if engine == "matplotlib" else _target_plot_plotly
        return self._plot(
            plot_func,
            which_classes,
            show_percentile,
            figsize,
            dpi,
            ncols,
            plot_params,
            engine,
            template,
        )


class PredictPlot(_InfoPlot):
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
        Plot average target value by different feature values (feature grids)
        For binary or one-hot encoded features, it is very intuitive.
        """
        plot_func = _predict_plot if engine == "matplotlib" else _predict_plot_plotly
        return self._plot(
            plot_func,
            which_classes,
            show_percentile,
            figsize,
            dpi,
            ncols,
            plot_params,
            engine,
            template,
        )


class _InteractInfoPlot:
    def __init__(
        self,
        df,
        features,
        feature_names,
        num_grid_points=10,
        grid_types="percentile",
        percentile_ranges=None,
        grid_ranges=None,
        cust_grid_points=None,
        show_outliers=False,
        endpoints=True,
    ):
        _check_dataset(df)
        params = {
            "num_grid_points": num_grid_points,
            "grid_types": grid_types,
            "percentile_ranges": percentile_ranges,
            "grid_ranges": grid_ranges,
            "cust_grid_points": cust_grid_points,
            "show_outliers": show_outliers,
            "endpoints": endpoints,
        }
        params = _expand_params_for_interact(params)

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

    def prepare_feature(self, df):
        use_cols = []
        for i, feature_info in enumerate(self.feature_infos, start=1):
            df, _, _ = feature_info.prepare(df)
            df = df.rename(columns={"x": f"x{i}"})
            use_cols += _make_list(feature_info.col_name)
        use_cols += self.target
        return df[["x1", "x2"] + use_cols]

    def agg_target(self, df):
        df["count"] = 1
        agg_dict = {}
        for target in self.target:
            agg_dict[target] = "mean" if self.plot_type == "interact_target" else _q2
        agg_dict["count"] = "count"
        plot_df = df.groupby(["x1", "x2"], as_index=False).agg(agg_dict)

        x1_values = list(np.arange(df["x1"].min(), df["x1"].max() + 1))
        x2_values = list(np.arange(df["x2"].min(), df["x2"].max() + 1))
        summary_df = pd.DataFrame(
            [{"x1": x1, "x2": x2} for x1, x2 in list(product(x1_values, x2_values))]
        )
        for x_col in ["x1", "x2"]:
            summary_df[x_col] = summary_df[x_col].map(int)
        summary_df = summary_df.merge(plot_df, on=["x1", "x2"], how="left").fillna(0)

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

        summary_df = summary_df[info_cols + ["count"] + self.target]
        self.plot_df, self.summary_df = plot_df, summary_df

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
        Plot average target value by different feature values (feature grids)
        For binary or one-hot encoded features, it is very intuitive.
        """
        plot_func = (
            _interact_info_plot
            if engine == "matplotlib"
            else _interact_info_plot_plotly
        )

        if plot_params is None:
            plot_params = {}

        plot_params.update(
            {
                "ncols": ncols,
                "annotate": annotate,
                "figsize": figsize,
                "dpi": dpi,
                "template": template,
                "engine": engine,
                "show_percentile": show_percentile,
            }
        )
        if which_classes is None:
            which_classes = list(np.arange(len(self.target)))
        else:
            _check_classes(which_classes, self.n_classes)
        fig, axes = plot_func(self, which_classes, plot_params)

        return fig, axes, self.summary_df


class InterectTargetPlot(_InteractInfoPlot):
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
            num_grid_points,
            grid_types,
            percentile_ranges,
            grid_ranges,
            cust_grid_points,
            show_outliers,
            endpoints,
        )
        self.plot_type = "interact_target"
        self.prepare_target(df, target)
        df = self.prepare_feature(df)
        self.agg_target(df)

    def prepare_target(self, df, target):
        _check_target(target, df)
        self.target = _make_list(target)
        self.n_classes = len(self.target)


class InterectPredictPlot(_InteractInfoPlot):
    def __init__(
        self,
        model,
        df,
        features,
        feature_names,
        pred_func=None,
        n_classes=None,
        num_grid_points=10,
        grid_types="percentile",
        percentile_ranges=None,
        grid_ranges=None,
        cust_grid_points=None,
        show_outliers=False,
        endpoints=True,
        predict_kwds={},
        chunk_size=-1,
    ):
        super().__init__(
            df,
            features,
            feature_names,
            num_grid_points,
            grid_types,
            percentile_ranges,
            grid_ranges,
            cust_grid_points,
            show_outliers,
            endpoints,
        )
        self.plot_type = "interact_predict"
        df = self.prepare_target(
            model, df, pred_func, n_classes, predict_kwds, chunk_size
        )
        df = self.prepare_feature(df)
        self.agg_target(df)

    def prepare_target(self, model, df, pred_func, n_classes, predict_kwds, chunk_size):
        self.n_classes, pred_func, from_model = _check_model(
            model, n_classes, pred_func
        )

        preds = _calc_preds(model, df, pred_func, from_model, predict_kwds, chunk_size)
        df = df[
            _make_list(self.feature_infos[0].col_name)
            + _make_list(self.feature_infos[1].col_name)
        ]
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
