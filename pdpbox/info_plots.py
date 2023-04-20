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
        _check_target(target, df)
        self.target = _make_list(target)
        self.n_classes = len(self.target)
        use_cols = self.feature_cols + self.target
        self.df = df[use_cols]

    def prepare_predict(
        self, model, df, pred_func, n_classes, predict_kwds, chunk_size
    ):
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
