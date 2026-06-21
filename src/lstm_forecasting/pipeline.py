"""
End-to-end CLI orchestration:

    1. Ingest OHLCV data (yfinance) for one or more tickers.
    2. Engineer technical-indicator features (EMA, RSI, MACD, ...).
    3. Walk-forward TimeSeriesSplit cross-validation of an LSTM model.
    4. Fit ARIMA/GARCH baselines on the same final test window.
    5. Evaluate all models with RMSE/MAE/MAPE + forecasting-lag, save
       plots and a summary report to the artifacts directory.

Run via the installed console script:

    lstm-forecast --ticker AAPL --epochs 50

or directly:

    python -m src.lstm_forecasting.pipeline --ticker AAPL
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from . import baselines, data_ingestion, evaluate, indicators, models, preprocessing
from .logging_utils import get_logger

logger = get_logger(__name__)


def load_config(path: str | Path | None) -> dict:
    """Load YAML config, falling back to the bundled default if none given."""
    if path is None:
        path = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"
    with open(path, "r") as f:
        return yaml.safe_load(f)


def run_for_ticker(ticker: str, cfg: dict, artifacts_dir: Path) -> dict:
    """Run the full ingestion -> features -> CV -> baseline -> eval pipeline.

    Returns a JSON-serializable summary dict of metrics for this ticker.
    """
    ticker_dir = artifacts_dir / ticker.replace("/", "-")
    ticker_dir.mkdir(parents=True, exist_ok=True)

    # 1. Ingestion
    raw_df = data_ingestion.fetch_ohlcv(
        ticker,
        start=cfg["data"]["start"],
        end=cfg["data"]["end"],
        interval=cfg["data"]["interval"],
        cache_dir=cfg["data"]["cache_dir"],
    )
    raw_df = raw_df.reset_index().rename(columns={"index": "Date"})
    if "Date" not in raw_df.columns:
        raw_df = raw_df.rename(columns={raw_df.columns[0]: "Date"})

    # 2. Feature engineering
    feat_cfg = cfg["features"]
    feature_df = indicators.add_technical_indicators(
        raw_df.set_index("Date"),
        price_col=feat_cfg["price_col"],
        ema_spans=tuple(feat_cfg["ema_spans"]),
        rsi_window=feat_cfg["rsi_window"],
        macd_params=tuple(feat_cfg["macd"]),
        bb_window=feat_cfg["bb_window"],
        vol_window=feat_cfg["vol_window"],
    )
    feature_df = feature_df.rename(columns={feature_df.columns[0]: "Date"})
    logger.info(
        "[%s] Feature-engineered dataset: %d rows, %d columns",
        ticker,
        *feature_df.shape,
    )

    # 3. Walk-forward CV for the LSTM
    win_cfg = cfg["windowing"]
    model_cfg = models.LSTMConfig(**cfg["model"])

    cv_fold_metrics = []
    last_fold = None
    for fold_idx, fold in enumerate(
        preprocessing.time_series_cv_splits(
            feature_df,
            target_column=feat_cfg["price_col"],
            lookback=win_cfg["lookback"],
            horizon=win_cfg["horizon"],
            n_splits=win_cfg["n_cv_splits"],
        ),
        start=1,
    ):
        n_features = fold.X_train.shape[-1]
        lstm = models.build_lstm_model(
            input_shape=(win_cfg["lookback"], n_features), config=model_cfg
        )
        models.train_model(
            lstm, fold.X_train, fold.y_train, config=model_cfg, verbose=0
        )

        preds_scaled = models.predict(lstm, fold.X_test)
        y_pred = fold.target_scaler.inverse_transform(
            preds_scaled.reshape(-1, 1)
        ).ravel()
        y_true = fold.target_scaler.inverse_transform(
            fold.y_test.reshape(-1, 1)
        ).ravel()

        fold_metrics = evaluate.evaluate_predictions(
            y_true, y_pred, name=f"LSTM_fold{fold_idx}"
        )
        cv_fold_metrics.append(fold_metrics)
        last_fold = (fold, y_true, y_pred, lstm)

    avg_rmse = (
        float(np.mean([m.rmse for m in cv_fold_metrics]))
        if cv_fold_metrics
        else float("nan")
    )
    avg_mae = (
        float(np.mean([m.mae for m in cv_fold_metrics]))
        if cv_fold_metrics
        else float("nan")
    )
    logger.info(
        "[%s] LSTM CV summary: avg RMSE=%.5f avg MAE=%.5f", ticker, avg_rmse, avg_mae
    )

    summary: dict = {
        "ticker": ticker,
        "lstm_cv_avg_rmse": avg_rmse,
        "lstm_cv_avg_mae": avg_mae,
        "lstm_cv_folds": [
            {"fold": i + 1, "rmse": m.rmse, "mae": m.mae, "mape": m.mape, "lag": m.lag}
            for i, m in enumerate(cv_fold_metrics)
        ],
    }

    # 4 & 5. Baselines + final-fold comparison plots on last CV fold test window
    if last_fold is not None:
        fold, y_true, y_pred, lstm = last_fold
        price_series = feature_df.set_index("Date")[feat_cfg["price_col"]]
        test_size = min(cfg["baselines"]["test_size"], len(y_true))

        try:
            arima_result = baselines.rolling_arima_forecast(
                price_series,
                order=tuple(cfg["baselines"]["arima_order"]),
                test_size=test_size,
            )
            arima_metrics = evaluate.evaluate_predictions(
                arima_result.y_true, arima_result.y_pred, name=arima_result.name
            )
            summary["arima"] = {
                "rmse": arima_metrics.rmse,
                "mae": arima_metrics.mae,
                "mape": arima_metrics.mape,
                "lag": arima_metrics.lag,
            }
        except Exception as exc:  # noqa: BLE001
            logger.error("[%s] ARIMA baseline failed: %s", ticker, exc)
            summary["arima"] = {"error": str(exc)}

        try:
            returns = indicators.log_returns(price_series)
            garch_result = baselines.rolling_garch_volatility_forecast(
                returns,
                test_size=test_size,
                p=cfg["baselines"]["garch_p"],
                q=cfg["baselines"]["garch_q"],
            )
            garch_metrics = evaluate.evaluate_predictions(
                garch_result.y_true,
                garch_result.y_pred,
                name=garch_result.name,
                max_lag=5,
            )
            summary["garch"] = {
                "rmse": garch_metrics.rmse,
                "mae": garch_metrics.mae,
                "mape": garch_metrics.mape,
                "lag": garch_metrics.lag,
            }
        except Exception as exc:  # noqa: BLE001
            logger.error("[%s] GARCH baseline failed: %s", ticker, exc)
            summary["garch"] = {"error": str(exc)}

        plot_dates = (
            fold.test_dates.iloc[-len(y_true) :]
            if fold.test_dates is not None
            else range(len(y_true))
        )
        evaluate.plot_predictions(
            plot_dates,
            y_true,
            {"LSTM": y_pred},
            title=f"{ticker}: LSTM Predicted vs. Actual Close",
            save_path=str(ticker_dir / "lstm_predictions.png"),
        )

        comparison_metrics = [
            evaluate.Metrics(
                rmse=summary["lstm_cv_avg_rmse"],
                mae=summary["lstm_cv_avg_mae"],
                mape=float(np.mean([m.mape for m in cv_fold_metrics])),
                lag=int(np.round(np.mean([m.lag for m in cv_fold_metrics]))),
                name="LSTM (CV avg)",
            )
        ]
        if "rmse" in summary.get("arima", {}):
            comparison_metrics.append(
                evaluate.Metrics(**summary["arima"], name="ARIMA")
            )
        evaluate.plot_metric_comparison(
            comparison_metrics, save_path=str(ticker_dir / "model_comparison.png")
        )

    with open(ticker_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    logger.info("[%s] Summary written to %s", ticker, ticker_dir / "summary.json")

    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="LSTM-based short-term equity price forecasting pipeline."
    )
    parser.add_argument(
        "--ticker",
        action="append",
        dest="tickers",
        help="Ticker symbol to run (repeatable, e.g. --ticker AAPL --ticker D05.SI). "
        "Defaults to the list in the config file.",
    )
    parser.add_argument(
        "--config", type=str, default=None, help="Path to a YAML config file."
    )
    parser.add_argument(
        "--start",
        type=str,
        default=None,
        help="Override history start date (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--epochs", type=int, default=None, help="Override LSTM training epochs."
    )
    parser.add_argument(
        "--lookback", type=int, default=None, help="Override sequence lookback window."
    )
    parser.add_argument(
        "--artifacts-dir",
        type=str,
        default=None,
        help=(
            "Override output directory for plots/metrics "
            "(default: configs/default.yaml output.artifacts_dir)."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    cfg = load_config(args.config)

    if args.tickers:
        cfg["data"]["tickers"] = args.tickers
    if args.start:
        cfg["data"]["start"] = args.start
    if args.epochs:
        cfg["model"]["epochs"] = args.epochs
    if args.lookback:
        cfg["windowing"]["lookback"] = args.lookback
    if args.artifacts_dir:
        cfg["output"]["artifacts_dir"] = args.artifacts_dir

    artifacts_dir = Path(cfg["output"]["artifacts_dir"])
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    all_summaries = {}
    for ticker in cfg["data"]["tickers"]:
        logger.info("=== Running pipeline for %s ===", ticker)
        try:
            all_summaries[ticker] = run_for_ticker(ticker, cfg, artifacts_dir)
        except Exception as exc:  # noqa: BLE001
            logger.error("Pipeline failed for %s: %s", ticker, exc, exc_info=True)
            all_summaries[ticker] = {"error": str(exc)}

    with open(artifacts_dir / "all_tickers_summary.json", "w") as f:
        json.dump(all_summaries, f, indent=2, default=float)
    logger.info(
        "All done. Combined summary at %s", artifacts_dir / "all_tickers_summary.json"
    )


if __name__ == "__main__":
    main()
