"""Exploratory ARIMA-based early anomaly detection for one temperature sensor.

Method:
    1. Fit ARIMA(1, 1, 0) to the first 288 (24-hour) readings.
    2. Estimate a *provisional* upper forecast-error threshold from training
       residuals, excluding the first 20 fitted values for model initialization.
    3. Obtain one-step-ahead forecasts for subsequent readings. Every forecast
       uses only observations before that reading, with fixed fitted parameters.
    4. Flag positive errors above the threshold and save a CSV and plot.

This is a prototype, not a validated early-warning system. In particular,
exceeding a forecast-error threshold is not proof of an injected anomaly.
"""

import logging
import math
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from statsmodels.tsa.arima.model import ARIMA, ARIMAResultsWrapper

LOGGER = logging.getLogger(__name__)


def load_sensor_data(pth_Dataset: Path, str_SensorName: str) -> pd.DataFrame:
    """Read and validate one sensor and its regular five-minute timestamps."""
    if not pth_Dataset.is_file():
        raise FileNotFoundError(f"Excel workbook not found: {pth_Dataset}")

    df_SensorData: pd.DataFrame = pd.read_excel(
        pth_Dataset, sheet_name="Sensor_Data"
    )
    if df_SensorData.empty:
        raise ValueError("Sensor_Data worksheet is empty.")
    if "DateTime" not in df_SensorData.columns:
        raise ValueError("Sensor_Data is missing its 'DateTime' column.")
    if str_SensorName not in df_SensorData.columns:
        raise ValueError(f"Sensor column '{str_SensorName}' not found.")

    df_SensorData = df_SensorData[["DateTime", str_SensorName]].copy()
    df_SensorData["DateTime"] = pd.to_datetime(
        df_SensorData["DateTime"], errors="raise"
    )
    if df_SensorData["DateTime"].isna().any():
        raise ValueError("DateTime contains missing values.")

    ser_TimeGaps: pd.Series = df_SensorData["DateTime"].diff().dropna()
    if not ser_TimeGaps.eq(pd.Timedelta(minutes=5)).all():
        raise ValueError("Expected strictly ordered readings every five minutes.")

    df_SensorData[str_SensorName] = pd.to_numeric(
        df_SensorData[str_SensorName], errors="coerce"
    ).replace([float("inf"), float("-inf")], float("nan"))

    return df_SensorData.reset_index(drop=True)


def fit_arima_model(
    ser_TrainingTemperatures: pd.Series,
    tpl_ArimaOrder: tuple[int, int, int],
) -> ARIMAResultsWrapper:
    """Fit an ARIMA model using assumed-normal training data only."""
    if len(ser_TrainingTemperatures) < 30:
        raise ValueError("At least 30 training readings are required.")
    if ser_TrainingTemperatures.isna().any():
        raise ValueError("The initial training period contains missing readings.")

    obj_ArimaModel: ARIMA = ARIMA(
        ser_TrainingTemperatures,
        order=tpl_ArimaOrder,
    )
    obj_FittedModel: ARIMAResultsWrapper = obj_ArimaModel.fit()
    return obj_FittedModel


def estimate_error_threshold(
    ser_TrainingTemperatures: pd.Series,
    obj_FittedModel: ARIMAResultsWrapper,
    int_WarmupReadings: int,
    flt_SigmaMultiplier: float,
) -> tuple[float, float, float]:
    """Estimate mean, standard deviation and a provisional upper error limit.

    In-sample forecast errors can be optimistic; this is an exploratory
    calibration rule, not an independently validated false-alarm threshold.
    """
    if not math.isfinite(flt_SigmaMultiplier) or flt_SigmaMultiplier <= 0:
        raise ValueError("Sigma multiplier must be positive and finite.")
    if int_WarmupReadings < 0 or int_WarmupReadings >= len(ser_TrainingTemperatures) - 2:
        raise ValueError("Warmup must leave at least three training residuals.")

    ser_FittedPredictions: pd.Series = obj_FittedModel.fittedvalues
    ser_TrainingErrors: pd.Series = (
        ser_TrainingTemperatures - ser_FittedPredictions
    ).iloc[int_WarmupReadings:]

    flt_ErrorMean: float = float(ser_TrainingErrors.mean())
    flt_ErrorStd: float = float(ser_TrainingErrors.std(ddof=1))
    if not math.isfinite(flt_ErrorMean) or not math.isfinite(flt_ErrorStd) or flt_ErrorStd <= 0:
        raise ValueError("Training forecast errors must have finite, positive variability.")

    flt_ErrorThreshold: float = flt_ErrorMean + flt_SigmaMultiplier * flt_ErrorStd
    if flt_ErrorThreshold <= 0:
        raise ValueError("Computed threshold is not positive; inspect training residuals.")
    return flt_ErrorMean, flt_ErrorStd, flt_ErrorThreshold


def forecast_one_step_ahead(
    obj_FittedModel: ARIMAResultsWrapper,
    ser_TestTemperatures: pd.Series,
) -> pd.Series:
    """Return causal one-step-ahead forecasts without refitting parameters.

    `extend` runs the state-space filter through the new observations. Its
    `fittedvalues` are predictions for time t conditioned on readings up to t-1.
    This is equivalent to forecasting once, observing the real reading, updating
    the state, and repeating, but avoids thousands of costly Python calls.
    """
    if ser_TestTemperatures.empty:
        raise ValueError("No test readings remain after the training period.")

    obj_ExtendedModel: ARIMAResultsWrapper = obj_FittedModel.extend(
        ser_TestTemperatures
    )
    ser_Forecasts: pd.Series = obj_ExtendedModel.fittedvalues.copy()
    ser_Forecasts.name = "Forecast_Temperature_C"
    return ser_Forecasts


def load_sensor_events(pth_Dataset: Path, str_SensorName: str) -> pd.DataFrame:
    """Load event labels only for visual inspection, never for fitting ARIMA."""
    df_GroundTruth: pd.DataFrame = pd.read_excel(
        pth_Dataset, sheet_name="Ground_Truth_Spikes"
    )
    lst_RequiredColumns: list[str] = ["sensor", "start", "end", "event_id"]
    if not all(str_Column in df_GroundTruth.columns for str_Column in lst_RequiredColumns):
        raise ValueError("Ground_Truth_Spikes is missing required columns.")

    df_SensorEvents: pd.DataFrame = df_GroundTruth.loc[
        df_GroundTruth["sensor"] == str_SensorName
    ].copy()
    df_SensorEvents["start"] = pd.to_datetime(df_SensorEvents["start"])
    df_SensorEvents["end"] = pd.to_datetime(df_SensorEvents["end"])
    return df_SensorEvents.sort_values("start")


def plot_results(
    df_SensorData: pd.DataFrame,
    df_Results: pd.DataFrame,
    df_SensorEvents: pd.DataFrame,
    str_SensorName: str,
    flt_ErrorThreshold: float,
    pth_OutputPlot: Path,
) -> None:
    """Plot temperatures, one-step forecasts, errors, and labelled intervals."""
    fig_Plot: Figure = plt.figure(figsize=(13, 8))
    ax_Temperature: Axes = fig_Plot.add_subplot(2, 1, 1)
    ax_Error: Axes = fig_Plot.add_subplot(2, 1, 2, sharex=ax_Temperature)

    ax_Temperature.plot(
        df_SensorData["DateTime"],
        df_SensorData[str_SensorName],
        linewidth=0.8,
        label="Observed temperature",
    )
    ax_Temperature.plot(
        df_Results["DateTime"],
        df_Results["Forecast_Temperature_C"],
        linewidth=0.8,
        alpha=0.75,
        label="ARIMA one-step forecast",
    )
    ax_Temperature.set_title(f"ARIMA(1,1,0) — {str_SensorName}")
    ax_Temperature.set_ylabel("Temperature (°C)")

    ax_Error.plot(
        df_Results["DateTime"],
        df_Results["Forecast_Error_C"],
        linewidth=0.8,
        label="Actual minus forecast",
    )
    ax_Error.axhline(
        flt_ErrorThreshold,
        color="red",
        linestyle="--",
        label="Provisional upper error threshold",
    )
    ax_Error.axhline(0.0, color="gray", linewidth=0.7)
    ax_Error.scatter(
        df_Results.loc[df_Results["Positive_Anomaly_Flag"], "DateTime"],
        df_Results.loc[df_Results["Positive_Anomaly_Flag"], "Forecast_Error_C"],
        color="red",
        s=9,
        label="Flagged reading",
    )
    ax_Error.set_ylabel("Forecast error (°C)")
    ax_Error.set_xlabel("Timestamp")

    for int_EventNumber, (_, ser_Event) in enumerate(df_SensorEvents.iterrows()):
        for ax_Plot in (ax_Temperature, ax_Error):
            ax_Plot.axvspan(
                ser_Event["start"],
                ser_Event["end"],
                color="orange",
                alpha=0.12,
                label="Injected anomaly interval" if int_EventNumber == 0 else None,
            )

    ax_Temperature.legend(loc="upper left")
    ax_Error.legend(loc="upper left")
    fig_Plot.tight_layout()
    fig_Plot.savefig(pth_OutputPlot, dpi=160)
    if plt.get_backend().lower() != "agg":
        plt.show()
    plt.close(fig_Plot)


def main() -> None:
    """Run ARIMA forecasts for one temperature sensor in the synthetic workbook."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Configuration: edit these values for future experiments.
    pth_Dataset: Path = Path.home() / "Desktop" / "Synthetic_Data_Generation.xlsx"
    str_SensorName: str = "Sensor 1"
    int_BaselineReadings: int = 288
    tpl_ArimaOrder: tuple[int, int, int] = (1, 1, 0)
    int_WarmupReadings: int = 20
    flt_SigmaMultiplier: float = 3.0
    bol_ShowDiagnostics: bool = False

    df_SensorData: pd.DataFrame = load_sensor_data(pth_Dataset, str_SensorName)
    ser_Temperatures: pd.Series = df_SensorData[str_SensorName]
    if int_BaselineReadings >= len(ser_Temperatures):
        raise ValueError("Training period must be shorter than the full dataset.")

    ser_TrainingTemperatures: pd.Series = ser_Temperatures.iloc[:int_BaselineReadings]
    ser_TestTemperatures: pd.Series = ser_Temperatures.iloc[int_BaselineReadings:]

    obj_FittedModel: ARIMAResultsWrapper = fit_arima_model(
        ser_TrainingTemperatures, tpl_ArimaOrder
    )
    flt_TrainingErrorMean, flt_TrainingErrorStd, flt_ErrorThreshold = (
        estimate_error_threshold(
            ser_TrainingTemperatures,
            obj_FittedModel,
            int_WarmupReadings,
            flt_SigmaMultiplier,
        )
    )

    ser_Forecasts: pd.Series = forecast_one_step_ahead(
        obj_FittedModel, ser_TestTemperatures
    )
    ser_ForecastErrors: pd.Series = ser_TestTemperatures - ser_Forecasts
    ser_PositiveFlags: pd.Series = (
        ser_ForecastErrors.gt(flt_ErrorThreshold) & ser_ForecastErrors.notna()
    )
    ser_NewAlarmFlags: pd.Series = ser_PositiveFlags & ~ser_PositiveFlags.shift(
        fill_value=False
    )

    df_Results: pd.DataFrame = pd.DataFrame(
        {
            "DateTime": df_SensorData["DateTime"].iloc[int_BaselineReadings:],
            "Actual_Temperature_C": ser_TestTemperatures,
            "Forecast_Temperature_C": ser_Forecasts,
            "Forecast_Error_C": ser_ForecastErrors,
            "Positive_Anomaly_Flag": ser_PositiveFlags,
        }
    )
    df_SensorEvents: pd.DataFrame = load_sensor_events(
        pth_Dataset, str_SensorName
    )

    pth_OutputCsv: Path = pth_Dataset.parent / "arima_sensor_results.csv"
    pth_OutputPlot: Path = pth_Dataset.parent / "arima_sensor_results.png"
    df_Results.to_csv(pth_OutputCsv, index=False)

    print("\n========== ARIMA SENSOR ANALYSIS ==========")
    print(f"Sensor: {str_SensorName}")
    print(f"Total readings: {len(ser_Temperatures)}")
    print(f"Missing readings: {int(ser_Temperatures.isna().sum())}")
    print(f"Training readings: {len(ser_TrainingTemperatures)}")
    print(f"Forecasted readings: {len(ser_TestTemperatures)}")
    print(f"ARIMA order: {tpl_ArimaOrder}")
    print(f"Training error mean: {flt_TrainingErrorMean:.4f}°C")
    print(f"Training error std: {flt_TrainingErrorStd:.4f}°C")
    print(f"Upper error threshold: {flt_ErrorThreshold:.4f}°C")
    print(f"Ground-truth events: {len(df_SensorEvents)}")
    print(f"Flagged readings: {int(ser_PositiveFlags.sum())}")
    print(f"New flag episodes: {int(ser_NewAlarmFlags.sum())}")
    print(f"CSV saved: {pth_OutputCsv}")

    if bol_ShowDiagnostics:
        print("\nFitted ARIMA parameter estimates:")
        print(obj_FittedModel.params.to_string())
        print("\nGround-truth events for this sensor:")
        print(df_SensorEvents.to_string(index=False))
        print("\nFirst flagged readings (if any):")
        print(df_Results.loc[ser_PositiveFlags].head(20).to_string(index=False))
        print("\nFirst forecast rows:")
        print(df_Results.head(10).to_string(index=False))

    plot_results(
        df_SensorData,
        df_Results,
        df_SensorEvents,
        str_SensorName,
        flt_ErrorThreshold,
        pth_OutputPlot,
    )
    LOGGER.info("Plot saved: %s", pth_OutputPlot)


if __name__ == "__main__":
    main()
