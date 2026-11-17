"""Exploratory upper-sided CUSUM detection for one temperature sensor.

Assumption: the first BASELINE_READINGS measurements represent normal operation.
This is a prototype, not a validated early-warning system.
"""

from pathlib import Path
from typing import Optional
import logging
import math

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import pandas as pd

LOGGER = logging.getLogger(__name__)


def load_sensor_data(pth_Dataset: Path, int_SheetIndex: int) -> pd.DataFrame:
    """Load a single Excel worksheet and validate it is not empty."""
    if not pth_Dataset.is_file():
        raise FileNotFoundError(f"Dataset not found: {pth_Dataset}")

    df_SensorData: pd.DataFrame = pd.read_excel(
        pth_Dataset, sheet_name=int_SheetIndex
    )
    if df_SensorData.empty:
        raise ValueError("The selected worksheet contains no rows.")

    return df_SensorData


def select_sensor(
    df_SensorData: pd.DataFrame, str_RequestedSensorName: Optional[str]
) -> str:
    """Select a specified sensor, or the first numeric sensor column."""
    if str_RequestedSensorName is not None:
        if str_RequestedSensorName not in df_SensorData.columns:
            raise ValueError(
                f"Sensor '{str_RequestedSensorName}' not found. "
                f"Available columns: {df_SensorData.columns.tolist()}"
            )
        return str_RequestedSensorName

    lst_SensorNames: list[str] = [
        str_ColumnName
        for str_ColumnName in df_SensorData.columns
        if isinstance(str_ColumnName, str)
        and pd.api.types.is_numeric_dtype(df_SensorData[str_ColumnName])
        and not any(
            str_Excluded in str_ColumnName.lower()
            for str_Excluded in ("time", "date", "index", "unnamed")
        )
    ]
    if not lst_SensorNames:
        raise ValueError(
            "No numeric sensor columns found. Set str_RequestedSensorName "
            "to an exact column name."
        )

    LOGGER.info("Automatically selected first numeric column: %s", lst_SensorNames[0])
    return lst_SensorNames[0]


def estimate_baseline(
    ser_Temperatures: pd.Series, int_BaselineReadings: int
) -> tuple[float, float]:
    """Estimate mean and sample standard deviation from assumed normal readings."""
    if int_BaselineReadings < 2 or len(ser_Temperatures) <= int_BaselineReadings:
        raise ValueError("Not enough readings for both calibration and detection.")

    ser_Baseline: pd.Series = ser_Temperatures.iloc[:int_BaselineReadings]
    int_ValidReadings: int = int(ser_Baseline.notna().sum())
    if int_ValidReadings < 0.8 * int_BaselineReadings:
        raise ValueError("More than 20% of baseline readings are missing or invalid.")

    flt_BaselineMean: float = float(ser_Baseline.mean())
    flt_BaselineStd: float = float(ser_Baseline.std(ddof=1))
    if not math.isfinite(flt_BaselineStd) or flt_BaselineStd <= 0.0:
        raise ValueError("Baseline standard deviation must be positive and finite.")

    return flt_BaselineMean, flt_BaselineStd


def calculate_upper_cusum(
    ser_Temperatures: pd.Series,
    flt_BaselineMean: float,
    flt_BaselineStd: float,
    flt_ReferenceValue: float,
) -> pd.Series:
    """Accumulate standardized positive deviations above the baseline.

    A missing reading leaves the score undefined at that point and resets the
    accumulator, rather than interpreting missingness as a normal measurement.
    """
    if flt_BaselineStd <= 0 or not math.isfinite(flt_BaselineStd):
        raise ValueError("flt_BaselineStd must be positive and finite.")
    if flt_ReferenceValue < 0 or not math.isfinite(flt_ReferenceValue):
        raise ValueError("flt_ReferenceValue must be non-negative and finite.")

    lst_CusumScores: list[float] = []
    flt_CurrentScore: float = 0.0

    for flt_Temperature in ser_Temperatures:
        if pd.isna(flt_Temperature):
            flt_CurrentScore = 0.0
            lst_CusumScores.append(float("nan"))
            continue

        flt_StandardisedDeviation: float = (
            float(flt_Temperature) - flt_BaselineMean
        ) / flt_BaselineStd
        flt_CurrentScore = max(
            0.0, flt_CurrentScore + flt_StandardisedDeviation - flt_ReferenceValue
        )
        lst_CusumScores.append(flt_CurrentScore)

    return pd.Series(
        lst_CusumScores,
        index=ser_Temperatures.index,
        dtype="float64",
        name="upper_cusum_score",
    )


def find_alarm_start_rows(
    ser_CusumScores: pd.Series, flt_DecisionThreshold: float
) -> list[int]:
    """Return row indices where CUSUM first crosses the threshold."""
    if flt_DecisionThreshold <= 0 or not math.isfinite(flt_DecisionThreshold):
        raise ValueError("flt_DecisionThreshold must be positive and finite.")

    ser_AboveThreshold: pd.Series = ser_CusumScores.gt(flt_DecisionThreshold)
    ser_NewAlarms: pd.Series = ser_AboveThreshold & ~ser_AboveThreshold.shift(
        fill_value=False
    )
    return [int(int_RowIndex) for int_RowIndex in ser_CusumScores.index[ser_NewAlarms]]

def extract_alarm_intervals(
    ser_CusumScores: pd.Series,
    flt_DecisionThreshold: float
) -> list[tuple[int, int]]:
    """
    Identify continuous periods where the CUSUM score exceeds
    the decision threshold.

    Returns inclusive (start, end) positional indices.
    Missing readings are treated as inactive alarms.
    """
    if flt_DecisionThreshold <= 0 or not math.isfinite(flt_DecisionThreshold):
        raise ValueError("Decision threshold must be positive and finite.")

    lst_AlarmIntervals: list[tuple[int, int]] = []
    int_AlarmStart: Optional[int] = None

    for int_Position, flt_Score in enumerate(ser_CusumScores):

        bol_AlarmActive: bool = (
            pd.notna(flt_Score)
            and flt_Score > flt_DecisionThreshold
        )

        if bol_AlarmActive and int_AlarmStart is None:
            # A new alarm period begins
            int_AlarmStart = int_Position

        elif not bol_AlarmActive and int_AlarmStart is not None:
            # The current alarm period ends
            lst_AlarmIntervals.append(
                (int_AlarmStart, int_Position - 1)
            )
            int_AlarmStart = None

    # Handle an alarm still active at the final reading
    if int_AlarmStart is not None:
        lst_AlarmIntervals.append(
            (int_AlarmStart, len(ser_CusumScores) - 1)
        )

    return lst_AlarmIntervals


def get_plot_axis(df_SensorData: pd.DataFrame) -> pd.Index:
    """Use valid timestamps when available, otherwise use measurement row numbers."""
    for str_ColumnName in df_SensorData.columns:
        if not isinstance(str_ColumnName, str):
            continue
        str_NormalisedName: str = str_ColumnName.strip().lower().replace(" ", "_")
        if str_NormalisedName in {"timestamp", "datetime", "date_time", "time", "date"}:
            ser_Timestamps: pd.Series = pd.to_datetime(
                df_SensorData[str_ColumnName], errors="coerce"
            )
            if ser_Timestamps.notna().all():
                return pd.DatetimeIndex(ser_Timestamps)
            LOGGER.warning("Timestamp column has invalid entries; using row numbers.")
            break

    return df_SensorData.index


def plot_results(
    idx_PlotAxis: pd.Index,
    ser_Temperatures: pd.Series,
    ser_CusumScores: pd.Series,
    lst_AlarmRows: list[int],
    str_SensorName: str,
    flt_BaselineMean: float,
    flt_DecisionThreshold: float,
    int_BaselineReadings: int,
    pth_OutputPlot: Path,
) -> None:
    """Save temperature and CUSUM time-series plots."""
    fig_Plot: Figure = plt.figure(figsize=(13, 8))
    ax_Temperature: Axes = fig_Plot.add_subplot(2, 1, 1)
    ax_Cusum: Axes = fig_Plot.add_subplot(2, 1, 2, sharex=ax_Temperature)

    ax_Temperature.plot(idx_PlotAxis, ser_Temperatures, linewidth=0.8)
    ax_Temperature.axhline(
        flt_BaselineMean, color="green", linestyle="--", label="Baseline mean"
    )
    ax_Temperature.set_title(f"Sensor: {str_SensorName}")
    ax_Temperature.set_ylabel("Temperature (°C)")

    ax_Cusum.plot(
        idx_PlotAxis[int_BaselineReadings:], ser_CusumScores, linewidth=0.8
    )
    ax_Cusum.axhline(
        flt_DecisionThreshold,
        color="red",
        linestyle="--",
        label="CUSUM decision threshold",
    )
    ax_Cusum.set_ylabel("Upper CUSUM score")
    ax_Cusum.set_xlabel("Timestamp / measurement index")

    for int_AlarmNumber, int_AlarmRow in enumerate(lst_AlarmRows):
        ax_Temperature.axvline(
            idx_PlotAxis[int_AlarmRow],
            color="red",
            alpha=0.6,
            label="CUSUM alarm start" if int_AlarmNumber == 0 else None,
        )

    ax_Temperature.legend()
    ax_Cusum.legend()
    fig_Plot.tight_layout()
    fig_Plot.savefig(pth_OutputPlot, dpi=160)
    LOGGER.info("Saved plot to: %s", pth_OutputPlot)
    plt.show()
    plt.close(fig_Plot)


def main() -> None:
    """Run exploratory upper-sided CUSUM for one temperature sensor."""
    # False = concise results; True = detailed diagnostic tables and alarm times.
    bol_ShowDiagnostics: bool = False
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    # Workbook and sensor configuration
    pth_Dataset: Path = Path.home() / "Desktop" / "Synthetic_Data_Generation.xlsx"
    int_SheetIndex: int = 2  # Sensor_Data worksheet
    str_RequestedSensorName: Optional[str] = None  # Defaults to first numeric sensor

    # 288 readings x 5 minutes = 24 hours (assumes regular timestamps).
    int_BaselineReadings: int = 288
    flt_ReferenceValue: float = 0.5
    flt_DecisionThreshold: float = 5.0

    # Load data
    if bol_ShowDiagnostics:
        with pd.ExcelFile(pth_Dataset) as excel_file:
            lst_SheetNames: list[str] = excel_file.sheet_names
            print("Available worksheets:", lst_SheetNames)

    df_SensorData: pd.DataFrame = load_sensor_data(pth_Dataset, int_SheetIndex)
    str_SensorName: str = select_sensor(df_SensorData, str_RequestedSensorName)
    ser_Temperatures: pd.Series = pd.to_numeric(
        df_SensorData[str_SensorName], errors="coerce"
    ).replace([float("inf"), float("-inf")], float("nan"))
    int_MissingReadings: int = int(ser_Temperatures.isna().sum())

    # Estimate the fixed baseline and calculate CUSUM scores.
    flt_BaselineMean, flt_BaselineStd = estimate_baseline(
        ser_Temperatures, int_BaselineReadings
    )
    ser_TestTemperatures: pd.Series = ser_Temperatures.iloc[int_BaselineReadings:]
    ser_CusumScores: pd.Series = calculate_upper_cusum(
        ser_TestTemperatures,
        flt_BaselineMean,
        flt_BaselineStd,
        flt_ReferenceValue,
    )
    lst_AlarmRows: list[int] = find_alarm_start_rows(
        ser_CusumScores, flt_DecisionThreshold
    )
    lst_AlarmIntervals: list[tuple[int, int]] = extract_alarm_intervals(
        ser_CusumScores, flt_DecisionThreshold
    )

    # Load injected anomaly labels for reference (not validated detection scores).
    df_GroundTruth: pd.DataFrame = pd.read_excel(
        pth_Dataset, sheet_name="Ground_Truth_Spikes"
    )
    df_SensorEvents: pd.DataFrame = df_GroundTruth.loc[
        df_GroundTruth["sensor"] == str_SensorName
    ].copy()
    df_SensorEvents["start"] = pd.to_datetime(df_SensorEvents["start"])
    df_SensorEvents["end"] = pd.to_datetime(df_SensorEvents["end"])
    df_SensorEvents = df_SensorEvents.sort_values(by="start")

    # Concise terminal summary for every run.
    print("\n========== CUSUM ANALYSIS ==========")
    print(f"Sensor: {str_SensorName}")
    print(f"Readings: {len(ser_Temperatures)}")
    print(f"Missing readings: {int_MissingReadings}")
    print(f"\nBaseline mean: {flt_BaselineMean:.3f}°C")
    print(f"Baseline standard deviation: {flt_BaselineStd:.3f}°C")
    print(f"\nReference value (k): {flt_ReferenceValue}")
    print(f"Decision threshold (h): {flt_DecisionThreshold}")
    print(f"\nGround-truth events: {len(df_SensorEvents)}")
    print(f"Alarm intervals: {len(lst_AlarmIntervals)}")
    print(f"Maximum CUSUM score: {ser_CusumScores.max():.1f}")
    print("Note: alarm intervals are not validated event-level detections.")

    # Detailed diagnostics remain available without cluttering normal runs.
    if bol_ShowDiagnostics:
        print("\n--- DATASET INSPECTION ---")
        print("Dataset shape:", df_SensorData.shape)
        print("Column names:", df_SensorData.columns.tolist())
        print("Column data types:\n", df_SensorData.dtypes)
        print("First five rows:\n", df_SensorData.head().to_string(index=False))

        print("\n--- GROUND TRUTH ---")
        print("Columns:", df_GroundTruth.columns.tolist())
        print("First ten rows:\n", df_GroundTruth.head(10).to_string(index=False))
        print(f"\nGround-truth events for {str_SensorName}:")
        print(df_SensorEvents.to_string(index=False))
        print("\nGround-truth event types:\n", df_GroundTruth["type"].value_counts())

        ser_Timestamps: pd.Series = pd.to_datetime(df_SensorData["DateTime"])
        ser_DailyMeans: pd.Series = ser_Temperatures.groupby(
            ser_Timestamps.dt.normalize()
        ).mean()
        ser_DailyDeviations: pd.Series = ser_DailyMeans - flt_BaselineMean
        df_DailySummary: pd.DataFrame = pd.DataFrame({
            "Daily Mean (°C)": ser_DailyMeans,
            "Difference from Baseline (°C)": ser_DailyDeviations,
        })

        df_CusumDiagnostic: pd.DataFrame = pd.DataFrame({
            "DateTime": ser_Timestamps.iloc[int_BaselineReadings:].to_numpy(),
            "CUSUM": ser_CusumScores.to_numpy(),
        })
        df_CusumDaily: pd.DataFrame = (
            df_CusumDiagnostic.set_index("DateTime")["CUSUM"]
            .resample("D")
            .agg(["first", "max", "last"])
        )
        print("\n--- DAILY CUSUM SCORES ---")
        print(df_CusumDaily.round(1).to_string())
        print("\n--- DAILY TEMPERATURE SUMMARY ---")
        print(df_DailySummary.round(3).to_string())

        print("\n--- ALARM INTERVALS ---")
        for int_Start, int_End in lst_AlarmIntervals:
            int_OriginalStart: int = int_BaselineReadings + int_Start
            int_OriginalEnd: int = int_BaselineReadings + int_End
            tms_StartTime: pd.Timestamp = ser_Timestamps.iloc[int_OriginalStart]
            tms_EndTime: pd.Timestamp = ser_Timestamps.iloc[int_OriginalEnd]
            print(f"Alarm interval: {tms_StartTime} to {tms_EndTime}")

        print("\n--- NEW ALARM STARTS VS GROUND TRUTH ---")
        for int_AlarmRow in lst_AlarmRows:
            # The current workbook has an unchanged RangeIndex (0..8639).
            tms_AlarmTime: pd.Timestamp = ser_Timestamps.iloc[int_AlarmRow]
            ser_MatchingEvents: pd.Series = (
                (df_SensorEvents["start"] <= tms_AlarmTime)
                & (df_SensorEvents["end"] >= tms_AlarmTime)
            )
            lst_EventIds: list[str] = (
                df_SensorEvents.loc[ser_MatchingEvents, "event_id"]
                .astype(str)
                .tolist()
            )
            str_Result: str = (
                ", ".join(lst_EventIds) if lst_EventIds else "Outside recorded event"
            )
            print(f"{tms_AlarmTime}: {str_Result}")

    # Preserve the original graph.
    idx_PlotAxis: pd.Index = get_plot_axis(df_SensorData)
    pth_OutputPlot: Path = pth_Dataset.parent / "cusum_sensor_results.png"
    plot_results(
        idx_PlotAxis,
        ser_Temperatures,
        ser_CusumScores,
        lst_AlarmRows,
        str_SensorName,
        flt_BaselineMean,
        flt_DecisionThreshold,
        int_BaselineReadings,
        pth_OutputPlot,
    )
    print(f"\nPlot saved: {pth_OutputPlot}")


if __name__ == "__main__":
    main()
