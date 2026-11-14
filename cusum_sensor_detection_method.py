"""Exploratory upper-sided CUSUM detection for one temperature sensor.

Assumption: the first BASELINE_READINGS measurements represent normal operation.
This is a prototype, not a validated early-warning system.
"""

from pathlib import Path
import logging
import math

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import pandas as pd

from typing import Optional
str_SensorName: Optional[str] = None


LOGGER = logging.getLogger(__name__)


def load_sensor_data(pth_Dataset: Path, int_SheetIndex: int) -> pd.DataFrame:
    """Load a single Excel worksheet and validate it is not empty."""
    if not pth_Dataset.is_file():
        raise FileNotFoundError(f"Dataset not found: {pth_Dataset}")

    df_SensorData: pd.DataFrame = pd.read_excel(

        pth_Dataset, sheet_name=int_SheetIndex
    )
    print("Dataset shape:", df_SensorData.shape)

    print("\nColumn names:")
    print(df_SensorData.columns.tolist())

    print("\nColumn data types:")
    print(df_SensorData.dtypes)

    print("\nFirst five rows:")
    print(df_SensorData.head())
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

    ser_AboveThreshold: pd.Series = ser_CusumScores.ge(flt_DecisionThreshold)
    ser_NewAlarms: pd.Series = ser_AboveThreshold & ~ser_AboveThreshold.shift(
        fill_value=False
    )
    return [int(int_RowIndex) for int_RowIndex in ser_CusumScores.index[ser_NewAlarms]]


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
    """Run exploratory CUSUM detection against one Excel temperature sensor."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Change the filename to the actual workbook on your Desktop.
    pth_Dataset: Path = Path.home() / "Desktop" / "Synthetic_Data_Generation.xlsx"
    int_SheetIndex: int = 2
    str_RequestedSensorName: Optional[str] = None  # Or use an exact name, e.g. "Sensor_001"

    # 288 readings x 5 minutes = 24 hours (assuming no missing timestamps).
    int_BaselineReadings: int = 288
    flt_ReferenceValue: float = 0.5
    flt_DecisionThreshold: float = 5.0
    #get correct sheet from spreadsheet
    with pd.ExcelFile(pth_Dataset) as excel_file:
        lst_SheetNames: list[str] = excel_file.sheet_names
        print("Available worksheets:", lst_SheetNames)

    df_SensorData: pd.DataFrame = load_sensor_data(pth_Dataset, int_SheetIndex)
    str_SensorName: str = select_sensor(df_SensorData, str_RequestedSensorName)
    ser_Temperatures: pd.Series = pd.to_numeric(
        df_SensorData[str_SensorName], errors="coerce"
    ).replace([float("inf"), float("-inf")], float("nan"))

    int_MissingReadings: int = int(ser_Temperatures.isna().sum())
    LOGGER.info("Sensor: %s | Rows: %d | Missing: %d",
                str_SensorName, len(ser_Temperatures), int_MissingReadings)

    flt_BaselineMean, flt_BaselineStd = estimate_baseline(
        ser_Temperatures, int_BaselineReadings
    )
    LOGGER.info("Baseline: mean=%.3f°C, std=%.3f°C",
                flt_BaselineMean, flt_BaselineStd)

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
    LOGGER.info("CUSUM alarm episodes: %d", len(lst_AlarmRows))
    if lst_AlarmRows:
        LOGGER.info("First alarm at worksheet data row index: %d", lst_AlarmRows[0])

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


if __name__ == "__main__":
    main()
