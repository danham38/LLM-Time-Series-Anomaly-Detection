"""Unit tests for current cusum_sensor_demo.py script.

Run:
    python -m pytest -v test_cusum_sensor_demo.py
"""

import pandas as pd
import pytest

from cusum_sensor_detection_method import (
    calculate_upper_cusum,
    estimate_baseline,
    find_alarm_start_rows,
    select_sensor,
)


def test_cusum_matches_hand_calculated_scores() -> None:
    """Check the recurrence against values worked out independently."""
    ser_Temperatures: pd.Series = pd.Series(
        [26.0, 26.25, 26.375, 26.0, 25.5], dtype="float64"
    )
    lst_ExpectedScores: list[float] = [0.0, 0.5, 1.5, 1.0, 0.0]

    ser_ActualScores: pd.Series = calculate_upper_cusum(
        ser_Temperatures=ser_Temperatures,
        flt_BaselineMean=26.0,
        flt_BaselineStd=0.25,
        flt_ReferenceValue=0.5,
    )

    assert ser_ActualScores.tolist() == pytest.approx(lst_ExpectedScores)
    assert ser_ActualScores.index.equals(ser_Temperatures.index)


def test_missing_reading_resets_cusum() -> None:
    """Confirm the existing policy: NaN output and a zeroed accumulator."""
    ser_Temperatures: pd.Series = pd.Series(
        [26.5, float("nan"), 26.25], dtype="float64"
    )

    ser_Scores: pd.Series = calculate_upper_cusum(
        ser_Temperatures=ser_Temperatures,
        flt_BaselineMean=26.0,
        flt_BaselineStd=0.25,
        flt_ReferenceValue=0.5,
    )

    assert ser_Scores.iloc[0] == pytest.approx(1.5)
    assert pd.isna(ser_Scores.iloc[1])
    assert ser_Scores.iloc[2] == pytest.approx(0.5)


def test_baseline_uses_only_first_requested_readings() -> None:
    """Check the baseline mean and sample standard deviation (ddof=1)."""
    ser_Temperatures: pd.Series = pd.Series(
        [26.0, 26.5, 27.0, 26.5, 35.0], dtype="float64"
    )

    flt_Mean: float
    flt_Std: float
    flt_Mean, flt_Std = estimate_baseline(
        ser_Temperatures=ser_Temperatures,
        int_BaselineReadings=4,
    )

    assert flt_Mean == pytest.approx(26.5)
    assert flt_Std == pytest.approx((0.5 / 3.0) ** 0.5)
    # The later anomalous reading (35.0) must not influence calibration.


def test_select_sensor_ignores_datetime_column() -> None:
    """Select the first numeric sensor without mistaking timestamps for sensors."""
    df_SensorData: pd.DataFrame = pd.DataFrame(
        {
            "DateTime": pd.date_range("2026-06-01", periods=3, freq="5min"),
            "Sensor 1": [26.0, 26.1, 26.2],
            "Sensor 2": [23.0, 23.1, 23.2],
        }
    )

    str_SelectedSensor: str = select_sensor(df_SensorData, None)

    assert str_SelectedSensor == "Sensor 1"


def test_alarm_starts_only_after_strict_threshold_exceedance() -> None:
    """Follow the proposal's strict alarm condition C_i^+ > H.

    Expected to FAIL with the current script because it uses .ge(H), meaning
    >= H. Change it to .gt(H) if the desired specification is strictly > H.
    """
    ser_Scores: pd.Series = pd.Series(
        [0.0, 4.9, 5.0, 5.5, 5.7, 4.0, 5.2], dtype="float64"
    )

    lst_AlarmStartRows: list[int] = find_alarm_start_rows(
        ser_CusumScores=ser_Scores,
        flt_DecisionThreshold=5.0,
    )

    assert lst_AlarmStartRows == [3, 6]
