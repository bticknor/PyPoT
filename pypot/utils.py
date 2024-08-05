import numpy as np
import pandas as pd
import pkgutil
from io import StringIO


def fetch_adquantiles_table():
    adq_table = pkgutil.get_data("pypot", "data/ADQuantiles.csv")
    adq_frame = pd.read_csv(StringIO(adq_table.decode('utf-8')), index_col=0)
    # need colnames to be floats
    adq_frame.columns = adq_frame.columns.astype(float)
    return adq_frame


def years_span_series(extremes_series):
    """Computes time range of series, using index."""
    time_span_series = max(extremes_series.index) - min(extremes_series.index)

    AVE_DAYS_YEAR = 365.25
    # time span of the series in years
    years_span = time_span_series.days / AVE_DAYS_YEAR
    return years_span


def _generate_clusters(exceedances, r):
    """Generates clusters of time series exceedances that are separated
    by more than a time delta of r.

    args:
        exceedences (pd.Series): series of exceedances with a pd.datetime index
        r (pd.Timedelta or str): time difference that defines independent exceedance events

    returns:
        (Generator): iterable of cluster series
    """
    # check if string is provided e.g. "24h", and convert to timedelta if so
    if not isinstance(r, pd.Timedelta):
        try:
            r = pd.to_timedelta(r)
        except Exception as error:
            raise ValueError(f"invalid value in {r} for the 'r' argument") from error

    # There can be only one cluster if there is only one exceedance
    if len(exceedances) == 1:
        yield exceedances
        return

    # Locate clusters separated by gaps not smaller than `r`
    gap_indices = np.argwhere(
        (exceedances.index[1:] - exceedances.index[:-1]) > r
    ).flatten()
    if len(gap_indices) == 0:
        # All exceedances fall within the same cluster
        yield exceedances
    else:
        for i, gap_index in enumerate(gap_indices):
            if i == 0:
                # First cluster contains all values left from the gap
                yield exceedances.iloc[: gap_index + 1]
            else:
                # Other clusters contain values between previous and current gaps
                yield exceedances.iloc[gap_indices[i - 1] + 1 : gap_index + 1]

        # Last cluster contains all values right from the last gap
        yield exceedances.iloc[gap_indices[-1] + 1 :]


def get_extremes_peaks_over_threshold(ts, threshold, r="24h"):
    """
    Get extreme events from time series using the Peaks Over Threshold method.

    Parameters
    ----------
    ts : pandas.Series
        Time series of the signal.
    threshold : float
        Threshold used to find exceedances.
    r : pandas.Timedelta or value convertible to timedelta, optional
        Duration of window used to decluster the exceedances.
        By default r='24H' (24 hours).
        See pandas.to_timedelta for more information.

    Returns
    -------
    extremes : pandas.Series
        Time series of extreme events.


    Original code can be found at:

    https://github.com/georgebv/pyextremes/blob/master/src/pyextremes/extremes/peaks_over_threshold.py
    """

    # Get exceedances
    exceedances = ts.loc[ts.values > threshold]

    assert len(exceedances) > 0, "threshold yields zero exceedences"

    # Locate clusters separated by gaps not smaller than `r`
    # and select max within each cluster
    extreme_indices, extreme_values = [], []
    for cluster in _generate_clusters(exceedances=exceedances, r=r):
        extreme_indices.append(cluster.idxmax())
        extreme_values.append(cluster.loc[extreme_indices[-1]])

    return pd.Series(
        data=extreme_values,
        index=pd.Index(data=extreme_indices, name=ts.index.name or "date-time"),
        dtype=np.float64,
        name=ts.name or "extreme values",
    )

