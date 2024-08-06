import numpy as np
from scipy.stats import linregress


def log_interpolate_p(stat, xi, quantiles_table):
    """Log interpolate the p-value corresponding to an Anderson-
    Darling statistic, using a table of precomputed quantiles for
    distributions of different Anderson-Darling distributions.

    For details, see:
    AUTOMATED THRESHOLD SELECTION FOR EXTREME VALUE
    ANALYSIS VIA ORDERED GOODNESS-OF-FIT TESTS WITH
    ADJUSTMENT FOR FALSE DISCOVERY RATE

    TODO CITE

    And their source code:

    https://github.com/brianbader/eva_package/blob/master/R/gpdAd.R

    args:
        stat (float): value of the AD statistic
        xi (float): value of the xi parameter estimate used to calculate the AD statistic,
            rounded to 2 decimal places
        quantiles table (pd.DataFrame): table of quantiles

    returns:
        (float) p-value
    """
    # series of critical values corresponding to AD distribution
    xi_crit_vals = quantiles_table.loc[xi]
    # first place in series where critical value exceeds stat
    bound_location = min(np.where(stat < xi_crit_vals)[0])
    # lower bound (on p value)
    bound = xi_crit_vals.index[bound_location]
    # corresponding critical value upper value
    upper = xi_crit_vals.iloc[bound_location]
    # upper bound on p value
    upper_p_val_bound = xi_crit_vals.index[bound_location - 1]
    # corresponding critical value lower value
    lower = xi_crit_vals.iloc[bound_location - 1]

    # log interpolation
    dif = (upper - stat) / (upper - lower)
    val = (dif * (-1 * np.log(bound) + np.log(upper_p_val_bound))) + np.log(bound)
    p = np.exp(val)
    return p


def regress_impute_p(stat, xi, quantiles_table):
    """Predict the small p-value associated with a large AD test statistic
    using the regression method of Bader et all.  Code is translated from
    this R code:

    https://github.com/brianbader/eva_package/blob/master/R/gpdAd.R

    args:
        stat (float): value of the AD statistic
        xi (float): value of the xi parameter estimate used to calculate the AD statistic,
            rounded to 2 decimal places
        quantiles table (pd.DataFrame): table of quantiles

    returns:
        (float) p-value
    """

    stats_xi = quantiles_table.loc[xi]
    # 50 p values and critical vals in the tail
    tail_p_vals = stats_xi.iloc[-50:]
    x = tail_p_vals.index
    y = -1 * np.log(tail_p_vals.values)
    # regression for interpolation
    slope, intercept, _, _, _ = linregress(x, y=y)
    predicted = slope * stat + intercept
    p = np.exp(-1 * predicted)
    return p


def AD_approx_p_val(stat, xi, quantiles_table):
    """Approximate the p value associated with an AD statistic,
    given xi estimator.  The asymptotic distribution of the AD
    statistic depends on xi, hence the dependency.  If the stat
    falls within the range of the precomputed table, interpolate it
    using the table.  If not, impute it using the regression method
    of Bader et. all.

    args:
        stat (float): value of the AD statistic
        xi (float): value of the xi parameter estimate used to calculate the AD statistic,
            rounded to 2 decimal places
        quantiles table (pd.DataFrame): table of quantiles

    returns:
        (float) p-value
    """
    if stat > quantiles_table.loc[xi].iloc[-1]:
        p = regress_impute_p(stat, xi, quantiles_table)
    else:
        p = log_interpolate_p(stat, xi, quantiles_table)
    return p



def forward_stop_u_selection(p_vals, alpha):
    """ForwardStop implementation, equation (2) from the paper.

    args:
        p_vals (np.array[float]): p_values of ordered hypothesis tests
        alpha (float): false positive rate control

    returns:
        (int): location in p_vals array of chosen u threshold
    """
    k = np.arange(len(p_vals)) + 1
    scaled_qsums = -1 / k * np.cumsum(np.log(1 - p_vals))
    # first value where
    max_k = max(np.where(scaled_qsums < alpha)[0])
    return max_k
