import numpy as np
from pypot.generalized_pareto import gp_cdf


def anderson_darling_statistic(x, xi, sigma):
    """Anderson Darling statistic for GPD.

    args:
        x (np.array): data
        xi (float): xi parameter
        sigma (float): sigma parameter

    returns:
        (float) value of the statistic
    """
    n = len(x)
    # sort the sample from low to high
    x = np.sort(x)
    # compute cdf values at sample points
    cdf_vals_samp = gp_cdf(x, xi, sigma)
    # reverse the cdf vals
    # i.e. (z(1), z(2), ... z(n)) -> (z(n), z(n-1), ... z(1))
    cdf_vals_desc = cdf_vals_samp[::-1]
    # array of 2i - 1 for i=1, ..., n
    coeff = 2 * np.arange(1, n + 1) - 1
    # sum term
    s = np.sum(coeff * (np.log(cdf_vals_samp) + np.log(1 - cdf_vals_desc)))
    return -1 * n - s/n


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
