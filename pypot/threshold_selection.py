import numpy as np
from scipy.stats import linregress
from pypot.generalized_pareto import anderson_darling_statistic, fit_GPD, gp_neg_loglik, gp_neg_loglik_jacob
from pypot.utils import fetch_adquantiles_table, get_extremes_peaks_over_threshold


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
    # for keying into table
    xi = round(xi, 2)
    if stat > quantiles_table.loc[xi].iloc[-1]:
        p = regress_impute_p(stat, xi, quantiles_table)
    else:
        p = log_interpolate_p(stat, xi, quantiles_table)
    return p


def forward_stop_adjusted_p(p_vals):
    """ForwardStop adjusted p-values, equation (2) from the paper.

    args:
        p_vals (np.array[float]): p_values of ordered hypothesis tests

    returns:
        (int): location in p_vals array of chosen u threshold
    """
    k = np.arange(len(p_vals)) + 1
    scaled_qsums = -1 / k * np.cumsum(np.log(1 - p_vals))
    return scaled_qsums


def run_AD_tests(series, thresh_down, thresh_up, l, r):
    """Run an ordered set of Anderson-Darling hypothesis tests.

    args:
        series (np.array): raw time series
        thresh_up (float): largest threshold to try
        thresh_down (float): smallest threshold to try
        l (int): number of thresholds in the grid between
            thresh_down and thresh_up
        r (str): time delta string to define independence
        alpha (float): false discovery rate control (i.e. 0.05 is 5%)

    returns:
        (tuple[np.array]): threshold, p_value, xi_hat, and sigma_hat for each test
    """
    assert thresh_down < thresh_up, "lower threshold bound must be below upper threshold bound"

    # threshold grid
    thresholds = np.linspace(thresh_down, thresh_up, l)

    p_vals = np.zeros(len(thresholds))
    xi_hats = np.zeros(len(thresholds))
    sigma_hats = np.zeros(len(thresholds))

    # quantiles table for p value approximation
    adq_frame = fetch_adquantiles_table()

    # initial guess for optimizer
    # TODO parameterize
    THETA_0 = (1/5, 1)

    # loop through thresholds to test
    for i, cand_threshold in enumerate(thresholds):
        # get extremes corresponding to threshold
        extremes = get_extremes_peaks_over_threshold(
            series,
            cand_threshold,
            r
        )
        # subtract away threshold for peaks
        x_cand = extremes - cand_threshold

        # fit GPD to peaks
        mle_cand = fit_GPD(
            x_cand,
            THETA_0,
            gp_neg_loglik,
            gp_neg_loglik_jacob
        )
        xi_hat_cand = mle_cand[0]
        xi_hats[i] = xi_hat_cand

        sigma_hat_cand = mle_cand[1]
        sigma_hats[i] = sigma_hat_cand

        # compute AD statistic
        ad_stat_cand = anderson_darling_statistic(x_cand, xi_hat_cand, sigma_hat_cand)

        # p-value of AD test
        p_cand = AD_approx_p_val(ad_stat_cand, round(xi_hat_cand, 2), adq_frame)

        p_vals[i] = p_cand

    # return p values and MLEs
    return thresholds, p_vals, xi_hats, sigma_hats


def forward_stop_u_selection(series, thresh_down, thresh_up, l, r, alpha=0.05):
    """Automatically select threshold for PoT analysis
    using forwardStop algorithm.

    args:
        series (np.array): raw time series
        thresh_up (float): largest threshold to try
        thresh_down (float): smallest threshold to try
        l (int): number of thresholds in the grid between
            thresh_down and thresh_up
        r (str): time delta string to define independence
        alpha (float): false discovery rate control (i.e. 0.05 is 5%)

    returns:
        (tuple[float, np.array[float, float]]): (threshold, [xi_hat, sigma_hat])
    """
    # run sequence of AD tests
    thresholds, p_vals, xi_hats, sigma_hats = run_AD_tests(series, thresh_down, thresh_up, l, r)
    # forward stop algorithm
    adjusted_p_vals = forward_stop_adjusted_p(p_vals)

    # first value where
    if alpha < min(adjusted_p_vals):
        raise RuntimeError("cannot control FDR at level {0}, try reducing alpha".format(alpha))

    threshold_selection_index = max(np.where(adjusted_p_vals < alpha)[0])
    chosen_threshold = thresholds[threshold_selection_index]
    chosen_xi_hat = xi_hats[threshold_selection_index]
    chosen_sigma_hat = sigma_hats[threshold_selection_index]
    return chosen_threshold, np.array([chosen_xi_hat, chosen_sigma_hat])
