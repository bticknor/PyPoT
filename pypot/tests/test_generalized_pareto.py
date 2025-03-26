import pytest
import numpy as np
import pandas as pd
from pypot.generalized_pareto import (
    gp_cdf,
    gp_inv_cdf,
    gp_density,
    gp_morans_statistic,
    gp_param_cov_matrix,
    gp_nll,
    gp_neg_loglik_jacob,
    fit_GPD,
    anderson_darling_statistic,
    beta_obs_fisher_info
)

# sample GP data from xi=0.15, sigma=1
GP_SAMPLE_DATA = np.array([
    5.59084048, 1.09075642, 0.63188548, 0.07402671, 0.05463866,
    0.02420525, 0.04950818, 0.75822677, 0.0988577, 1.51150359,
    2.3063707,  0.75220847, 1.39267889, 0.50577769, 0.06134641,
    0.11424503, 0.77943162, 0.61614704, 0.30465203, 1.04671569
])

# covariate values
covs = np.array([
    0.98631076,  1.83246939,  0.63043955,  0.46071496, -0.08755576,
    -0.53340898, -0.68804057,  0.18736298,  0.67797429,  0.61334551,
    -0.47799352,  0.77138241,  1.28980673,  0.1141364 , -1.50600225,
    -0.33098507, -0.2324566 ,  1.54141299,  2.65899127,  0.94126726
]).reshape(20, 1)

ones = np.ones((20, 1))
# design matrix
X = np.hstack((ones, covs))

# sample nonstationary GP data from xi=0.15, beta_0 = 1, beta_1 =0.5
Y = np.array([
    24.88549266,  7.41204306,  2.35413337,  0.2533529 ,  0.14216151,
    0.05039365,  0.09540373,  2.26349192,  0.37715964,  5.58328696,
    4.93660775,  3.00701961,  7.21478593,  1.4555883 ,  0.0785343 ,
    0.26318388,  1.88623236,  3.61986545,  3.12961686,  4.55529811
])

FIT_DATASET = pd.DataFrame(
    np.column_stack((Y, covs)), columns=["y", "x"]
)


def test_gp_nll():
    result = gp_nll(
        np.array([0.15, 1, 1/2]),
        Y,
        X
        )
    expected = 467.89249
    assert result == pytest.approx(expected, abs=1e-5)


def test_gp_cdf():
    result = gp_cdf(
        np.array([2]),
        0.15,
        1
    )
    assert result == pytest.approx(0.82607, abs=1e-5)


def test_gp_inv_cdf():
    result = gp_inv_cdf(
        0.5,
        0.15,
        1
    )
    assert result == pytest.approx(0.73046, abs=1e-5)


def test_gp_density():
    result = gp_density(
        1,
        0.15,
        1
    )
    assert result == pytest.approx(0.34249, abs=1e-5)


def test_gp_morans_statistic():
    result = gp_morans_statistic(
        GP_SAMPLE_DATA,
        0.15,
        1
    )
    assert result == pytest.approx(76.51697, abs=1e-5)


def test_gp_param_cov_matrix():
    result = gp_param_cov_matrix(0.15, 1, 10)
    expected = np.array([[0.09775, -0.115], [-0.115, 0.23]])
    assert np.allclose(result, expected, atol=1e-5)


def test_gp_neg_loglik_jacob():
    result = gp_neg_loglik_jacob(
        np.array([0.15, 1, 1/2]),
        Y,
        X
    )
    expected = np.array([ 0.99723,  4.57968, -2.02859])
    assert np.allclose(result, expected, atol=1e-5)


def test_fit_gpd():
    result_slsqp = fit_GPD(
        FIT_DATASET,
        "y",
        ["x"]
    )
    result_diff_evo = fit_GPD(
        FIT_DATASET,
        "y",
        ["x"],
        method="diff_evo"
    )
    # error tolerance higher due to nondeterminism in optimizer
    assert np.allclose(result_slsqp, result_diff_evo, atol=1e-2)
    expected = np.array([ 0.44127043, -0.01992052,  1.20555396])
    assert np.allclose(result_slsqp, expected, atol=1e-2)


def test_anderson_darling_statistic():
    result = anderson_darling_statistic(GP_SAMPLE_DATA, 0.15, 1)
    expected = 1.75737
    assert result == pytest.approx(expected, abs=1e-5)


def test_beta_obs_fisher_info():

    result = beta_obs_fisher_info(
        Y, X, np.array([0.15, 1, 1/2])
    )
    expected = np.array([12.12098, 13.02688])
    assert np.allclose(result, expected, atol=1e-5)
