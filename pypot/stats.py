import numpy as np

def GPD_param_cov_matrix(theta_hat, n):
    """Compute the asymptotically approximated covariance matrix
    for efficient estimators (e.g. MLE or MPS).

    Args:
        theta_hat (iterable): point estimates
        n (int): sample size

    Returns:
        np.ndarray: covariance matrix
    """
    # unpack MLE point estimates
    xi_hat = theta_hat[0]
    sigma_hat = theta_hat[1]
    # unnormalized estimated asymptotic variances
    var_xi_hat = (1 - xi_hat**2)
    var_sigma_hat = 2 * sigma_hat **2 * (1 + xi_hat)
    # unnormalized estimated asymptotic covariance
    cov_xi_sigma_hat = -1 * sigma_hat * (1 + xi_hat)
    # unnormalized cov matrix
    cov_matrix = np.array([
        [var_xi_hat, cov_xi_sigma_hat],
        [cov_xi_sigma_hat, var_sigma_hat]
    ])
    # normalize
    return n**(-1) * cov_matrix
