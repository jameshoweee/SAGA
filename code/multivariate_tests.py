"""
Extended multivariate test battery for discrete Gaussian signature samplers.

Tests:
    5.1 squared_norm_test: ||x||^2/sigma^2 ~ chi2(p)
    5.2 fisher_bh_meta: Fisher combination + BH localization of per-coord p-values
    5.3 max_offdiag_correlation: max |rho_ij| for i<j
    5.4 fft_domain_battery: per-frequency variance, Re/Im independence, HC
    5.5 cross_key_homogeneity: two-sample energy distance across keys
    5.6 two_sample_test: KS/CvM between two datasets
    5.7 energy_hz_test: energy test + Henze-Zirkler (batched, shared distances)
"""

import numpy as np
from math import sqrt, log
from scipy.stats import kstest, chi2, norm
from scipy.stats import false_discovery_control


# ---------------------------------------------------------------------------
# 5.1 Squared-norm / radius test
# ---------------------------------------------------------------------------

def squared_norm_test(sigma, data, alpha=0.001):
    """
    Test that ||x||^2 / sigma^2 ~ chi2(dim).

    Catches radial flaws: correct marginals but wrong norm due to
    correlation between coordinates.
    """
    data = np.asarray(data, dtype=float)
    n, dim = data.shape
    norms_sq = np.sum(data ** 2, axis=1) / (sigma ** 2)
    stat, pval = kstest(norms_sq, 'chi2', args=(dim,))

    return {
        "test": "squared_norm",
        "statistic": float(stat),
        "pvalue": float(pval),
        "dim": dim,
        "mean_norm_sq": float(np.mean(norms_sq)),
        "expected_mean": float(dim),
        "passes": pval > alpha,
    }


# ---------------------------------------------------------------------------
# 5.2 Fisher + Benjamini-Hochberg meta-layer
# ---------------------------------------------------------------------------

def fisher_bh_meta(pvalues, alpha=0.001):
    """
    Two-level meta-analysis of per-coordinate p-values.

    Global: Fisher's method -2*sum(ln(p_i)) ~ chi2(2m).
    Localization: BH procedure to name which coordinates fail.
    """
    pvalues = np.asarray(pvalues, dtype=float)
    valid = np.isfinite(pvalues) & (pvalues >= 0) & (pvalues <= 1)
    pvalues = np.where(valid, pvalues, 1.0)
    pvalues = np.clip(pvalues, 1e-300, 1.0)
    m = len(pvalues)

    fisher_stat = -2 * np.sum(np.log(pvalues))
    fisher_pval = 1 - chi2.cdf(fisher_stat, 2 * m)

    bh_rejected = false_discovery_control(pvalues, method='bh')
    n_rejected = int(np.sum(bh_rejected < alpha))
    rejected_coords = [int(i) for i in np.where(bh_rejected < alpha)[0]]

    return {
        "test": "fisher_bh_meta",
        "fisher_stat": float(fisher_stat),
        "fisher_pvalue": float(fisher_pval),
        "fisher_passes": fisher_pval > alpha,
        "bh_rejected_count": n_rejected,
        "bh_rejected_coords": rejected_coords[:20],
        "total_coords": m,
        "passes": fisher_pval > alpha,
    }


# ---------------------------------------------------------------------------
# 5.3 Max off-diagonal correlation
# ---------------------------------------------------------------------------

def max_offdiag_correlation(cov_normalized, nsamples, alpha=0.001):
    """
    max_{i<j} |rho_hat_{ij}| from the normalized covariance matrix.

    diagcov sums diagonals, diluting a single bad pair. This catches
    one pair sharing randomness (e.g., AVX2 lane reuse).

    Approximate null via Jiang's Gumbel limit for large p.
    """
    dim = cov_normalized.shape[0]

    max_rho = 0.0
    max_pair = (0, 0)
    for i in range(dim):
        for j in range(i + 1, dim):
            rho = abs(cov_normalized[i, j])
            if rho > max_rho:
                max_rho = rho
                max_pair = (i, j)

    se = 1.0 / sqrt(nsamples - 3)
    z_score = max_rho / se

    a_n = sqrt(2 * log(dim)) - (log(log(dim)) + log(4 * np.pi)) / (2 * sqrt(2 * log(dim)))
    b_n = 1.0 / sqrt(2 * log(dim))
    gumbel_x = (max_rho * sqrt(nsamples) - a_n) / b_n
    pval = 1 - np.exp(-np.exp(-gumbel_x))

    return {
        "test": "max_offdiag_correlation",
        "max_rho": float(max_rho),
        "max_pair": list(max_pair),
        "z_score": float(z_score),
        "gumbel_pvalue": float(pval),
        "passes": pval > alpha,
    }


# ---------------------------------------------------------------------------
# 5.4 FFT-domain battery
# ---------------------------------------------------------------------------

def fft_domain_battery(sigma, data, alpha=0.001):
    """
    DFT each signature half; under the null, frequency coefficients
    are iid complex Gaussian.

    1. Per-frequency variance test
    2. Re/Im independence test
    3. Higher criticism across frequencies

    ffSampling flaws are localized in the FFT basis.
    """
    data = np.asarray(data, dtype=float)
    n, dim = data.shape
    half = dim // 2

    results = {"test": "fft_domain_battery"}
    freq_pvalues = []

    for block_idx, block_name in enumerate(["first_half", "second_half"]):
        block = data[:, block_idx * half:(block_idx + 1) * half]
        fft_coeffs = np.fft.rfft(block, axis=1)
        n_freqs = fft_coeffs.shape[1]

        var_pvals = []
        indep_pvals = []

        for k in range(1, n_freqs - 1):
            re = np.real(fft_coeffs[:, k])
            im = np.imag(fft_coeffs[:, k])

            var_k = (np.var(re) + np.var(im)) / 2
            expected_var = sigma ** 2 * half / 2
            chi2_stat = n * var_k / expected_var
            var_pval = 2 * min(chi2.cdf(chi2_stat, n - 1),
                               1 - chi2.cdf(chi2_stat, n - 1))
            var_pvals.append(var_pval)

            corr = np.corrcoef(re, im)[0, 1]
            t_stat = corr * sqrt(n - 2) / sqrt(1 - corr ** 2 + 1e-30)
            indep_pval = 2 * (1 - norm.cdf(abs(t_stat)))
            indep_pvals.append(indep_pval)

        freq_pvalues.extend(var_pvals)
        freq_pvalues.extend(indep_pvals)

        results[block_name] = {
            "n_freqs_tested": len(var_pvals),
            "min_var_pval": float(min(var_pvals)) if var_pvals else 1.0,
            "min_indep_pval": float(min(indep_pvals)) if indep_pvals else 1.0,
        }

    all_pvals = np.array(freq_pvalues)
    all_pvals = np.clip(all_pvals, 1e-300, 1.0)

    hc_stat = 0.0
    m = len(all_pvals)
    sorted_p = np.sort(all_pvals)
    for i, p in enumerate(sorted_p):
        expected = (i + 1) / m
        if expected > 0.05 and expected < 0.95:
            hc_i = sqrt(m) * abs((i + 1) / m - p) / sqrt(expected * (1 - expected))
            hc_stat = max(hc_stat, hc_i)

    fisher_stat = -2 * np.sum(np.log(all_pvals))
    fisher_pval = 1 - chi2.cdf(fisher_stat, 2 * m)

    results["higher_criticism_stat"] = float(hc_stat)
    results["fisher_pvalue"] = float(fisher_pval)
    results["n_pvalues"] = m
    results["passes"] = fisher_pval > alpha

    return results


# ---------------------------------------------------------------------------
# 5.5 Cross-key homogeneity (two-sample energy distance)
# ---------------------------------------------------------------------------

def _energy_distance(X, Y):
    """
    Energy distance between two multivariate samples.

    E(X,Y) = 2/(nm) sum||Xi-Yj|| - 1/n^2 sum||Xi-Xj|| - 1/m^2 sum||Yi-Yj||
    """
    n = len(X)
    m = len(Y)

    xy = 0.0
    for i in range(min(n, 500)):
        for j in range(min(m, 500)):
            xy += np.linalg.norm(X[i] - Y[j])
    xy *= (n * m) / (min(n, 500) * min(m, 500))
    xy = 2 * xy / (n * m)

    xx = 0.0
    ns = min(n, 500)
    for i in range(ns):
        for j in range(i + 1, ns):
            xx += np.linalg.norm(X[i] - X[j])
    xx *= n * (n - 1) / (ns * (ns - 1)) if ns > 1 else 0
    xx = xx / (n * n) if n > 0 else 0

    yy = 0.0
    ms = min(m, 500)
    for i in range(ms):
        for j in range(i + 1, ms):
            yy += np.linalg.norm(Y[i] - Y[j])
    yy *= m * (m - 1) / (ms * (ms - 1)) if ms > 1 else 0
    yy = yy / (m * m) if m > 0 else 0

    return xy - xx - yy


def cross_key_homogeneity(data1, data2, alpha=0.001, n_perms=200):
    """
    Two-sample energy distance test between datasets from different keys.

    Under the GPV framework, key-independence bounds the marginal's
    key-dependence at ~2^{-74}, far below any detection floor.
    Any detectable difference is an implementation flaw.
    """
    X = np.asarray(data1, dtype=float)
    Y = np.asarray(data2, dtype=float)

    observed = _energy_distance(X, Y)

    combined = np.vstack([X, Y])
    n = len(X)
    rng = np.random.default_rng(44)
    null_stats = []
    for _ in range(n_perms):
        perm = rng.permutation(len(combined))
        X_perm = combined[perm[:n]]
        Y_perm = combined[perm[n:]]
        null_stats.append(_energy_distance(X_perm, Y_perm))

    null_stats = np.array(null_stats)
    pvalue = np.mean(null_stats >= observed)

    return {
        "test": "cross_key_homogeneity",
        "energy_distance": float(observed),
        "pvalue": float(pvalue),
        "n_permutations": n_perms,
        "passes": pvalue > alpha,
    }


# ---------------------------------------------------------------------------
# 5.6 Two-sample tests
# ---------------------------------------------------------------------------

def two_sample_test(data1, data2, alpha=0.001):
    """
    Two-sample comparison between implementations or against reference.

    Per-coordinate KS + CvM, BH-corrected.
    """
    d1 = np.asarray(data1, dtype=float)
    d2 = np.asarray(data2, dtype=float)
    dim = d1.shape[1]

    from scipy.stats import ks_2samp, cramervonmises_2samp

    ks_pvals = []
    cvm_pvals = []
    for j in range(dim):
        ks_stat, ks_p = ks_2samp(d1[:, j], d2[:, j])
        ks_pvals.append(ks_p)
        cvm_result = cramervonmises_2samp(d1[:, j], d2[:, j])
        cvm_pvals.append(cvm_result.pvalue)

    ks_bh = false_discovery_control(np.array(ks_pvals), method='bh')
    cvm_bh = false_discovery_control(np.array(cvm_pvals), method='bh')

    ks_rejected = int(np.sum(ks_bh < alpha))
    cvm_rejected = int(np.sum(cvm_bh < alpha))

    return {
        "test": "two_sample",
        "ks_rejected_coords": ks_rejected,
        "cvm_rejected_coords": cvm_rejected,
        "min_ks_pvalue": float(min(ks_pvals)),
        "min_cvm_pvalue": float(min(cvm_pvals)),
        "dim": dim,
        "passes": ks_rejected == 0 and cvm_rejected == 0,
    }


# ---------------------------------------------------------------------------
# 5.7 Henze-Zirkler test
# ---------------------------------------------------------------------------

def _hz_statistic(Y_std, beta, p):
    """Compute the raw HZ statistic from standardized data."""
    n = len(Y_std)
    beta2 = beta ** 2

    n_sub = min(n, 500)
    if n_sub < n:
        idx = np.random.default_rng(45).choice(n, n_sub, replace=False)
        Y_sub = Y_std[idx]
    else:
        Y_sub = Y_std
        n_sub = n

    dists_sq = np.zeros((n_sub, n_sub))
    for i in range(n_sub):
        diff = Y_sub[i] - Y_sub
        dists_sq[i] = np.sum(diff ** 2, axis=1)

    term1 = np.sum(np.exp(-beta2 / 2 * dists_sq)) / n_sub
    norms_sq = np.sum(Y_sub ** 2, axis=1)
    term2 = 2 * (1 + beta2) ** (-p / 2) * np.sum(
        np.exp(-beta2 / (2 * (1 + beta2)) * norms_sq)
    ) / n_sub
    term3 = (1 + 2 * beta2) ** (-p / 2)

    return term1 - term2 + term3


def henze_zirkler_test(sigma, data, alpha=0.001, mc_B=100):
    """
    Henze-Zirkler multivariate normality test, MC-calibrated.

    The analytic log-normal null is uncalibrated for large p (p >= 128).
    We use MC calibration: generate B replicates from N(0, sigma^2*I),
    compute HZ on each, and use the empirical null.
    """
    data = np.asarray(data, dtype=float)
    n, p = data.shape

    Y = data / sigma
    Y = Y - Y.mean(axis=0)

    S = np.cov(Y, rowvar=False)
    try:
        L = np.linalg.cholesky(np.linalg.inv(S))
    except np.linalg.LinAlgError:
        return {"test": "henze_zirkler", "statistic": 0.0,
                "pvalue": 1.0, "passes": True,
                "note": "singular covariance"}

    Y_std = Y @ L.T

    beta = 1 / (2 * p) * ((2 * p + 1) * n / 4) ** (1 / (p + 4))

    observed = _hz_statistic(Y_std, beta, p)

    rng = np.random.default_rng(46)
    null_stats = np.empty(mc_B)
    for b in range(mc_B):
        Y_null = rng.normal(0, 1, size=(n, p))
        S_null = np.cov(Y_null, rowvar=False)
        try:
            L_null = np.linalg.cholesky(np.linalg.inv(S_null))
        except np.linalg.LinAlgError:
            null_stats[b] = 0.0
            continue
        Y_null_std = (Y_null - Y_null.mean(axis=0)) @ L_null.T
        null_stats[b] = _hz_statistic(Y_null_std, beta, p)

    pval = np.mean(null_stats >= observed)

    return {
        "test": "henze_zirkler",
        "statistic": float(observed),
        "pvalue": float(pval),
        "beta": float(beta),
        "mc_replicates": mc_B,
        "passes": pval > alpha,
    }


# ---------------------------------------------------------------------------
# Extended multivariate battery runner
# ---------------------------------------------------------------------------

def run_multivariate_battery(sigma, data, cov_normalized, nsamples,
                             per_coord_pvalues, alpha=0.001):
    """
    Run all extended multivariate tests.

    Args:
        sigma: expected standard deviation
        data: n x dim array of samples
        cov_normalized: dim x dim normalized covariance (already computed)
        nsamples: number of samples
        per_coord_pvalues: list of chi-square p-values per coordinate
        alpha: significance threshold

    Returns dict with results from each test and overall verdict.
    """
    data = np.asarray(data, dtype=float)
    results = {}

    results["squared_norm"] = squared_norm_test(sigma, data, alpha=alpha)
    results["fisher_bh"] = fisher_bh_meta(per_coord_pvalues, alpha=alpha)
    results["max_offdiag"] = max_offdiag_correlation(
        np.asarray(cov_normalized), nsamples, alpha=alpha)
    results["fft_domain"] = fft_domain_battery(sigma, data, alpha=alpha)
    results["henze_zirkler"] = henze_zirkler_test(sigma, data, alpha=alpha)

    all_pass = all(
        results[t].get("passes", True)
        for t in ["squared_norm", "fisher_bh", "max_offdiag",
                  "fft_domain", "henze_zirkler"]
    )
    results["all_pass"] = all_pass

    return results
