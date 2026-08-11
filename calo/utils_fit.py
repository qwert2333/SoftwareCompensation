# calo/utils_fit.py
import numpy as np
from scipy.optimize import curve_fit

def gauss(x, mu, sigma, A):
    return A * np.exp(-(x - mu)**2 / (2 * sigma**2))

def resolution_func(E, a, b):
    return np.sqrt((a / np.sqrt(E))**2 + b**2)

def pcov_is_valid(pcov):
    if pcov is None:
        return False
    if not np.all(np.isfinite(pcov)):
        return False
    # diag 不能为负或 0（极端退化）
    d = np.diag(pcov)
    return np.all(d > 0)

def fit_gauss_hist(preds, nbins=60, nsigma=1.5, n_iter=2):
    preds = np.asarray(preds, dtype=np.float64)
    preds = preds[np.isfinite(preds)]
    if preds.size == 0:
        raise ValueError("No finite predictions to fit")

    q16, q50, q84 = np.percentile(preds, [16, 50, 84])
    sigma0 = max(0.5 * (q84 - q16), float(np.std(preds)), 1e-6)
    q_lo, q_hi = np.percentile(preds, [0.5, 99.5])
    lo = max(float(np.min(preds)), min(float(q_lo), float(q50 - 4.0 * sigma0)))
    hi = min(float(np.max(preds)), max(float(q_hi), float(q50 + 4.0 * sigma0)))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.min(preds))
        hi = float(np.max(preds))
        if hi <= lo:
            hi = lo + 1e-3

    n_eff_bins = int(np.clip(max(np.sqrt(preds.size), 25), 25, nbins))
    hist, edges = np.histogram(preds, bins=n_eff_bins, range=(lo, hi))
    centers = 0.5*(edges[:-1] + edges[1:])

    mu = float(np.mean(preds))
    sig = float(np.std(preds) + 1e-9)
    A = float(max(hist.max(), 1.0))
    popt = [mu, sig, A]
    pcov = None

    for _ in range(n_iter):
        mu, sig = float(popt[0]), abs(float(popt[1]))
        m = (centers > mu - nsigma*sig) & (centers < mu + nsigma*sig) & (hist > 0)
        xfit, yfit = centers[m], hist[m]
        if len(xfit) < 6 or np.sum(yfit) == 0:
            break
        bounds = ([-np.inf, 1e-6, 0.0], [np.inf, np.inf, np.inf])
        popt, pcov = curve_fit(gauss, xfit, yfit, p0=[mu, sig, max(A,1.0)],
                               bounds=bounds, maxfev=20000)

    mu_fit = float(popt[0])
    sig_fit = abs(float(popt[1]))
    A_fit = float(popt[2])

    # chi2 on fit window
    m = (centers > mu_fit - nsigma*sig_fit) & (centers < mu_fit + nsigma*sig_fit) & (hist > 0)
    expv = gauss(centers[m], mu_fit, sig_fit, A_fit)
    obs = hist[m]
    chi2 = float(np.sum((obs-expv)**2 / (expv + 1e-6)))
    ndf = int(np.sum(m) - 3)
    chi2_ndf = chi2/ndf if ndf > 0 else float("nan")

    return (mu_fit, sig_fit, A_fit, pcov, chi2_ndf, hist, edges, centers)
