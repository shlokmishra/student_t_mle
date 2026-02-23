# Build normalized posterior p(theta | hat_theta = mu_star) from MLE samples at theta=0.
# Location family: p(hat_theta | theta=a) = p(hat_theta - a | 0). Posterior propto prior * likelihood.
#
# Bandwidth (bw_method): Scott/Silverman assume roughly Gaussian data. The MLE distribution
# for Student-t/Cauchy is heavy-tailed and non-Gaussian, so these rules often oversmooth
# (too large h → KDE too flat → posterior too diffuse). For moderate n (e.g. n=100), a
# small fixed bandwidth (e.g. 0.001) usually works better; the "right" value can depend on
# n and k (smaller n → MLE more spread → sometimes a slightly larger h is needed).

import numpy as np
import scipy.stats as stats
from scipy.integrate import quad


def _grid_bounds_for_normalisation(prior_mean, prior_std, mu_star, mles, n_sigma=5):
    """
    Choose grid bounds so that prior * KDE has negligible mass outside [lo, hi].
    Prior mass is in prior_mean +/- n_sigma*prior_std.
    Likelihood (KDE of mu_star - mu) has spread in mu of order std(mles) around mu_star.
    """
    mles = np.asarray(mles)
    prior_radius = n_sigma * prior_std
    like_scale = np.std(mles)
    if like_scale <= 0:
        like_radius = 5.0
    else:
        like_radius = n_sigma * like_scale
    lo = min(prior_mean - prior_radius, mu_star - like_radius)
    hi = max(prior_mean + prior_radius, mu_star + like_radius)
    return lo, hi


def validate_posterior_1d(posterior_pdf, lo=-20.0, hi=20.0, n_grid=5000):
    """
    Validate that the 1D posterior integrates to 1 (post = like * prior / norm_const).

    posterior_pdf: callable mu -> density (e.g. from get_normalized_posterior_pdf).
    lo, hi: integration bounds; n_grid: number of points.
    Returns: integral (float), should be close to 1.0.
    """
    mu_grid = np.linspace(lo, hi, n_grid)
    vals = np.maximum(posterior_pdf(mu_grid), 0.0)
    integral = float(np.trapezoid(vals, mu_grid))
    return integral

import numpy as np
from scipy import stats
from scipy.integrate import quad
from scipy.special import gammaln, logsumexp

# ---------------------------------------------------------------------
# Helper KDE backends
# ---------------------------------------------------------------------

def _robust_scale(x):
    x = np.asarray(x)
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    if mad > 0:
        return mad / 0.6745
    iqr = np.subtract(*np.percentile(x, [75, 25]))
    if iqr > 0:
        return iqr / 1.349
    return float(np.std(x, ddof=1) + 1e-12)

def _silverman_bandwidth(x):
    x = np.asarray(x)
    n = x.size
    sigma = _robust_scale(x)
    return 0.9 * sigma * n ** (-1 / 5)

class _GaussianKDEWrapper:
    def __init__(self, samples, bw_method):
        self.kde = stats.gaussian_kde(np.asarray(samples), bw_method=bw_method)

    def logpdf(self, x):
        x = np.asarray(x)
        return self.kde.logpdf(x.reshape(1, -1)).ravel()

class _SJAsinhKDE:
    """
    KDE on y = asinh(m/s) with Sheather-Jones bandwidth on y,
    then back-transform:
        f_M(m) = f_Y(asinh(m/s)) / sqrt(m^2 + s^2)
    """
    def __init__(self, samples, s=None, gridsize=8192, eps=1e-300):
        self.samples = np.asarray(samples)
        self.eps = eps
        self.s = float(s) if s is not None else float(_robust_scale(self.samples))

        y = np.arcsinh(self.samples / self.s)
        self.y = y

        # Try statsmodels SJ; fallback to Gaussian KDE on y with Silverman.
        self._use_statsmodels = False
        try:
            from statsmodels.nonparametric.kde import KDEUnivariate
            kde = KDEUnivariate(y)
            kde.fit(bw="sj", fft=False, gridsize=gridsize, cut=0)
            self._use_statsmodels = True
            self.support = np.asarray(kde.support)
            self.density = np.asarray(kde.density)
        except Exception:
            # Fallback: Gaussian KDE on y
            h = _silverman_bandwidth(y)
            self._fallback_kde = stats.gaussian_kde(y, bw_method=h / (np.std(y, ddof=1) + 1e-12))

    def logpdf(self, m):
        m = np.asarray(m)
        yg = np.arcsinh(m / self.s)

        if self._use_statsmodels:
            fy = np.interp(yg, self.support, self.density, left=0.0, right=0.0)
        else:
            fy = self._fallback_kde.evaluate(yg)

        # Back-transform Jacobian dy/dm = 1/sqrt(m^2 + s^2)
        fm = np.maximum(fy / np.sqrt(m * m + self.s * self.s), self.eps)
        return np.log(fm)

class _StudentAbramsonKDE:
    """
    Student-t kernel KDE with adaptive bandwidths (Abramson square-root law).

    Final density:
        f(x) = (1/n) sum_i [ 1/h_i * t_nu( (x - x_i)/h_i ) ]
    where h_i = h0 * (f_pilot(x_i)/g)^(-alpha), g = geometric mean of f_pilot(x_i),
    alpha=0.5 gives Abramson.

    NOTE: This is O(n*m) to evaluate m points. For huge n, consider binning/FFT
    approximations (not implemented here).
    """
    def __init__(
        self,
        samples,
        nu=3,
        h0=None,
        pilot_nu=5,
        pilot_h=None,
        alpha=0.5,
        eps=1e-300,
        chunk=4096,
    ):
        self.x = np.asarray(samples)
        self.n = self.x.size
        self.nu = float(nu)
        self.alpha = float(alpha)
        self.eps = eps
        self.chunk = int(chunk)

        if h0 is None:
            h0 = _silverman_bandwidth(self.x)
        if pilot_h is None:
            pilot_h = h0
        self.h0 = float(h0)
        self.pilot_h = float(pilot_h)
        self.pilot_nu = float(pilot_nu)

        # Precompute pilot density at sample points: f_pilot(x_i)
        # Using fixed Student kernel with bandwidth pilot_h.
        f_pilot_x = self._fixed_student_kde_at_points(self.x, self.x, self.pilot_h, self.pilot_nu)
        f_pilot_x = np.maximum(f_pilot_x, self.eps)

        g = np.exp(np.mean(np.log(f_pilot_x)))
        hi = self.h0 * (f_pilot_x / g) ** (-self.alpha)
        self.hi = np.maximum(hi, 1e-15)

    @staticmethod
    def _student_logkernel(u, nu):
        # log pdf of standard Student-t with df=nu at u
        # c = Gamma((nu+1)/2) / (sqrt(nu*pi)*Gamma(nu/2))
        # log c = gammaln((nu+1)/2) - 0.5*log(nu*pi) - gammaln(nu/2)
        logc = gammaln((nu + 1.0) / 2.0) - 0.5 * np.log(nu * np.pi) - gammaln(nu / 2.0)
        return logc - ((nu + 1.0) / 2.0) * np.log1p((u * u) / nu)

    def _fixed_student_kde_at_points(self, eval_points, centers, h, nu):
        # f(eval) = mean_i 1/h * t_nu((eval - centers_i)/h)
        eval_points = np.asarray(eval_points)
        centers = np.asarray(centers)
        h = float(h)
        nu = float(nu)

        out = np.empty(eval_points.shape[0], dtype=float)
        for a in range(0, eval_points.shape[0], self.chunk):
            b = min(a + self.chunk, eval_points.shape[0])
            xp = eval_points[a:b][:, None]  # (m_chunk, 1)
            u = (xp - centers[None, :]) / h
            log_terms = self._student_logkernel(u, nu) - np.log(h)
            out[a:b] = np.exp(logsumexp(log_terms, axis=1) - np.log(centers.size))
        return out

    def logpdf(self, x_eval):
        x_eval = np.asarray(x_eval)
        # Evaluate adaptive KDE:
        # log f(x) = logmeanexp_i [ log t((x-x_i)/h_i) - log h_i ]
        out = np.empty(x_eval.shape[0], dtype=float)

        # chunk over eval points to manage memory
        for a in range(0, x_eval.shape[0], self.chunk):
            b = min(a + self.chunk, x_eval.shape[0])
            xg = x_eval[a:b][:, None]  # (m_chunk, 1)

            # u_ij = (xg_j - x_i)/h_i, broadcast h_i over rows
            u = (xg - self.x[None, :]) / self.hi[None, :]
            log_terms = self._student_logkernel(u, self.nu) - np.log(self.hi)[None, :]
            out[a:b] = logsumexp(log_terms, axis=1) - np.log(self.n)

        return np.log(np.maximum(np.exp(out), self.eps))

# ---------------------------------------------------------------------
# Main function (extended)
# ---------------------------------------------------------------------

def get_normalized_posterior_pdf(
    mu_star,
    params,
    mle_samples,
    verbose=False,
    use_grid=True,
    n_grid=4000,
):
    """
    Return a callable for the normalized posterior density p(mu | hat_mu = mu_star).

    Likelihood proxy: p(hat_mu=mu_star | mu) ≈ f0(mu_star - mu),
    where f0 is KDE fit once to MLE samples generated under mu=0.

    bw_method options:
      - "scott" / "silverman" / float: scipy.stats.gaussian_kde on raw MLE samples
      - "t_abram": Student-t kernel + Abramson adaptive bandwidth (heavy-tail friendly)
      - "SJ_transform": asinh-transform + Sheather-Jones KDE on transformed scale (heavy-tail friendly)
    """
    mles = np.asarray(mle_samples)
    prior_mean = params["prior_mean"]
    prior_std = params["prior_std"]
    bw_method = params.get("kde_bw_method", "scott")

    # Build likelihood KDE backend
    if isinstance(bw_method, str) and bw_method.lower() == "t_abram":
        nu = params.get("t_nu", 3)
        pilot_nu = params.get("t_pilot_nu", 5)
        h0 = params.get("t_h0", None)          # if None -> Silverman
        pilot_h = params.get("t_pilot_h", None)
        alpha = params.get("t_alpha", 0.5)
        chunk = params.get("kde_chunk", 4096)
        if verbose:
            print(f"Fitting Student-Abramson KDE: nu={nu}, pilot_nu={pilot_nu}, h0={h0}, pilot_h={pilot_h}, alpha={alpha}")
        kde_like = _StudentAbramsonKDE(
            mles, nu=nu, h0=h0, pilot_nu=pilot_nu, pilot_h=pilot_h, alpha=alpha, chunk=chunk
        )

    elif isinstance(bw_method, str) and bw_method.lower() == "sj_transform":
        s = params.get("sj_s", None)  # optional fixed scale for asinh; else robust
        gridsize = params.get("sj_gridsize", 8192)
        if verbose:
            print(f"Fitting asinh+SJ KDE: s={s}, gridsize={gridsize}")
        kde_like = _SJAsinhKDE(mles, s=s, gridsize=gridsize)

    else:
        # Default: Gaussian KDE (scott/silverman/float)
        if verbose:
            print("Fitting Gaussian KDE on MLE samples using bw_method =", bw_method)
        kde_like = _GaussianKDEWrapper(mles, bw_method=bw_method)

    def log_unnorm(mu):
        mu_a = np.atleast_1d(mu)
        log_prior = stats.norm.logpdf(mu_a, loc=prior_mean, scale=prior_std)

        # Likelihood proxy: f0(mu_star - mu)
        z = (mu_star - mu_a)
        log_like = kde_like.logpdf(z)

        out = log_prior + log_like
        return float(out[0]) if np.isscalar(mu) else out

    if use_grid:
        lo, hi = _grid_bounds_for_normalisation(prior_mean, prior_std, mu_star, mles)
        mu_grid = np.linspace(lo, hi, n_grid)
        # stabilize: subtract max log
        lv = log_unnorm(mu_grid)
        lv_max = float(np.max(lv))
        unnorm_vals = np.exp(lv - lv_max)
        integral = float(np.trapezoid(unnorm_vals, mu_grid)) * np.exp(lv_max)
    else:
        integral, _ = quad(lambda mu: np.exp(log_unnorm(mu)), -np.inf, np.inf)

    def normalized_pdf(mu):
        return np.exp(log_unnorm(mu)) / integral

    return normalized_pdf