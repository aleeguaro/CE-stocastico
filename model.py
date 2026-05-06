
import numpy as np

def _safe_cholesky(corr, max_tries=8):
    corr = np.asarray(corr, dtype=float)
    eps = 1e-10
    for _ in range(max_tries):
        try:
            return np.linalg.cholesky(corr)
        except np.linalg.LinAlgError:
            corr = corr + np.eye(corr.shape[0]) * eps
            eps *= 10
    w, v = np.linalg.eigh(corr)
    w = np.clip(w, 1e-8, None)
    corr_psd = (v * w) @ v.T
    return np.linalg.cholesky(corr_psd + np.eye(corr.shape[0]) * 1e-8)

def simulate_drivers(n_sims, years, mu, vol, corr, seed=42):
    rng = np.random.default_rng(seed)
    L = _safe_cholesky(corr)
    z = rng.standard_normal(size=(n_sims, years, 3))
    eps = z @ L.T

    brent = mu["brent"] + vol["brent"] * eps[:, :, 0]
    fx    = mu["fx"]    + vol["fx"]    * eps[:, :, 1]
    infl  = mu["infl"]  + vol["infl"]  * eps[:, :, 2]
    infl = np.clip(infl, -0.99, None)
    return brent, fx, infl

def simulate_pnl(n_sims, years, base, betas,
                 driver_mu, driver_vol, corr, level_shift,
                 tax_rate=0.28, seed=42):

    brent_chg, fx_chg, infl = simulate_drivers(
        n_sims, years, driver_mu, driver_vol, corr, seed
    )

    brent_mult = np.exp(np.cumsum(brent_chg, axis=1)) * level_shift["brent_mult"]
    fx_mult    = np.exp(np.cumsum(fx_chg, axis=1))    * level_shift["fx_mult"]
    infl_mult  = np.cumprod(1 + infl, axis=1)

    rev  = base["rev"]  * (1 + betas["rev_infl"]*(infl_mult-1) + betas["rev_fx"]*(fx_mult-1))
    cogs = base["cogs"] * (1 + betas["cogs_brent"]*(brent_mult-1) +
                           betas["cogs_infl"]*(infl_mult-1) +
                           betas["cogs_fx"]*(fx_mult-1))
    opex = base["opex"] * (1 + betas["opex_infl"]*(infl_mult-1))
    da   = base["da"]   * (1 + betas["da_infl"]*(infl_mult-1))

    rng = np.random.default_rng(seed+1)
    extra = base["extra"] + rng.normal(0, base["extra_vol"], size=(n_sims, years))

    ebitda = rev - cogs - opex
    ebit   = ebitda - da
    ebt    = ebit + extra
    tax    = np.maximum(0, tax_rate * ebt)
    net    = ebt - tax

    return {"ebitda": ebitda, "ebit": ebit, "ebt": ebt, "net": net}
