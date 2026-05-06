import numpy as np

DRIVERS = ["brent", "fx", "infl"]

def robust_cholesky(corr):
    eps = 1e-8
    for _ in range(10):
        try:
            return np.linalg.cholesky(corr)
        except np.linalg.LinAlgError:
            corr = corr + np.eye(3) * eps
            eps *= 10
    w, v = np.linalg.eigh(corr)
    w = np.clip(w, 1e-6, None)
    corr = v @ np.diag(w) @ v.T
    return np.linalg.cholesky(corr)

def simulate_drivers(n_sims, years, mu, vol, corr, seed=42):
    rng = np.random.default_rng(seed)
    L = robust_cholesky(corr)

    z = rng.standard_normal((n_sims, years, 3))
    shocks = z @ L.T

    brent = mu["brent"] + vol["brent"] * shocks[:, :, 0]
    fx    = mu["fx"]    + vol["fx"]    * shocks[:, :, 1]
    infl  = mu["infl"]  + vol["infl"]  * shocks[:, :, 2]
    infl = np.clip(infl, -0.99, None)

    return brent, fx, infl

def simulate_pnl(
    n_sims,
    years,
    base,
    betas,
    driver_mu,
    driver_vol,
    corr,
    level_shift,
    tax_rate,
    seed=42
):
    brent_shock, fx_shock, infl = simulate_drivers(
        n_sims, years, driver_mu, driver_vol, corr, seed
    )

    brent_mult = np.exp(np.cumsum(brent_shock, axis=1)) * level_shift["brent_mult"]
    fx_mult    = np.exp(np.cumsum(fx_shock, axis=1))    * level_shift["fx_mult"]
    infl_mult  = np.cumprod(1 + infl, axis=1)

    rev  = base["rev"]  * (1 + betas["rev_infl"]*(infl_mult-1) + betas["rev_fx"]*(fx_mult-1))
    cogs = base["cogs"] * (1 + betas["cogs_brent"]*(brent_mult-1)
                             + betas["cogs_infl"]*(infl_mult-1)
                             + betas["cogs_fx"]*(fx_mult-1))
    opex = base["opex"] * (1 + betas["opex_infl"]*(infl_mult-1))
    da   = base["da"]   * (1 + betas["da_infl"]*(infl_mult-1))

    rng = np.random.default_rng(seed+1)
    extra = base["extra"] + rng.normal(0, base["extra_vol"], (n_sims, years))

    ebitda = rev - cogs - opex
    ebit   = ebitda - da
    ebt    = ebit + extra
    tax    = np.maximum(0, tax_rate * ebt)
    net    = ebt - tax

    return {
        "rev": rev,
        "cogs": cogs,
        "opex": opex,
        "da": da,
        "extra": extra,
        "ebitda": ebitda,
        "ebit": ebit,
        "ebt": ebt,
        "tax": tax,
        "net": net,
    }
