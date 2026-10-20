import numpy as np
import time

from MonteCarloApp.core.black_scholes import gbm_paths, bs_price


def _simulate_ST(S0: float, T: float, r: float, sigma: float, Z: np.ndarray) -> np.ndarray:
    return float(S0) * np.exp((float(r) - 0.5 * float(sigma) ** 2) * float(T) + float(sigma) * np.sqrt(float(T)) * Z)


def _mc_stats_from_samples(samples: np.ndarray) -> dict:
    samples = np.asarray(samples, dtype=float)
    n = samples.size
    price = float(np.mean(samples))
    std = float(np.std(samples, ddof=1)) if n > 1 else 0.0
    se = std / np.sqrt(n) if n > 0 else 0.0
    return {
        "price": price,
        "std": std,
        "se": float(se),
        "ic95_low": float(price - 1.96 * se),
        "ic95_high": float(price + 1.96 * se),
        "n": int(n),
    }


def european_option_mc_report(
    S0: float, K: float, T: float, r: float, sigma: float, N: int,
    option_type: str = "call",
    method: str = "standard",
    seed=42,
    with_samples: bool = False
) -> dict:
    option_type = str(option_type).strip().lower()
    if option_type not in {"call", "put"}:
        raise ValueError("option_type doit être 'call' ou 'put'")

    method = str(method).strip().lower()
    if method not in {"standard", "antithetic", "control_variate"}:
        raise ValueError("method doit être 'standard', 'antithetic' ou 'control_variate'")

    N = int(N)
    if N < 2:
        raise ValueError("N doit être >= 2")

    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()

    if method in {"standard", "control_variate"}:
        Z = rng.standard_normal(N)
    else:
        if N % 2 != 0:
            N -= 1
        Z_half = rng.standard_normal(N // 2)
        Z = np.concatenate([Z_half, -Z_half])

    ST = _simulate_ST(S0, T, r, sigma, Z)

    if option_type == "call":
        payoff = np.maximum(ST - float(K), 0.0)
    else:
        payoff = np.maximum(float(K) - ST, 0.0)

    disc = np.exp(-float(r) * float(T))
    samples = disc * payoff  # <- IMPORTANT (distribution)

    if method == "antithetic":
        # Les tirages Z et -Z ne sont pas indépendants : l'estimateur doit être
        # évalué sur les N/2 moyennes de paires, qui elles sont i.i.d.
        pairs = 0.5 * (samples[: N // 2] + samples[N // 2:])
        stats = _mc_stats_from_samples(pairs)
        # écart-type « par simulation », comparable à celui de la méthode standard
        stats["std"] = float(stats["std"] * np.sqrt(2.0))
        stats["n"] = int(N)
    elif method == "control_variate":
        # Variable de contrôle X = e^{-rT} S_T - S0, d'espérance nulle sous Q
        X = disc * ST - float(S0)
        vx = np.var(X, ddof=1)
        beta = np.cov(samples, X, ddof=1)[0, 1] / vx if vx > 1e-18 else 0.0
        samples = samples - beta * X
        stats = _mc_stats_from_samples(samples)
    else:
        stats = _mc_stats_from_samples(samples)
    dt_ms = (time.perf_counter() - t0) * 1000.0

    bs = float(bs_price(S0, K, T, r, sigma, option_type))
    abs_error = abs(stats["price"] - bs)

    rep = {
        "method": {"standard": "Standard", "antithetic": "Antithétique", "control_variate": "Variable de contrôle"}[method],
        "option_type": option_type,
        "N": int(stats["n"]),
        "price": stats["price"],
        "std": stats["std"],
        "ic95_low": stats["ic95_low"],
        "ic95_high": stats["ic95_high"],
        "time_ms": float(dt_ms),
        "bs": bs,
        "abs_error_vs_bs": float(abs_error),
    }

    if with_samples:
        rep["samples"] = samples

    return rep



def compare_european_options(S0, K, T, r, sigma, N, option_type="call", seed=42):
    rep_std = european_option_mc_report(S0, K, T, r, sigma, N, option_type, method="standard", seed=seed)
    rep_ant = european_option_mc_report(S0, K, T, r, sigma, N, option_type, method="antithetic", seed=seed)

    var_std = rep_std["std"] ** 2
    var_ant = rep_ant["std"] ** 2
    ratio = (var_std / var_ant) if var_ant > 0 else np.nan

    rep_ratio = {
        "method": "Ratio Var(Standard)/Var(Antithétique)",
        "option_type": option_type,
        "N": min(rep_std["N"], rep_ant["N"]),
        "price": float(ratio),
        "std": np.nan,
        "ic95_low": np.nan,
        "ic95_high": np.nan,
        "time_ms": np.nan,
        "bs": rep_std["bs"],
        "abs_error_vs_bs": np.nan,
    }
    return [rep_std, rep_ant, rep_ratio]

def payoff_distributions_european(S0, K, T, r, sigma, N, option_type="call", seed=42):
    rep_std = european_option_mc_report(
        S0, K, T, r, sigma, N,
        option_type=option_type, method="standard",
        seed=seed, with_samples=True
    )
    rep_ant = european_option_mc_report(
        S0, K, T, r, sigma, N,
        option_type=option_type, method="antithetic",
        seed=seed, with_samples=True
    )
    rep_cv = european_option_mc_report(
        S0, K, T, r, sigma, N,
        option_type=option_type, method="control_variate",
        seed=seed, with_samples=True
    )

    return rep_std["samples"], rep_ant["samples"], rep_cv["samples"]



def convergence_curves_options(S0, K, T, r, sigma, option_type="call", N_list=(500, 1000, 2000, 5000, 10000, 20000), seed=42):
    prices_std = []
    prices_ant = []
    bs = float(bs_price(S0, K, T, r, sigma, option_type))

    for n in N_list:
        rep_std = european_option_mc_report(S0, K, T, r, sigma, int(n), option_type, method="standard", seed=seed)
        rep_ant = european_option_mc_report(S0, K, T, r, sigma, int(n), option_type, method="antithetic", seed=seed)
        prices_std.append(rep_std["price"])
        prices_ant.append(rep_ant["price"])

    return list(N_list), prices_std, prices_ant, bs


# ----- LSM (option américaine) : tu peux garder ton code, mais ici il est optionnel -----
def _poly_basis(x: np.ndarray, degree: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    degree = int(degree)
    cols = [np.ones_like(x)]
    for d in range(1, degree + 1):
        cols.append(x ** d)
    return np.column_stack(cols)


def american_option_lsm(S0, K, T, r, sigma, n_steps, n_paths, option_type="put", degree=2, return_diagnostics=False):
    option_type = str(option_type).strip().lower()
    if option_type not in {"put", "call"}:
        raise ValueError("option_type doit être 'put' ou 'call'")

    paths = gbm_paths(S0, T, r, sigma, int(n_steps), int(n_paths))

    dt = float(T) / int(n_steps)
    disc = np.exp(-float(r) * dt)

    if option_type == "put":
        payoff = lambda s: np.maximum(float(K) - s, 0.0)
    else:
        payoff = lambda s: np.maximum(s - float(K), 0.0)

    cf = payoff(paths[-1, :])

    coefs = [None] * (int(n_steps) + 1)
    n_itm = np.zeros(int(n_steps) + 1, dtype=int)

    for t in range(int(n_steps) - 1, 0, -1):
        St = paths[t, :]
        imm = payoff(St)
        itm = imm > 0
        n_itm[t] = int(np.sum(itm))

        cont = np.zeros_like(imm)

        if n_itm[t] >= (int(degree) + 1):
            X = _poly_basis(St[itm], int(degree))
            Y = disc * cf[itm]
            beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
            coefs[t] = beta
            cont[itm] = X @ beta

        exercise = (imm > cont) & itm
        cf = np.where(exercise, imm, disc * cf)

    price = float(np.mean(disc * cf))

    if not return_diagnostics:
        return price

    return price, {"paths": paths, "coefs": coefs, "n_itm": n_itm, "dt": dt}

def lsm_immediate_vs_continuation(diag: dict, K: float, option_type: str, t_index: int, degree: int):
    """
    Retourne (St, imm, cont, exercise_mask) au temps t_index
    pour visualiser payoff immédiat vs continuation estimée.
    """
    option_type = str(option_type).strip().lower()
    if option_type not in {"put", "call"}:
        raise ValueError("option_type doit être 'put' ou 'call'")

    paths = diag["paths"]
    coefs = diag["coefs"]

    t_index = int(t_index)
    if t_index <= 0 or t_index >= paths.shape[0] - 1:
        raise ValueError("t_index doit être dans [1, n_steps-1]")

    St = paths[t_index, :]

    if option_type == "put":
        imm = np.maximum(float(K) - St, 0.0)
    else:
        imm = np.maximum(St - float(K), 0.0)

    itm = imm > 0

    cont = np.zeros_like(imm)
    beta = coefs[t_index]

    # Si on n'a pas pu régresser à ce pas de temps, continuation=0
    if beta is not None and np.sum(itm) >= (int(degree) + 1):
        X = _poly_basis(St[itm], int(degree))
        cont[itm] = X @ beta

    exercise = (imm > cont) & itm
    return St, imm, cont, exercise

