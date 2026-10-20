import numpy as np
import math

def gbm_paths(S0: float, T: float, r: float, sigma: float, n_steps: int, n_paths: int, seed=None) -> np.ndarray:
    """
    Simule n_paths trajectoires de GBM (Black–Scholes) :
        dS_t = r S_t dt + sigma S_t dW_t
    Discrétisation exacte :
        S_{t+dt} = S_t * exp((r-0.5*sigma^2)dt + sigma*sqrt(dt)Z)

    Retour: matrice (n_steps+1, n_paths) avec S(0)=S0.
    """
    n_steps = int(n_steps)
    n_paths = int(n_paths)
    if n_steps <= 0 or n_paths <= 0:
        raise ValueError("n_steps et n_paths doivent être > 0")

    dt = float(T) / n_steps
    drift = (float(r) - 0.5 * float(sigma) ** 2) * dt
    diff = float(sigma) * np.sqrt(dt)

    Z = np.random.default_rng(seed).standard_normal((n_steps, n_paths))
    log_increments = drift + diff * Z

    paths = np.empty((n_steps + 1, n_paths), dtype=float)
    paths[0, :] = float(S0)
    paths[1:, :] = float(S0) * np.exp(np.cumsum(log_increments, axis=0))
    return paths

def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def bs_price(S0: float, K: float, T: float, r: float, sigma: float, option_type: str = 'call') -> float:
    option_type = str(option_type).strip().lower()
    if option_type not in {'call', 'put'}:
        raise ValueError("option_type doit être 'call' ou 'put'")
    
    S0 = float(S0)
    K = float(K)
    T = float(T)
    r = float(r)
    sigma = float(sigma)
    
    if T <= 0:
        # option à maturité
        if option_type == "call":
            return max(S0 - K, 0.0)
        return max(K - S0, 0.0)

    if sigma <= 0:
        # cas sigma=0 (déter.)
        fwd = S0 * math.exp(r * T)
        disc = math.exp(-r * T)
        if option_type == "call":
            return disc * max(fwd - K, 0.0)
        return disc * max(K - fwd, 0.0)
        
    d1 = (math.log(S0 / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    
    if option_type == 'call':
        price = S0 * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)
    else:
        price = K * math.exp(-r * T) * _norm_cdf(-d2) - S0 * _norm_cdf(-d1)
        
    return float(price)
