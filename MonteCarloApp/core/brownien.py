import numpy as np

def brownien_motion(T: float, n_steps: int) -> np.ndarray:
    """
    Simule un mouvement brownien W(t) sur [0, T] avec n_steps pas.
    Retourne un vecteur de taille n_steps+1 avec W(0)=0.
    """
    n_steps = int(n_steps)
    if n_steps <= 0:
        raise ValueError("n_steps doit être > 0")

    dt = float(T) / n_steps
    dW = np.sqrt(dt) * np.random.randn(n_steps)
    W = np.concatenate([[0.0], np.cumsum(dW)])
    return W
