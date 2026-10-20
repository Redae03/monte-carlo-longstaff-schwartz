import numpy as np   


# Simulation de la loi expo par la méthode d'inversion
def exponential_inversion(lamb, n):
    U = np.random.rand(n)
    X = -np.log(U)/lamb
    return X


# Simulation de la loi normal N(0,1) par la méthode de Box-Muller
def box_muller(n):
     U1 = np.random.rand(n)
     U2 = np.random.rand(n)
     
     R = np.sqrt(-2 * np.log(U1))
     Z1 = R * np.cos(2 * np.pi * U2)
     Z2 = R * np.sin(2 * np.pi * U2)
     
     return Z1, Z2
 
def uniform_ab(a: float, b: float, n: int):
    """
    Génère une loi uniforme U(a,b) à partir de U(0,1).
    """
    a = float(a)
    b = float(b)
    n = int(n)

    if n <= 0:
        raise ValueError("n doit être > 0")
    if b <= a:
        raise ValueError("b doit être strictement > a")

    U = np.random.rand(n)
    return a + (b - a) * U

     