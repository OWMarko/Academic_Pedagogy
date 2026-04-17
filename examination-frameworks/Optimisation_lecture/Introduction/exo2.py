import numpy as np
import matplotlib.pyplot as plt

# --- DÉFINITION DE LA FONCTION f2 ET SES DÉRIVÉES ---
def f2(x):
    return 2*x**3 - 3*x**2 - 12*x + 4

def df2(x):
    return 6*x**2 - 6*x - 12

def d2f2(x):
    return 12*x - 6

def newton_opti_1d(df, d2f, x0, tol=1e-6, max_iter=100):
    """
    Cherche un extremum en cherchant un zéro de la dérivée première df.
    Nécessite la dérivée seconde d2f.
    """
    x = x0
    historique = [x]
    
    for i in range(max_iter):
        grad = df(x)
        
        # Critère d'arrêt : le gradient (la dérivée) est presque nul
        if abs(grad) < tol:
            print(f"Convergence atteinte en {i} itérations.")
            return x, historique
            
        hess = d2f(x)
        if hess == 0:
            print("Erreur : dérivée seconde nulle, division par zéro.")
            return None, historique
            
        # Mise à jour de Newton : on soustrait f'/f''
        x = x - grad / hess
        historique.append(x)
        
    print("Non convergence.")
    return x, historique

# Test 1 : Partons d'une valeur positive (devrait converger vers le minimum local)
print("--- Recherche à partir de x0 = 3.0 ---")
x_min, hist_min = newton_opti_1d(df2, d2f2, 3.0)
print(f"Extremum trouvé en x = {x_min:.5f}")
print(f"Valeur de f''(x) = {d2f2(x_min):.2f} (>0 donc c'est un MINIMUM)")

# Test 2 : Partons d'une valeur négative (devrait converger vers le maximum local)
print("\n--- Recherche à partir de x0 = -3.0 ---")
x_max, hist_max = newton_opti_1d(df2, d2f2, -3.0)
print(f"Extremum trouvé en x = {x_max:.5f}")
print(f"Valeur de f''(x) = {d2f2(x_max):.2f} (<0 donc c'est un MAXIMUM)")