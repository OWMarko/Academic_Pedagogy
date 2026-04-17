import numpy as np
import matplotlib.pyplot as plt

def gradient_projete_pas_constant(A, b, rho, x0, tol, nitmax):
    x = np.array(x0, dtype=float)
    history = [x.copy()]
    
    for k in range(nitmax):
        # 1. Calcul du gradient au point actuel
        grad = np.dot(A, x) - b
        
        # 2. Pas de descente vers un point intermédiaire y
        y = x - rho * grad
        
        # 3. Projection sur K (R2+) : on force les composantes à être >= 0
        x_new = np.maximum(0, y)
        
        # 4. Test d'arrêt sur l'évolution des itérées
        if np.linalg.norm(x_new - x) < tol:
            print(f"Convergence du gradient projeté en {k} itérations.")
            break
            
        x = x_new
        history.append(x.copy())
        
    return x, np.array(history)

# Paramètres
A = np.array([[2, 1], [1, 4]])
b = np.array([0, -3])
x0 = # Point de départ admissible
rho = 0.1
tol = 1e-6

x_constraint, hist_proj = gradient_projete_pas_constant(A, b, rho, x0, tol, 1000)
print(f"Minimum sous contraintes : {x_constraint}")