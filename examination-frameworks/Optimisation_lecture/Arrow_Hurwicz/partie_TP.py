import numpy as np

def arrow_hurwicz(A, b, c, d, rho1, rho2, u0, p0, tol, nitmax):
    """
    Algorithme d'Arrow-Hurwicz pour la minimisation sous contrainte.
    """
    u = np.array(u0, dtype=float)
    p = float(p0)
    history = [u.copy()]
    
    for k in range(nitmax):
        u_old = u.copy()
        
        # Étape 1 : Descente de gradient sur le Lagrangien (Variable primale u)
        grad_L_u = A @ u - b + p * c
        u = u - rho1 * grad_L_u
        
        # Étape 2 : Montée de gradient projetée (Variable duale p)
        constraint_val = np.dot(c, u_old) - d # On utilise souvent l'itérée précédente pour p
        p = max(0, p + rho2 * constraint_val)
        
        history.append(u.copy())
        
        # Test d'arrêt sur la stabilité de u
        if np.linalg.norm(u - u_old) < tol:
            print(f"Convergence Arrow-Hurwicz atteinte en {k} itérations.")
            break
            
    return u, p, np.array(history)

# --- Exemple de test ---
A = np.array([[2, 1], [1, 4]])
b = np.array([0, -3])
c = np.array([1, 1])
d = 1.0

# Initialisation
u0 = [0.0, 0.0]
p0 = 0.0

# Note : rho1 et rho2 doivent être choisis assez petits
sol_u, sol_p, hist = arrow_hurwicz(A, b, c, d, 0.05, 0.1, u0, p0, 1e-6, 5000)

print(f"Solution u : {sol_u}")
print(f"Multiplicateur p : {sol_p:.4f}")