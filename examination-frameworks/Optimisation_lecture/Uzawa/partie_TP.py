import numpy as np

def uzawa(A, b, c, d, mu, lambda0, tol, nitmax):
    """
    Algorithme d'Uzawa pour une contrainte d'inégalité affine.
    h(x) = <c, x> - d <= 0
    """
    # Initialisation
    lambd = float(lambda0)
    u = np.zeros(A.shape)
    
    for k in range(nitmax):
        u_old = u.copy()
        
        # Étape 1 : Minimisation en u (résolution de système linéaire)
        # On résout A*u = b - lambda*c
        rhs = b - lambd * c
        u = np.linalg.solve(A, rhs)
        
        # Étape 2 : Mise à jour du multiplicateur lambda
        # h(u) = np.dot(c, u) - d
        contrainte = np.dot(c, u) - d
        lambd_next = max(0, lambd + mu * contrainte)
        
        # Critère d'arrêt sur la stabilité de la solution primale
        if np.linalg.norm(u - u_old) < tol:
            print(f"Convergence atteinte en {k} itérations.")
            break
            
        lambd = lambd_next
    else:
        print("Nombre maximal d'itérations atteint.")
        
    return u, lambd

# Exemple de test avec les données du TP
A = np.array([[2, 1], [1, 4]])
b = np.array([0, -3])
c = np.array([1, 1])
d = 1.0

# mu doit être choisi avec précaution (généralement lié à la plus petite valeur propre de A)
sol_u, sol_lambd = uzawa(A, b, c, d, mu=0.5, lambda0=0.0, tol=1e-6, nitmax=1000)

print(f"Solution approchée u : {sol_u}")
print(f"Multiplicateur final lambda : {sol_lambd:.4f}")