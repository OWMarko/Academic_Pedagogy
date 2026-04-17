import numpy as np
import matplotlib.pyplot as plt

def minimisation_penalisation(A, b, epsilon, x0, rho, tol, nitmax):
    """
    Minimisation par pénalisation extérieure.
    rho : pas de descente (doit être petit pour la stabilité)
    """
    x = np.array(x0, dtype=float)
    history = [x.copy()]
    
    for k in range(nitmax):
        # 1. Calcul de grad J(x) = Ax - b
        grad_J = np.dot(A, x) - b
        
        # 2. Calcul de grad Psi(x)
        grad_psi = np.zeros_like(x)
        if x[1] < 0:
            grad_psi[1] = 2 * x[1]
            
        # 3. Gradient total pénalisé
        grad_total = grad_J + (1/epsilon) * grad_psi
        
        # Critère d'arrêt
        if np.linalg.norm(grad_total) < tol:
            print(f"Convergence pour eps={epsilon} en {k} itérations.")
            break
            
        # Mise à jour (pas constant)
        x = x - rho * grad_total
        history.append(x.copy())
        
    return x, np.array(history)

# Test
A = np.array([[2, 1], [1, 4]])
b = np.array([0, -3])
x0 = # initialisation

# Note : le pas rho doit être très petit quand epsilon diminue !
sol, hist = minimisation_penalisation(A, b, epsilon=1e-2, x0=x0, rho=0.005, tol=1e-6, nitmax=10000)
print(f"Solution approchée : {sol}")