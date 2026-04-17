import numpy as np
import matplotlib.pyplot as plt

def projection_Rd_plus(u):
    """Projette le vecteur u sur le quart de plan positif (K = Rd+)."""
    return np.maximum(0, u)

def gradient_projete_quadratique(A, b, rho, u0, eps, nitmax):
    """
    Question 1 : Gradient projeté pour la forme quadratique J(x) = 1/2 <Ax, x> - <b, x>
    """
    u = np.array(u0, dtype=float)
    
    for k in range(nitmax):
        # Calcul du gradient
        grad = np.dot(A, u) - b
        
        # Pas de descente + Projection
        u_next = projection_Rd_plus(u - rho * grad)
        
        # Test d'arrêt sur la différence entre deux itérées (stabilité)
        if np.linalg.norm(u_next - u) < eps:
            print(f"Convergence (Q1) atteinte en {k} itérations.")
            return u_next
        
        u = u_next
        
    print("Attention : nitmax atteint sans convergence (Q1).")
    return u

def gradient_projete_moindres_carres(M, f, rho, x0, eps, nitmax):
    """
    Question 2 : Gradient projeté pour J(x) = ||Mx - f||^2
    """
    x = np.array(x0, dtype=float)
    MT = M.T # On pré-calcule la transposée
    
    for k in range(nitmax):
        # Calcul du gradient : 2 * M.T * (M*x - f)
        grad = 2 * np.dot(MT, np.dot(M, x) - f)
        
        # Pas de descente + Projection
        x_next = projection_Rd_plus(x - rho * grad)
        
        # Test d'arrêt
        if np.linalg.norm(x_next - x) < eps:
            print(f"Convergence (Q2) atteinte en {k} itérations.")
            return x_next
        
        x = x_next
        
    print("Attention : nitmax atteint sans convergence (Q2).")
    return x

# TEST DES ALGORITHMES

# Données de l'énoncé
M_test = np.array([[2, 1], 
                   [1, 4]])
f_test = np.array([0, -3])

# Paramètres de contrôle
rho = 0.05
u0 = # à mettre
eps = 1e-6
nitmax = 1000

# Exécution du Test 2
solution_q2 = gradient_projete_moindres_carres(M_test, f_test, rho, u0, eps, nitmax)

print("-" * 30)
print(f"Résultat Question 2 :")
print(f"x_final = {solution_q2}")