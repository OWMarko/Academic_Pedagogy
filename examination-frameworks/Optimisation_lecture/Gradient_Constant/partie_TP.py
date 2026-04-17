import numpy as np
import matplotlib.pyplot as plt

# 3)
def gradient_pas_constant(A, b, rho, x0, tol, nitmax):
    """
    Implémentation de la descente de gradient à pas fixe.
    """
    x = np.array(x0, dtype=float)
    history = [x.copy()]
    costs = []
    
    # Calcul de la solution exacte pour l'erreur (x* = A^-1 * b)
    x_star = np.linalg.solve(A, b)
    errors = []

    for k in range(nitmax):
        # Calcul du gradient : Ax - b
        grad = A.dot(x) - b
        norm_grad = np.linalg.norm(grad)
        
        # Calcul du coût J(x) et de l'erreur
        cost = 0.5 * np.dot(A.dot(x), x) - np.dot(b, x)
        costs.append(cost)
        errors.append(np.linalg.norm(x - x_star))
        
        if norm_grad < tol:
            print(f"Convergence atteinte en {k} itérations.")
            break
            
        # Mise à jour
        x = x - rho * grad
        history.append(x.copy())
    else:
        print("Nombre max d'itérations atteint sans convergence.")

    return x, np.array(history), np.array(costs), np.array(errors)

# 5) 
# Paramètres
A = np.array([[2, 1], [1, 4]])
b = np.array([0, -3])
x0 = np.array([0,0])
tol = 1e-6
nitmax = 1000

# Simulation
x_final, history, costs, errors = gradient_pas_constant(A, b, 0.1, x0, tol, nitmax)

# Tracé
plt.figure(figsize=(10, 5))
plt.semilogy(errors, label="Erreur $||x_k - x^*||$")
plt.title("Évolution de l'erreur (Échelle logarithmique)")
plt.xlabel("Itérations k")
plt.ylabel("Erreur")
plt.grid(True)
plt.legend()
plt.show()

