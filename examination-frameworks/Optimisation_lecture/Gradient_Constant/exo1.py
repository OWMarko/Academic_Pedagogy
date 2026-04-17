import numpy as np
import matplotlib.pyplot as plt

# 1) Programmation de la méthode du gradient à pas constant
def gradient_descent_fixed_step(grad_J, x0, rho, tol=1e-6, max_iter=1000, *args):
    """
    Algorithme du gradient à pas constant.
    Retourne la solution approchée et l'historique des itérées.
    """
    x = np.array(x0, dtype=float)
    history = [x.copy()]
    
    for i in range(max_iter):
        grad = grad_J(x, *args)
        
        if np.linalg.norm(grad) < tol:
            print(f"Convergence à l'itération {i}")
            break
            
        x = x - rho * grad
        history.append(x.copy())
        
    else:
        print(f"Arrêt après {max_iter} itérations (non convergence).")
        
    return np.array(history)

# Fonction utilitaire pour tracer l'évolution
def plot_evolution(history, title):
    plt.figure(figsize=(8, 4))
    if history.ndim == 1 or history.shape[1] == 1:
        plt.plot(history, marker='o', markersize=4)
        plt.ylabel('x')
    else:
        # On trace la distance par rapport au point final pour visualiser la convergence
        distances = [np.linalg.norm(x - history[-1]) for x in history]
        plt.plot(distances, marker='o', markersize=4)
        plt.ylabel('||x_k - x*||')
        
    plt.xlabel('Itérations (k)')
    plt.title(title)
    plt.grid(True)
    plt.yscale('log') # Echelle logarithmique souvent plus lisible en optimisation
    plt.show()

# 2) Test sur J(x) = x^2 + sin(x) sur R
print("--- Test 2 : x^2 + sin(x) ---")
def grad_test2(x):
    return 2*x + np.cos(x)

hist2 = gradient_descent_fixed_step(grad_test2, x0=[2.0], rho=0.1)
print(f"Minimum trouvé : {hist2[-1]:.5f}\n")
plot_evolution(hist2, "Convergence pour J(x) = x^2 + sin(x)")

# 3) Test sur J1(x,y) = (x-1)^2 + 10(y-1)^2 et J2(x,y) = (x-1)^2 + 10(x^2-y)^2
print("--- Test 3 : Fonctions quadratique et de Rosenbrock ---")
def grad_J1(X):
    x, y = X, X[1]
    return np.array([2*(x - 1), 20*(y - 1)])

def grad_J2(X): # Fonction de type Rosenbrock
    x, y = X, X[1]
    return np.array([2*(x - 1) + 40*x*(x**2 - y), -20*(x**2 - y)])

hist3_1 = gradient_descent_fixed_step(grad_J1, x0=[0.0, 0.0], rho=0.08)
print(f"Minimum J1 trouvé : {hist3_1[-1]}")

hist3_2 = gradient_descent_fixed_step(grad_J2, x0=[-0.5, 1.5], rho=0.005) # Pas très petit nécessaire !
print(f"Minimum J2 trouvé : {hist3_2[-1]}\n")

# 4) Test sur le problème des moindres carrés (distance à N points)
print("--- Test 4 : Centre de gravité de N points ---")
def grad_J4(x, A):
    # Le gradient de sum(||x - Ai||^2) est sum(2*(x - Ai))
    N = A.shape
    return 2 * N * x - 2 * np.sum(A, axis=0)

# Définition de 4 points arbitraires dans R^2
A_points = np.array([[0,0], [0,2], [2,0], [2, 2]])
# Le minimum théorique est la moyenne des points (le barycentre) soit (1, 1)

hist4 = gradient_descent_fixed_step(grad_J4, x0=[5.0, -3.0], rho=0.05, args=(A_points,))
print(f"Barycentre trouvé : {hist4[-1]}\n")

# 5) Test sur J(x) = sum(i * x_i^2) dans R^n
print("--- Test 5 : Fonction quadratique en dimension n ---")
def grad_J5(x):
    # Le gradient a pour composante i : 2 * i * x_i
    n = len(x)
    i_indices = np.arange(1, n + 1)
    return 2 * i_indices * x

n_dim = 10
x0_dim_n = np.ones(n_dim) # Point de départ : vecteur de 1
# Le pas max théorique est 2/L où L est la plus grande valeur propre (ici 2*n)
# Pour n=10, L=20, on prend un pas rho < 0.1
hist5 = gradient_descent_fixed_step(grad_J5, x0=x0_dim_n, rho=0.04)
print(f"Minimum dimension {n_dim} trouvé (norme) : {np.linalg.norm(hist5[-1]):.5e}")
plot_evolution(hist5, f"Convergence pour sum(i * x_i^2) dans R^{n_dim}")