import numpy as np
import matplotlib.pyplot as plt

def gradient_pas_optimal(A, b, x0, tol=1e-6, nitmax=1000):
    """
    Algorithme du gradient à pas optimal pour une fonctionnelle quadratique.
    """
    x = np.array(x0, dtype=float)
    history = [x.copy()]
    
    print(f"{'Iter':<5} | {'Produit scalaire <g_k+1, g_k>':<30}")
    print("-" * 40)

    for k in range(nitmax):
        # 1. Calcul du résidu r_k = b - Ax_k (qui est l'opposé du gradient)
        # Rappel : grad J(x) = Ax - b, donc r_k = -grad J(x_k)
        rk = b - np.dot(A, x)
        
        # Critère d'arrêt sur la norme du gradient (ou du résidu)
        if np.linalg.norm(rk) < tol:
            print(f"--- Convergence atteinte en {k} itérations ---")
            break
            
        # 2. Calcul du pas optimal rho_k
        Ark = np.dot(A, rk)
        rho_k = np.dot(rk, rk) / np.dot(Ark, rk)
        
        # Sauvegarde du gradient actuel pour la question 3
        grad_k = -rk 
        
        # 3. Mise à jour de la position
        x = x + rho_k * rk
        history.append(x.copy())
        
        # --- Vérification de la question 3 ---
        # Calcul du nouveau gradient au point x_k+1
        grad_next = np.dot(A, x) - b
        prod_scalaire = np.dot(grad_next, grad_k)
        if k < 5: # On affiche les 5 premières pour l'exemple
            print(f"{k+1:<5} | {prod_scalaire:<30.2e}")
            
    return x, np.array(history)

# --- Paramètres de l'exercice ---
A = np.array([[2, 1], 
              [1, 4]])
b = np.array([0, -3])
x0 = np.array([0, 0])

# Exécution
x_final, history = gradient_pas_optimal(A, b, x0)

print(f"\nSolution finale : {x_final}")