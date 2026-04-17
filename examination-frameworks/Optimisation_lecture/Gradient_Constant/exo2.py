import numpy as np
import matplotlib.pyplot as plt

# 1) Gradient à pas constant
def gradient_pas_constant(A, b, x0, rho, tol=1e-6, max_iter=1000):
    x = np.array(x0, dtype=float)
    history = [x.copy()]
    
    for k in range(max_iter):
        r = b - A.dot(x) # Résidu = -Gradient
        
        if np.linalg.norm(r) < tol:
            break
            
        x = x + rho * r
        history.append(x.copy())
        
    return np.array(history), k

# 2) Gradient à pas optimal
def gradient_pas_optimal(A, b, x0, tol=1e-6, max_iter=1000):
    x = np.array(x0, dtype=float)
    history = [x.copy()]
    
    for k in range(max_iter):
        r = b - A.dot(x)
        
        if np.linalg.norm(r) < tol:
            break
            
        # Calcul du pas optimal
        Ar = A.dot(r)
        rho = np.dot(r, r) / np.dot(Ar, r)
        
        x = x + rho * r
        history.append(x.copy())
        
    return np.array(history), k

# 3) Algorithme du Gradient Conjugué (CG)
def gradient_conjugue(A, b, x0, tol=1e-6, max_iter=1000):
    x = np.array(x0, dtype=float)
    history = [x.copy()]
    
    r = b - A.dot(x)
    p = r.copy() # Première direction = direction du gradient
    
    for k in range(max_iter):
        if np.linalg.norm(r) < tol:
            break
            
        Ap = A.dot(p)
        # Taille du pas dans la direction conjuguée p
        alpha = np.dot(r, r) / np.dot(Ap, p)
        
        x = x + alpha * p
        history.append(x.copy())
        
        r_next = r - alpha * Ap
        # Calcul du coefficient beta pour conjuguer la nouvelle direction
        beta = np.dot(r_next, r_next) / np.dot(r, r)
        
        p = r_next + beta * p
        r = r_next
        
    return np.array(history), k

# TEST ET COMPARAISON DES ALGORITHMES
np.random.seed(42)
N = 50
# Création d'une matrice symétrique définie positive mal conditionnée
M = np.random.randn(N, N)
A = M.T.dot(M) + 0.1 * np.eye(N) # S'assure que A est SPD
b = np.random.randn(N)
x0 = np.zeros(N)

# Solution exacte pour calculer l'erreur
x_star = np.linalg.solve(A, b)

# Le pas constant nécessite de connaître la valeur propre max de A
# La théorie dit que 0 < rho < 2/lambda_max pour converger
lambda_max = np.max(np.linalg.eigvals(A))
rho = 1.0 / lambda_max # Un pas sécuritaire

hist_cst, iter_cst = gradient_pas_constant(A, b, x0, rho)
hist_opt, iter_opt = gradient_pas_optimal(A, b, x0)
hist_cg, iter_cg = gradient_conjugue(A, b, x0)

print(f"Itérations Pas Constant : {iter_cst}")
print(f"Itérations Pas Optimal  : {iter_opt}")
print(f"Itérations Grad Conjugué: {iter_cg}")

# Calcul des erreurs par rapport à x_star pour chaque itération
err_cst = [np.linalg.norm(x - x_star) for x in hist_cst]
err_opt = [np.linalg.norm(x - x_star) for x in hist_opt]
err_cg = [np.linalg.norm(x - x_star) for x in hist_cg]

# Affichage graphique
plt.figure(figsize=(10, 6))
plt.plot(err_cst, label=f'Pas constant (rho={rho:.4f})', linewidth=2)
plt.plot(err_opt, label='Pas optimal', linewidth=2)
plt.plot(err_cg, label='Gradient Conjugué', linewidth=2, color='red')

plt.yscale('log')
plt.xlabel('Nombre d\'itérations')
plt.ylabel('Erreur ||x_k - x*||')
plt.title('Comparaison de la vitesse de convergence')
plt.legend()
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.show()