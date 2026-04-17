import numpy as np

def newton_1d(f, df, x0, tol=1e-6, max_iter=100):
    """
    Algorithme de Newton basique en dimension 1.
    Retourne la racine trouvée et le nombre d'itérations.
    """
    x = x0
    for i in range(max_iter):
        fx = f(x)
        
        # Critère d'arrêt
        if abs(fx) < tol:
            return x, i
            
        dfx = df(x)
        if dfx == 0:
            print("Erreur : dérivée nulle.")
            return None, i
            
        # Mise à jour de Newton
        x = x - fx / dfx
        
    print("Non convergence atteinte.")
    return x, max_iter

# Application à la question 1
def f1(x):
    return np.exp(-x) - x

def df1(x):
    return -np.exp(-x) - 1

x0 = 0.0 # Point de départ arbitraire
racine1, iters1 = newton_1d(f1, df1, x0)

print(f"Question 1 : Racine approchée = {racine1:.6f} trouvée en {iters1} itérations.")

# Application à la question 2
def f2(x):
    return x**2 - 2

def df2(x):
    return 2*x

# Test avec un point de départ positif
racine2_pos, iters2_pos = newton_1d(f2, df2, 2.0)
print(f"Question 2 (départ 2.0) : Racine = {racine2_pos:.6f} en {iters2_pos} itérations.")

# Test avec un point de départ négatif
racine2_neg, iters2_neg = newton_1d(f2, df2, -2.0)
print(f"Question 2 (départ -2.0) : Racine = {racine2_neg:.6f} en {iters2_neg} itérations.")


# Question 3
def newton_2d(F, J, X0, tol=1e-6, max_iter=100):
    """
    Algorithme de Newton en dimension n.
    """
    X = np.array(X0, dtype=float)
    
    for i in range(max_iter):
        Fx = F(X)
        
        # Critère d'arrêt sur la norme du vecteur F(X)
        if np.linalg.norm(Fx) < tol:
            return X, i
            
        Jx = J(X)
        
        # Résolution du système linéaire J * dX = -F
        # C'est la seule "fonction magique" tolérée car coder le pivot de Gauss 
        # prendrait trop de place, mais c'est l'approche standard.
        dX = np.linalg.solve(Jx, -Fx)
        
        # Mise à jour
        X = X + dX
        
    print("Non convergence atteinte.")
    return X, max_iter

# Définition du système 2D
def F3(X):
    x, y = X, X[1]
    return np.array([np.exp(x) - y, x**2 + y**2 - 16])

def J3(X):
    x, y = X, X[1]
    return np.array([
        [np.exp(x), -1.0],
        [2.0 * x, 2.0 * y]
    ])

# Recherche de la première intersection (en partant de x>0, y>0)
X0_1 = [1.0, 3.0]
sol_1, iters_1 = newton_2d(F3, J3, X0_1)
print(f"Question 3 (Intersection 1) : Solution {sol_1} en {iters_1} itérations.")

# Recherche de la deuxième intersection (en partant de x<0, y>0)
X0_2 = [-3.0, 1.0]
sol_2, iters_2 = newton_2d(F3, J3, X0_2)
print(f"Question 3 (Intersection 2) : Solution {sol_2} en {iters_2} itérations.")

# Extension en dimension D 
def newton_nd(F, J, X0, tol=1e-6, max_iter=100):
    """
    Algorithme de Newton basique en dimension N.
    
    Arguments:
    - F : Fonction qui prend un vecteur X (taille N) et retourne un vecteur F(X) (taille N)
    - J : Fonction qui prend un vecteur X et retourne la matrice Jacobienne (taille N x N)
    - X0 : Point de départ (liste ou array numpy de taille N)
    
    Retourne la racine trouvée et le nombre d'itérations.
    """
    # Conversion du point de départ en un array numpy (vecteur de flottants)
    X = np.array(X0, dtype=float)
    
    for i in range(max_iter):
        Fx = F(X)
        
        # Critère d'arrêt : norme euclidienne du résidu
        if np.linalg.norm(Fx) < tol:
            return X, i
            
        Jx = J(X)
        
        try:
            # Étape cruciale : Résolution du système linéaire J * dX = -F
            # On N'UTILISE PAS np.linalg.inv(Jx) !
            dX = np.linalg.solve(Jx, -Fx)
        except np.linalg.LinAlgError:
            print(f"Erreur à l'itération {i} : La Jacobienne est singulière (non inversible).")
            return None, i
            
        # Mise à jour de Newton
        X = X + dX
        
    print("Non convergence atteinte : nombre maximum d'itérations dépassé.")
    return X, max_iter


# Application à Newton en D = 3

def F_test(X):
    x, y, z = X, X[1], X[2]
    return np.array([
        x**2 + y - 2,
        y**2 + z - 2,
        z**2 + x - 2
    ])

def J_test(X):
    x, y, z = X, X[1], X[2]
    # Matrice 3x3 avec les dérivées partielles
    return np.array([
        [2*x, 1.0, 0.0],
        [0.0, 2*y, 1.0],
        [1.0, 0.0, 2*z]
    ])

# Test de l'algorithme depuis un point de départ arbitraire
X0 = [2.0, 0.5, 3.0]
racine, iters = newton_nd(F_test, J_test, X0)

print(f"Point de départ : X0 = {X0}")
print(f"Racine trouvée  : X* = {racine}")
print(f"En {iters} itérations.")