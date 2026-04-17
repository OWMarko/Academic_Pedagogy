import numpy as np
import matplotlib.pyplot as plt

# Une forme quadratique ax^2 + bxy + cy^2 a pour matrice symétrique :
# [[a, b/2], [b/2, c]]

A1 = np.array([[1.0, 0.5], 
               [0.5, 2.0]])

A2 = np.array([[1.0, 0.5], 
               [0.5, -2.0]])

A3 = np.array([[6.0, 1.5], 
               [1.5, 2.0]])

A4 = np.array([[-2.0, 0.5], 
               [0.5, -3.5]])

matrices = {"f1": A1, "f2": A2, "f3": A3, "f4": A4}

# Calcul et affichage des valeurs propres
print("--- Matrices et Valeurs Propres ---")
for nom, A in matrices.items():
    valeurs_propres = np.linalg.eigvals(A)
    print(f"\nMatrice {nom} :\n{A}")
    print(f"Valeurs propres : {valeurs_propres.round(3)}")
    
    # Analyse des caractéristiques
    if np.all(valeurs_propres > 0):
        nature = "Définie positive (minimum global en 0, forme de cuvette/ellipses)"
    elif np.all(valeurs_propres < 0):
        nature = "Définie négative (maximum global en 0, forme de colline)"
    else:
        nature = "Indéfinie (point selle en 0, forme d'hyperboles)"
    print(f"-> {nature}")

# Tracé des lignes de niveau ---
x = np.linspace(-5, 5, 400)
y = np.linspace(-5, 5, 400)
X, Y = np.meshgrid(x, y)

fig, axs = plt.subplots(2, 2, figsize=(12, 10))
axs = axs.ravel()

for i, (nom, A) in enumerate(matrices.items()):
    # Calcul vectorisé de la forme quadratique: Z = a*X^2 + 2b*X*Y + c*Y^2
    Z = A*X**2 + (A[1] + A[1])*X*Y + A[1,1]*Y**2
    
    ax = axs[i]
    # Tracé des contours
    contour = ax.contour(X, Y, Z, levels=20, cmap='coolwarm')
    ax.clabel(contour, inline=True, fontsize=8)
    
    # Mise en forme du graphique
    vp = np.linalg.eigvals(A)
    ax.set_title(f"{nom} | VP1 $\\approx$ {vp:.2f}, VP2 $\\approx$ {vp[1]:.2f}")
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.5)
    ax.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.set_xlabel('x')
    ax.set_ylabel('y')

plt.tight_layout()
plt.show()