import numpy as np
import matplotlib.pyplot as plt

# Génération de données d'exemple (n points avec un peu de bruit)
np.random.seed(42)
n = 20
x_data = np.linspace(0, 10, n)
# Générons des données suivant approximativement y = 2x + 5
y_data = 2 * x_data + 5 + np.random.normal(0, 2, n)

def solve_least_squares_qr(A, y):
    """Résout Az = y au sens des moindres carrés via QR."""
    Q, R = np.linalg.qr(A) # Décomposition QR de A
    # On résout Rz = Q^T * y
    z = np.linalg.solve(R, np.dot(Q.T, y))
    return z

# 1) Cas de la droite : y = ax + b

# Construction de la matrice A : colonnes [x, 1]
A_affine = np.vstack([x_data, np.ones(len(x_data))]).T
a, b = solve_least_squares_qr(A_affine, y_data)

# 2) Cas quadratique : y = ax^2 + bx + c

# Construction de la matrice A : colonnes [x^2, x, 1]
A_quad = np.vstack([x_data**2, x_data, np.ones(len(x_data))]).T
aq, bq, cq = solve_least_squares_qr(A_quad, y_data)

# --- Affichage graphique ---
plt.figure(figsize=(10, 6))
plt.scatter(x_data, y_data, color='black', label='Points de données $(x_i, y_i)$')

# Tracé de la droite
x_plot = np.linspace(0, 10, 100)
plt.plot(x_plot, a*x_plot + b, 'r--', label=f'Affine : $y = {a:.2f}x + {b:.2f}$')

# Tracé de la courbe quadratique
plt.plot(x_plot, aq*x_plot**2 + bq*x_plot + cq, 'g-', label=f'Quadratique : $y = {aq:.2f}x^2 + {bq:.2f}x + {cq:.2f}$')

plt.title("Ajustement par moindres carrés (Décomposition QR)")
plt.legend()
plt.grid(True)
plt.show()

print(f"Coefficients affines : a={a:.4f}, b={b:.4f}")
print(f"Coefficients quadratiques : a={aq:.4f}, b={bq:.4f}, c={cq:.4f}")