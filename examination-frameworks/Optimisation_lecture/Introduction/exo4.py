import numpy as np
import matplotlib.pyplot as plt

def g(x, y):
    """Fonction g(x,y) = x^2 + y^2"""
    return x**2 + y**2

def grad_g(x, y):
    """Gradient analytique de g : [2x, 2y]"""
    return np.array([2*x, 2*y])

# Grille de calcul
x = np.linspace(-2.5, 2.5, 100)
y = np.linspace(-2.5, 2.5, 100)
X, Y = np.meshgrid(x, y)
Z = g(X, Y)

# Création de la figure globale
fig = plt.figure(figsize=(14, 6))

# Graphique
ax1 = fig.add_subplot(121, projection='3d')
surf = ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
ax1.set_title("Surface 3D de $g(x,y) = x^2 + y^2$")
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('g(x,y)')
ax2 = fig.add_subplot(122)

niveaux = [1, 2, 3]
contour = ax2.contour(X, Y, Z, levels=niveaux, colors=['blue', 'orange', 'green'])
ax2.clabel(contour, inline=True, fontsize=12, fmt="g=%1.0f")

# On sélectionne quelques points sur ces cercles (rayons = sqrt(1), sqrt(2), sqrt(3))
rayons = np.sqrt(niveaux)
angles = np.linspace(0, 2*np.pi, 8, endpoint=False) # 8 points répartis sur chaque cercle

# Tracé des champs de vecteurs
for r in rayons:
    for theta in angles:
        px = r * np.cos(theta)
        py = r * np.sin(theta)
        
        gx, gy = grad_g(px, py)
        
        # Gradient (en rouge) : il pointe vers l'extérieur, perpendiculaire au cercle
        ax2.quiver(px, py, gx, gy, color='red', angles='xy', scale_units='xy', scale=8, width=0.005)
        
        # Opposé du gradient (en noir) : la direction de la plus forte pente vers le minimum
        ax2.quiver(px, py, -gx, -gy, color='black', angles='xy', scale_units='xy', scale=8, width=0.005, alpha=0.5)

# Marquage du minimum global
ax2.plot(0, 0, 'k*', markersize=12, label='Minimum global (0,0)')

# Paramètres esthétiques importants
ax2.set_aspect('equal') # INDISPENSABLE pour que les cercles soient bien ronds et l'orthogonalité visible
ax2.set_title("Lignes de niveau et vecteurs gradients")
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.grid(True, linestyle='--', alpha=0.6)

# Astuce pour la légende avec quiver
ax2.quiverkey(ax2.quiver(0,0,0,0, color='red'), 0.85, 0.95, 1, 'Gradient $\\nabla g$', labelpos='E')
ax2.quiverkey(ax2.quiver(0,0,0,0, color='black', alpha=0.5), 0.85, 0.90, 1, 'Opposé $-\\nabla g$', labelpos='E')

plt.tight_layout()
plt.show()