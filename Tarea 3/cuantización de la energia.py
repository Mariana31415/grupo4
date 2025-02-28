import numpy as np
import matplotlib.pyplot as plt

# Parámetros
hbar = 1.0  # Constante de Planck reducida
m = 1.0     # Masa
omega = 1.0 # Frecuencia angular
L = 6.0     # Rango espacial: [-L, L]
N = 1000    # Número de puntos en la malla
x = np.linspace(-L, L, N)  # Malla espacial
h = x[1] - x[0]  # Tamaño del paso

# Construcción de la matriz Hamiltoniana
main_diag = (hbar**2 / (m * h**2)) + 0.5 * m * omega**2 * x**2  # Diagonal principal
off_diag = -0.5 * (hbar**2 / (m * h**2)) * np.ones(N-1)  # Diagonales adyacentes
H = np.diag(main_diag) + np.diag(off_diag, 1) + np.diag(off_diag, -1)

# Cálculo de valores y vectores propios
eigvals, eigvecs = np.linalg.eigh(H)  # Resuelve la matriz simétrica

# Seleccionar los primeros 5 niveles
n_levels = 5
energies = eigvals[:n_levels]
wavefunctions = eigvecs[:, :n_levels]

# Normalizar las funciones de onda
for i in range(n_levels):
    norm = np.sqrt(np.sum(wavefunctions[:, i]**2) * h)
    wavefunctions[:, i] /= norm

# Graficar
plt.figure(figsize=(10, 6))

# Pozo de potencial parabólico (línea punteada)
V = 0.5 * m * omega**2 * x**2
plt.plot(x, V, 'k--', label='Pozo de potencial')

# Factor de escala para reducir la amplitud de las funciones de onda
scale_factor = 0.5  # Puedes ajustar este valor según necesites

# Funciones de onda desplazadas por sus energías
for n in range(n_levels):
    psi = scale_factor * wavefunctions[:, n]  # Escalar la función de onda
    E_n = energies[n]
    plt.plot(x, psi + E_n, label=f'n = {n}, E = {E_n:.2f}')

# Configuración del gráfico
plt.xlabel('x')
plt.ylabel('Energía')
plt.grid(True)
plt.ylim(0, 6)  # Ajustado para mostrar hasta E_4 ≈ 4.5 más algo de margen
plt.xlim(-6, 6)
plt.show()