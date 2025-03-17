import numpy as np
import matplotlib.pyplot as plt

# Definir la función g(x; n, alpha)
def g(x, n=10, alpha=4/5):
    k = np.arange(1, n + 1)
    terms = np.exp(-(x - k)**2 / k) / k**alpha
    return np.sum(terms)

# Parámetros
num_samples = 500000  # Número de muestras
burn_in = 10000      # Período de burn-in
sigma = 1.0          # Desviación estándar de la propuesta
x_init = 5.0         # Valor inicial

# Algoritmo de Metropolis-Hastings
np.random.seed(42)  # Para reproducibilidad
x_current = x_init
samples = []

for i in range(num_samples + burn_in):
    x_proposed = np.random.normal(x_current, sigma)
    accept_prob = min(1, g(x_proposed) / g(x_current))
    if np.random.rand() < accept_prob:
        x_current = x_proposed
    if i >= burn_in:
        samples.append(x_current)

# Crear y guardar el histograma
plt.hist(samples, bins=200, density=True, color='blue', alpha=0.7)
plt.title("Histograma de muestras de g(x; n=10, α=4/5)")
plt.xlabel("x")
plt.ylabel("Densidad")
plt.savefig("1_a.pdf")
plt.close()

print("Histograma guardado como '1_a.pdf'")