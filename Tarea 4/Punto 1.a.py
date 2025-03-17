import numpy as np
import matplotlib.pyplot as plt

# Definir la función g(x; n, alpha)
def g(x, n=10, alpha=0.8):
    """
    Calcula el valor de g(x; n, alpha) para un x dado.
    Parámetros:
        x: valor en el que se evalúa la función
        n: número de términos en la suma (default: 10)
        alpha: exponente en el denominador (default: 0.8)
    """
    total = 0
    for k in range(1, n + 1):
        total += np.exp(-((x - k) ** 2) / k) / (k ** alpha)
    return total

# Parámetros del algoritmo
n_samples = 550000  # Generar más muestras para incluir burn-in
burn_in = 50000     # Número de muestras iniciales a descartar
sigma = 1.0         # Desviación estándar de la distribución de propuesta
n_bins = 200        # Número de bins para el histograma

# Inicialización
x_current = 0.0     # Valor inicial arbitrario
samples = []        # Lista para almacenar las muestras

# Algoritmo de Metropolis-Hastings
np.random.seed(42)  # Para reproducibilidad
for i in range(n_samples):
    # Generar un valor propuesto desde una distribución normal
    x_proposed = np.random.normal(x_current, sigma)
    
    # Calcular la probabilidad de aceptación
    g_current = g(x_current)
    g_proposed = g(x_proposed)
    acceptance_prob = min(1, g_proposed / g_current)
    
    # Aceptar o rechazar el valor propuesto
    if np.random.rand() < acceptance_prob:
        x_current = x_proposed
    
    # Almacenar la muestra después del período de burn-in
    if i >= burn_in:
        samples.append(x_current)

# Convertir las muestras a un array de numpy
samples = np.array(samples)

# Crear el histograma
plt.figure(figsize=(10, 6))
plt.hist(samples, bins=n_bins, density=True, color='skyblue', edgecolor='black')
plt.title('Histograma de los datos generados con Metropolis-Hastings')
plt.xlabel('x')
plt.ylabel('Densidad')
plt.grid(True, alpha=0.3)

# Guardar el histograma en un archivo PDF
plt.savefig('1_a.pdf', format='pdf', bbox_inches='tight')
plt.show()

# Imprimir estadísticas básicas para verificación
print(f"Número total de muestras generadas (después de burn-in): {len(samples)}")
print(f"Media de las muestras: {np.mean(samples):.4f}")
print(f"Desviación estándar de las muestras: {np.std(samples):.4f}")