import re
from collections import defaultdict
import numpy as np
import random
import multiprocessing
from numba import jit
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import quad


#punto 1
print('punto 1')
#punto 1
def g_x(x, n, alpha):
    """Función g(x; n, α) no normalizada."""
    return sum(np.exp(-k*(x - k)**2) * k**(-alpha) for k in range(1, int(n)+1))

def metropolis_hastings(alpha, N, ancho_propuesta, n):
    """Algoritmo de Metropolis-Hastings para muestrear de la distribución g(x)."""
    samples = []
    x = np.random.uniform(0, 10)
    
    for _ in range(N):
        x_propuesta = np.random.normal(x, ancho_propuesta)
        aceptacion_ratio = min(1, g_x(x_propuesta, n, alpha) / g_x(x, n, alpha))
        if np.random.rand() < aceptacion_ratio:
            x = x_propuesta
        samples.append(x)
    
    return np.array(samples)

def f_x(x):
    """Función f(x) con integral conocida (exp(-x^2))."""
    return np.exp(-x**2)

def calcular_A(samples, n, alpha):
    """Calcula la integral A usando muestreo Monte Carlo."""
    g_values = np.array([g_x(x, n, alpha) for x in samples])
    f_values = np.array([f_x(x) for x in samples])
    estimaciones = f_values / g_values
    A = np.mean(estimaciones)
    incertidumbre = np.std(estimaciones) / np.sqrt(len(samples))
    return A, incertidumbre

# Parámetros
n_samples = 500000
n = 10
alpha = 4/5
ancho_propuesta = 2.0  # Ajuste del ancho de propuesta

# Obtener datos con Metropolis-Hastings
data = metropolis_hastings(alpha, N=n_samples, ancho_propuesta=ancho_propuesta, n=n)

# Calcular A y su incertidumbre
A_estimado, error_A = calcular_A(data, n, alpha)
print(f"1.b) A estimado = {A_estimado} ± {error_A}")

'''
# Crear histograma diferenciado
plt.figure(figsize=(10, 6))
plt.hist(data, bins=200, density=True, color='darkorange', alpha=0.6, edgecolor='black', label='Muestras de g(x)')
plt.xlabel("x")
plt.ylabel("Densidad")
plt.title("Histograma de muestras de g(x; n, α)")
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
#plt.savefig("1_a.pdf")
'''








#punto 2

print('Punto 2')

D1 = 50  # cm
D2 = 50  # cm
wavelength = 670e-7  # cm
A = 0.04  # cm
a = 0.01  # cm
d = 0.1  # cm
N = 100000

def monte_carlo_intensity(z):
    x_samples = np.random.uniform(-A/2, A/2, N)
    y = np.random.uniform(d/2, a+d/2, N)
    lista = [-1, 1]
    y_samples = y * np.random.choice(lista, len(y))

    phase1 = 2 * np.pi * (D1 + D2) / wavelength
    phase2 = np.pi / (wavelength * D1)
    phase3 = np.pi / (wavelength * D2)

    integrand = np.exp(1j * phase1) * np.exp(1j * phase2 * (x_samples - y_samples)**2) * np.exp(1j * phase3 * (z - y_samples)**2)
    integral_value = np.abs(np.mean(integrand))**2
    return integral_value

z_values = np.linspace(-0.4, 0.4, 500)
quantum_intensity = np.array([monte_carlo_intensity(z) for z in z_values])

theta = np.arctan(z_values / D2)
classic_intensity = np.cos(np.pi * d * np.sin(theta) / wavelength)**2 * np.sinc(a * np.sin(theta) / wavelength)**2

quantum_intensity /= np.max(quantum_intensity)
classic_intensity /= np.max(classic_intensity)

'''
plt.figure(figsize=(10, 6))
plt.plot(z_values, quantum_intensity, label='Modelo Cuántico (Monte Carlo)', color='royalblue', linewidth=2, alpha=0.9)
plt.plot(z_values, classic_intensity, label='Modelo Clásico', color='crimson', linestyle='dashdot', linewidth=2, alpha=0.8)

plt.xlabel(r'$z$ (cm)', fontsize=14)
plt.ylabel('Intensidad Normalizada', fontsize=14)
plt.title('Comparación de Intensidad Cuántica y Clásica', fontsize=16)

plt.legend(fontsize=12, loc='upper right')
plt.grid(True, linestyle='dotted', linewidth=0.8, alpha=0.6)
#plt.savefig('2.pdf')
'''

print('En la mecánica clásica, la interferencia depende de las longitudes de onda y las posiciones relativas de las ranuras.\n'
'Mientras que en cuántica, el fenómeno de interferencia se debe a la superposición de funciones de onda generada por los fotones. \n'
'Se generan fotones aleatoriamente dentro de las dimensiones de la rendija y se calcula la fase de la onda en función de su trayectoria.\n'
'Donde luego se promedian las contribuciones de cada una para obtener la intensidad. Tenemos que cuando el número de fotónes aumenta, se acerca más al caso clásico como se\n'
've gráficamente.')
























#Punto 3


N = 150        # Tamaño de la malla
J = 0.2        # Constante de interacción
beta = 10      # Inverso de la temperatura (1/kT)

frames = 500
steps_per_frame = 400

state = np.random.choice([-1, 1], size=(N, N))

def metropolis_step(state):
    """
    Realiza una iteración del algoritmo de Metropolis:
    """
    i = np.random.randint(0, N)
    j = np.random.randint(0, N)
    
    sum_neighbors = (
        state[(i+1) % N, j] +
        state[(i-1) % N, j] +
        state[i, (j+1) % N] +
        state[i, (j-1) % N]
    )
    

    deltaE = 2 * J * state[i, j] * sum_neighbors
    
    if deltaE <= 0 or np.random.rand() < np.exp(-beta * deltaE):
        state[i, j] *= -1  # Se invierte el espín
    return state

def update(frame):
    #global state
    #for _ in range(steps_per_frame):
    #    metropolis_step(state)
    #im.set_data(state)
    #return [im]
    return 
# Crear la animación
#anim = FuncAnimation(fig, update, frames=frames, interval=50, blit=True)

























#Cuarto punto

print('Punto 4')
# Función para limpiar el texto
def clean_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        s = file.read()
    s = s.replace("\r\n", "\n").replace("\n\n", "#").replace("\n", " ").replace("#", "\n\n")
    s = re.sub(r'[^a-zA-Z0-9\s]', '', s)  # Elimina todo excepto letras, números y espacios
    s = re.sub(r'\s+', ' ', s)  # Normaliza los espacios
    s = s.lower().strip()
    return s

# 4.2) Entrenamiento y predicción
@jit(nopython=True)
def build_ngram_frequencies(text, n):
    ngram_list = []
    next_char_list = []
    text_len = len(text)
    for i in range(text_len - n):
        ngram = text[i:i+n]
        next_char = text[i+n]
        ngram_list.append(ngram)
        next_char_list.append(next_char)
    return ngram_list, next_char_list

def train_and_predict(text, n):
    if len(text) < n + 1:
        raise ValueError(f"El texto es demasiado corto para n={n}.")
    
    ngram_list, next_char_list = build_ngram_frequencies(text, n)
    ngram_frequencies = defaultdict(lambda: defaultdict(int))
    for i in range(len(ngram_list)):
        ngram_frequencies[ngram_list[i]][next_char_list[i]] += 1
    
    for ngram, next_chars in ngram_frequencies.items():
        total = sum(next_chars.values())
        for next_char in next_chars:
            next_chars[next_char] /= total
    
    initial_ngram = np.random.choice(list(ngram_frequencies.keys()))
    return ngram_frequencies, initial_ngram




#Generar el texto con m= 1500

def generate_text(ngram_frequencies, initial_ngram, m=1500, line_length=80):
    generated = initial_ngram
    current_ngram = initial_ngram
    
    for _ in range(m):
        next_chars = ngram_frequencies.get(current_ngram, {})
        if not next_chars:
            break
        next_char = np.random.choice(list(next_chars.keys()), p=list(next_chars.values()))
        generated += next_char
        current_ngram = generated[-len(initial_ngram):]
    
    formatted_text = '\n'.join([generated[i:i+line_length] for i in range(0, len(generated), line_length)])
    return formatted_text




# Cargar y limpiar el texto y graficar






file_path = 'Prideandprejudice.txt'
cleaned_text = clean_text(file_path)
if not cleaned_text:
    
    #raise ValueError("El texto limpio está vacío. Verifica el archivo de entrada.")


    with open('words_alpha.txt', 'r') as f:
        valid_words = set(word.strip() for word in f.readlines())

#4.c Análisis



with open('words_alpha.txt', 'r') as f:
    valid_words = set(word.strip() for word in f.readlines())

results = {}
for n in range(1, 8):
    ngram_frequencies, initial_ngram = train_and_predict(cleaned_text, n)
    generated_text = generate_text(ngram_frequencies, initial_ngram, m=1500)

    # Contar palabras válidas
    words = generated_text.split()
    valid_count = sum(1 for word in words if word in valid_words)
    valid_percentage = (valid_count / len(words)) * 100 if words else 0

    results[n] = valid_percentage

for n, percentage in results.items():
    print(f"n={n}: Porcentaje de palabras válidas = {percentage:.2f}%")

print('4.c Analisis: Para n=1, las palabras que genera el texto no tienen sentido. Al tener n=5 o 6, los n-gramas incluyen secuencias mas coherentes y con n>=7 el texto tiene sentido\n'
'Sin embargo, se evidenció que al tener un n entre 3 y 5, es cuando el porcentaje de palabras válidas tiene un incremento. Cuando llega a n = 7 solo aumenta un poco más y al\n'
'aumentar demasiado los n obtuvimos que se queda estable el porcentaje.')






results = {}
n_values = []
percentages = []
for n in range(1, 8):
    ngram_frequencies, initial_ngram = train_and_predict(cleaned_text, n)
    generated_text = generate_text(ngram_frequencies, initial_ngram, m=1500)
    words = generated_text.split()
    valid_count = sum(1 for word in words if word in valid_words)
    valid_percentage = (valid_count / len(words)) * 100 if words else 0
    results[n] = valid_percentage
    n_values.append(n)
    percentages.append(valid_percentage)
    with open(f'gen_text_n{n}.txt', 'w', encoding='utf-8') as f:
        f.write(f'N-grama {n}:\n\n{generated_text}\n\n{"-"*50}\n\n')
    #print(f'Texto generado para n={n} guardado en gen_text_n{n}.txt')



# Graficar los resultados
#plt.figure(figsize=(10, 5))
#plt.plot(n_values, percentages, marker='o', linestyle='-')
#plt.xlabel('n (tamaño del n-grama)')
#plt.ylabel('Porcentaje de palabras válidas')
#plt.title('Porcentaje de palabras válidas en función de n')
#plt.grid(True)
#plt.savefig('ngrams_valid_words.pdf')