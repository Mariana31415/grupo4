import re
from collections import defaultdict
import numpy as np
import random
import multiprocessing
from numba import jit
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


#Punto 3
print('Punto 3')

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
    print(f'Texto generado para n={n} guardado en gen_text_n{n}.txt')



# Graficar los resultados
#plt.figure(figsize=(10, 5))
#plt.plot(n_values, percentages, marker='o', linestyle='-')
#plt.xlabel('n (tamaño del n-grama)')
#plt.ylabel('Porcentaje de palabras válidas')
#plt.title('Porcentaje de palabras válidas en función de n')
#plt.grid(True)
#plt.savefig('ngrams_valid_words.pdf')