import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.integrate import simpson
from scipy.integrate import trapezoid
import numpy as np
import sympy as sym
import scipy
from datetime import datetime
from tabulate import tabulate
from scipy.signal import savgol_filter
from PIL import Image


print('punto 2')
print('2. a)')
df = pd.read_csv('H_field.csv')
x = np.array(df['t'], dtype=float)
y = np.array(df['H'], dtype=float)

f_fast = np.fft.rfft(y)
freq = np.fft.rfftfreq(len(x),x[1]-x[0])
#print(f'2.a) {f_fast = :.5f} ; f_general = ') 
#print(f"2.a){f_fast = :.5f}")
t = df["t"].values
idx_max = np.argmax(np.abs(f_fast))  # Índice de la frecuencia dominante
freq_dominante = freq[idx_max]  # Frecuencia de oscilación
fase_fast = freq_dominante*t%1
print(f"2.a){freq_dominante = :.5f}")

plt.figure(figsize=(10, 6))
plt.scatter(fase_fast,y, color='green', s=5, label="H como funcion de fase fast") 
plt.xlabel("H")
plt.ylabel("Fase fast")
plt.title("H como funcion de fase fast")
plt.grid(True)
plt.savefig("2.a.pdf")
plt.legend()
plt.show()


print('manchas solares punto 2')
print('2.b')

# Cargar los datos desde el archivo
archivo = open('list_aavso-arssn_daily.txt', 'r')
year, month, day, manchas_sol = [], [], [], []

for linea in archivo:
    if linea.strip():  # Ignorar líneas vacías
        partes = linea.split()
        if len(partes) == 4 and partes[0] != 'Year':  # Ignorar encabezado
            num_fltY, num_fltM, num_fltD, num_fltS = map(int, partes)
            if num_fltY < 2012:  # Filtrar datos hasta 2012
                year.append(num_fltY)
                month.append(num_fltM)
                day.append(num_fltD)
                manchas_sol.append(num_fltS)

archivo.close()

# Convertir a DataFrame
fechas = pd.to_datetime([f'{a}-{m}-{d}' for a, m, d in zip(year, month, day)])
df = pd.DataFrame({'fecha': fechas, 'manchas': manchas_sol}).set_index('fecha')

# Suavizado con Savitzky-Golay
manchas_suavizadas_sg = savgol_filter(df['manchas'], window_length=51, polyorder=3)

# Graficar datos originales y suavizados
plt.figure(figsize=(10, 5))

# Datos originales en un azul más vibrante
plt.plot(df.index, df['manchas'], label="Datos Originales", color="#1f77b4", linewidth=1.5)

# Predicción en un tono más suave pero llamativo
plt.plot(df.index, manchas_suavizadas_sg, label="Predicción FFT", color="#ff7f0e", linestyle="dashed", linewidth=2)

plt.xlabel("Días desde 2012", fontsize=12)
plt.ylabel("Número de manchas solares", fontsize=12)
plt.title("Predicción de Manchas Solares con FFT", fontsize=14, fontweight='bold')

plt.legend(fontsize=12)
plt.grid(True, linestyle="--", alpha=0.6)
plt.show()


# FFT de los datos suavizados
frecuencias = np.fft.rfftfreq(len(manchas_suavizadas_sg), d=1)
transformada = np.fft.rfft(manchas_suavizadas_sg)
densidad_espectral = np.abs(transformada) ** 2



# ---- SEGUNDA FFT CON ESCALA EN AÑOS ----
año = np.array(df.index.year, dtype=float)
datossolar = np.array(df['manchas'], dtype=float)
date = np.array(df.index)

# Calcular intervalo de muestreo
intervalo_tiempo = np.diff(año)  # Diferencias entre años consecutivos

# FFT de los datos solares
f_fast = np.fft.rfft(datossolar)

# Frecuencia en ciclos por año
freq = np.fft.rfftfreq(len(año), d=np.mean(intervalo_tiempo))

# Encontrar la frecuencia dominante (excluyendo la componente DC)
idx_max = np.argmax(np.abs(f_fast[1:])) + 1
freq_dominante = freq[idx_max]

# Calcular el período en años
P_solar = 1 / freq_dominante  

# Graficar la FFT en escala logarítmica
plt.figure(figsize=(10, 6))
plt.scatter(freq, np.abs(f_fast), color='green', s=5, label="FFT Magnitude")
plt.xscale('log')
plt.yscale('log')
plt.xlabel("Frecuencia (ciclos/año)")
plt.ylabel("|FFT|")
plt.title("FFT vs Frecuencia - Datos Solares")
plt.grid(True)
plt.legend()
plt.show()

print(f"Periodo solar estimado: {P_solar:.2f} años")



print('2.b.b')

t_dias = np.arange(len(datossolar))

X = np.fft.rfft(datossolar) / len(datossolar)  # Normalización correcta
f = np.fft.rfftfreq(len(datossolar), d=1) 
# Seleccionar los primeros 10 armónicos
n_armonicos = 10

X_n = X[:n_armonicos]
f_n = f[:n_armonicos]

def y(t, f, X):
    y_res = np.zeros_like(t, dtype=np.float64)
    for Xk, fk in zip(X, f):
        # Añadir la parte real (coseno) y la parte imaginaria (seno)
        term = np.abs(Xk) * np.cos(2 * np.pi * fk * t + np.angle(Xk))
        term += np.abs(Xk) * np.sin(2 * np.pi * fk * t + np.angle(Xk))
        
        # Añadir el complejo conjugado
        y_res += np.real(term) + np.conj(np.imag(term))  # Sumar conjugado de la parte imaginaria

    return y_res.real

# Fecha de inicio de los datos
fecha_inicio = datetime(2012, 1, 1)

# Fecha de predicción (10 de febrero de 2025)
fecha_prediccion = datetime(2025, 2, 10)

# Calcular días transcurridos
t_prediccion = (fecha_prediccion - fecha_inicio).days

# Calcular predicción
n_manchas_hoy = int(round(y(np.array([t_prediccion]), f_n, X_n)[0]))

# Imprimir el resultado en el formato requerido
print(f'2.b.b) {{n_manchas_hoy = {n_manchas_hoy:.2f}}}')


t_pred = np.arange(0, t_prediccion + 10000)  # Ampliar predicción hasta 2027
manchas_pred = np.round(y(t_pred, f_n, X_n)).astype(int)



print('punto 3')

# Definición del filtro gaussiano
def gaussian_filter(freq, alpha):
    return np.exp(- (freq * alpha) ** 2)

# Cargar los datos de manchas solares
data = np.loadtxt("OGLE-LMC-CEP-0001.dat")

# Extraer tiempo y magnitud de los datos OGLE
time = data[:, 0]
magnitude = data[:, 1]

# Datos de manchas solares (preprocesados)
t_dias = np.arange(len(datossolar))

# Transformada de Fourier de la señal
fft_original = np.fft.fft(datossolar)
frequencies = np.fft.fftfreq(len(datossolar), d=np.mean(intervalo_tiempo))

# Valores de α a probar
alpha_values = [0.1, 0.5, 1, 2]  # Desde un filtro suave hasta uno más agresivo

# Crear subplots
fig, axes = plt.subplots(len(alpha_values), 2, figsize=(12, 10))
fig.subplots_adjust(hspace=0.4)

# Aplicar el filtro y graficar para cada valor de α
for i, alpha in enumerate(alpha_values):
    # Construcción del filtro y señal filtrada
    filter_gaussian = gaussian_filter(frequencies, alpha)
    fft_filtered = fft_original * filter_gaussian
    signal_filtered = np.fft.ifft(fft_filtered).real

    # Graficar señal original y filtrada en el dominio del tiempo
    axes[i, 0].plot(t_dias, datossolar, label="Señal original", color="blue")
    axes[i, 0].plot(t_dias, signal_filtered, label=f"Señal filtrada (α={alpha})", linestyle="dashed", color="red")
    axes[i, 0].set_title("Señal en el dominio del tiempo")
    axes[i, 0].legend()
    
    # Graficar transformada de Fourier original y filtrada
    axes[i, 1].plot(frequencies, np.abs(fft_original), label="FFT original", color="green")
    axes[i, 1].plot(frequencies, np.abs(fft_filtered), label="FFT filtrada", linestyle="dashed", color="purple")
    axes[i, 1].set_title("Transformada de Fourier")
    axes[i, 1].legend()
    
    # Añadir texto con el valor de α
    axes[i, 0].text(0.05, 0.85, f"α = {alpha}", transform=axes[i, 0].transAxes, fontsize=12)

# Guardar la figura como "3.1.pdf"
plt.savefig("3.1.pdf", format="pdf")

# Mostrar la figura
plt.show()



print('3.b')



def remove_periodic_noise(image_path, output_path, high_freq_cutoff=15, line_freq_range=5):
    """
    Elimina ruido periódico de una imagen usando filtrado en el dominio de Fourier.

    Parámetros:
    - image_path: Ruta de la imagen original.
    - output_path: Ruta para guardar la imagen filtrada.
    - high_freq_cutoff: Tamaño del filtro para eliminar altas frecuencias.
    - line_freq_range: Rango de frecuencias a eliminar (basado en el patrón de la persiana).
    """
    # Cargar la imagen en escala de grises usando PIL
    img = Image.open(image_path)
    img_array = np.array(img)
    N, M = img_array.shape  # Dimensiones de la imagen
    
    # Aplicar Transformada de Fourier
    FFT = np.fft.fft2(img_array)
    FFT = np.fft.fftshift(FFT)  # Centramos la FFT

    # Suprimir las frecuencias asociadas con el ruido periódico (patrón de la persiana)
    # Este paso es un filtro que elimina ciertas frecuencias en la imagen
    for i in range(N):
        for j in range(M):
            # Si la frecuencia está dentro del rango del patrón de la persiana, se anula
            if abs(i - N//2) % line_freq_range == 0 or abs(j - M//2) % line_freq_range == 0:
                FFT[i, j] = 0  # Eliminar estas frecuencias

    # Transformada inversa para recuperar la imagen sin ruido
    FFT = np.fft.ifftshift(FFT)  # Deshacer el desplazamiento
    img_filtered = np.fft.ifft2(FFT).real  # Obtener imagen filtrada

    # Guardar la imagen procesada
    Image.fromarray(np.uint8(img_filtered)).save(output_path)

    # Mostrar la imagen filtrada
    plt.figure(figsize=(6,6))
    plt.imshow(img_filtered, cmap="gray")
    plt.title(f"Imagen Filtrada: {output_path}")
    plt.axis("off")
    plt.show()

# Aplicar el filtrado a ambas imágenes
remove_periodic_noise("catto.png", "3.b.a.png", high_freq_cutoff=20, line_freq_range=12)
remove_periodic_noise("Noisy_Smithsonian_Castle.jpg", "3.b.b.png", high_freq_cutoff=50, line_freq_range=55)