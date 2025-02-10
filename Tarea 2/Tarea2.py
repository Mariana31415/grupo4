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
from numpy.typing import NDArray
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks, peak_widths



# Función para generar datos de prueba
def datos_prueba(t_max: float, dt: float, amplitudes: np.ndarray[float],
                 frecuencias: np.ndarray[float], ruido: float = 0.0) -> tuple[np.ndarray[float], np.ndarray[float]]:
    ts = np.arange(0., t_max, dt)
    ys = np.zeros_like(ts, dtype=float)
    for A, f in zip(amplitudes, frecuencias):
        ys += A * np.sin(2 * np.pi * f * ts)
    ys += np.random.normal(loc=0, size=len(ys), scale=ruido) if ruido else 0
    return ts, ys

# Función para calcular la Transformada de Fourier
def Fourier(t: np.ndarray[float], y: np.ndarray[float], f: float) -> complex:
    N = len(t)
    return np.sum(y * np.exp(-2j * np.pi * t * f)) / N

# Generación de datos de prueba
t_max = 10.0
dt = 0.01
amplitudes = np.array([1.0, 2.0, 1.5])
frecuencias = np.array([1.0, 2.0, 3.0])

# Datos con y sin ruido
ts_no_ruido, ys_no_ruido = datos_prueba(t_max, dt, amplitudes, frecuencias, ruido=0.0)
ts_con_ruido, ys_con_ruido = datos_prueba(t_max, dt, amplitudes, frecuencias, ruido=0.5)

# Cálculo de la transformada de Fourier
frequencies = fftfreq(len(ts_no_ruido), dt)
transformada_no_ruido = np.abs(fft(ys_no_ruido))
transformada_con_ruido = np.abs(fft(ys_con_ruido))

# Graficación
plt.figure(figsize=(10, 6))
plt.plot(frequencies[:len(frequencies)//2], transformada_no_ruido[:len(frequencies)//2], label='Sin ruido')
plt.plot(frequencies[:len(frequencies)//2], transformada_con_ruido[:len(frequencies)//2], label='Con ruido')
plt.xlabel('Frecuencia')
plt.ylabel('Valor absoluto de la transformada')
plt.title('Transformada de Fourier de señales con y sin ruido')
plt.legend()
plt.grid(True)
plt.savefig('1.a.pdf')
plt.close()

# Respuesta a la pregunta sobre el efecto del ruido
respuesta_1a = "1.a) Atenuación de picos, ruido similar amplitudes."
print(respuesta_1a)



# Parámetros iniciales
dt = 0.01  # Intervalo de muestreo
frecuencia_conocida = 1.0  # Frecuencia en Hz
t_max_range = np.logspace(1, 2.5, 15)  # Valores de t_max logarítmicos
fwhm_values = []

for tm in t_max_range:
    # Generar datos
    t = np.arange(0, tm, dt)
    y = np.sin(2 * np.pi * frecuencia_conocida * t)

    # Calcular FFT
    Y = np.abs(fft(y))
    freqs = np.fft.fftfreq(len(t), d=dt)
    freqs = freqs[:len(freqs) // 2]  # Tomar solo la parte positiva
    Y = Y[:len(Y) // 2]

    # Detección de picos en la FFT
    peaks, _ = find_peaks(Y)
    if len(peaks) > 0:
        results_half = peak_widths(Y, peaks, rel_height=0.5)
        fwhm = results_half[0][0] * (freqs[1] - freqs[0])
        fwhm_values.append(fwhm)
    else:
        fwhm_values.append(np.nan)  # En caso de no detectar picos

# Graficar FWHM vs. t_max
plt.figure(figsize=(10, 6))
plt.plot(t_max_range, fwhm_values, marker='o', linestyle='-', label='FWHM vs. $t_{max}$')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('$t_{max}$ (s)')
plt.ylabel('FWHM')
plt.title('Ancho a Media Altura vs. Intervalo de Tiempo')
plt.legend()
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.savefig('1.b.pdf')
plt.close()


file_path = "OGLE-LMC-CEP-0001.dat"
# Leer las primeras líneas del archivo para ver su estructura
with open(file_path, "r") as file:
    lines = file.readlines()

# Mostrar las primeras líneas del archivo
lines[:10]

import numpy as np

# Cargar los datos desde el archivo
data = np.loadtxt(file_path)

# Separar las columnas
t = data[:, 0]  # Tiempo en días
y = data[:, 1]  # Intensidad
sigma_y = data[:, 2]  # Incertidumbre

# Ver los primeros valores
t[:5], y[:5], sigma_y[:5]
import matplotlib.pyplot as plt

# Calcular diferencias de tiempo
dt = np.diff(t)

# Calcular una estimación de la frecuencia de Nyquist
nyquist_freq = 1 / (2 * np.median(dt))

# Imprimir el resultado
print(f"1.c) f Nyquist: {nyquist_freq:.4f} 1/día")
from scipy.fft import fft, fftfreq

# Quitar el promedio de la señal para eliminar el pico en f = 0
y_zero_mean = y - np.mean(y)

# Definir el rango de frecuencias de interés (según la recomendación del profesor)
freqs = np.linspace(0, 8, 1000)  # 1000 puntos entre 0 y 8 d⁻¹

# Implementar la transformada de Fourier manualmente
def Fourier(t: np.ndarray, y: np.ndarray, f: np.ndarray) -> np.ndarray:
    return np.array([np.sum(y * np.exp(-2j * np.pi * f_i * t)) for f_i in f])

# Calcular la transformada de Fourier
fourier_transform = Fourier(t, y_zero_mean, freqs)

# Obtener la frecuencia con el pico más alto (excluyendo f=0)
f_true = freqs[np.argmax(np.abs(fourier_transform[1:])) + 1]

# Imprimir el resultado
print(f"1.c) f true: {f_true:.4f} 1/día")

# Calcular la fase
phi = np.mod(f_true * t, 1)

# Graficar la intensidad en función de la fase
plt.figure(figsize=(8, 5))
plt.scatter(phi, y, s=10, alpha=0.5, color='b', label="Datos experimentales")
plt.xlabel("Fase φ (mod(f_true * t, 1))")
plt.ylabel("Intensidad (magnitud)")
plt.title("Diagrama de fase de la señal")
plt.legend()
plt.grid(True)

# Guardar la gráfica como 1.c.pdf
plt.savefig("1.c.pdf")
plt.close()







# Cargar datos
df = pd.read_csv('H_field.csv')
x = np.array(df['t'], dtype=float)  # Tiempo
y = np.array(df['H'], dtype=float)  # Intensidad H

# FFT y frecuencias
F = np.fft.fft(y)  # ← Usar fft en vez de rfft para evitar reducción de tamaño
freq = np.fft.fftfreq(len(x), np.median(np.diff(x)))  # Intervalo de tiempo no uniforme

# Encontrar la frecuencia dominante
idx_max = np.argmax(np.abs(F))
freq_dominante = abs(freq[idx_max])  # Asegurar que sea positiva

# Calcular fases
fase_fast = (freq_dominante * x) % 1  # ← Asegurar que tenga el mismo tamaño que `y`
fase_gen = (freq_dominante * x) % 1   # ← Similar al caso anterior

# Imprimir resultado
print(f"2.a) freq_dominante = {freq_dominante:.5f}; f_general = {freq_dominante:.5f}")

# Graficar y guardar sin mostrar
plt.figure(figsize=(10, 6))
plt.scatter(fase_fast, y, color='green', s=5, label="H como función de fase fast")
plt.scatter(fase_gen, y, color='red', s=5, label="H como función de fase general")
plt.xlabel("Fase")
plt.ylabel("H")
plt.title("H como función de las fases")
plt.grid(True)
plt.legend()
plt.savefig("2.a.pdf")
plt.close()  # ← Cierra la figura sin error




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
print(f"Periodo solar estimado: {P_solar:.2f} años")





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
plt.close(fig)










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
    plt.title(f"Imagen Filtrada: {output_path}")
    plt.axis("off")
    plt.imshow(img_filtered, cmap="gray")
 

# Aplicar el filtrado a ambas imágenes
imagen1 = remove_periodic_noise("catto.png", "3.b.a.png", high_freq_cutoff=20, line_freq_range=12)
imagen2 = remove_periodic_noise("Noisy_Smithsonian_Castle.jpg", "3.b.b.png", high_freq_cutoff=50, line_freq_range=55)
plt.close(imagen1)
plt.close(imagen2)