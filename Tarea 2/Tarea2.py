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

