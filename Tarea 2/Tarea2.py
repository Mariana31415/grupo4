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

file_path = "list_aavso-arssn_daily.txt"

with open(file_path, "r") as file:
  lines = file.readlines()

data = [line.strip().split() for line in lines]

# Filtrar filas incorrectas (asegurarse de que solo queden las de datos)
data = [row for row in data if len(row) == 4]  # Solo filas con 4 elementos

# Crear el DataFrame con las columnas correctas
dfsolar = pd.DataFrame(data, columns=["year", "month", "day", "SSN"])

# Convertir a tipos numéricos, ignorando errores
dfsolar = dfsolar.apply(pd.to_numeric, errors='coerce')

# Eliminar filas con valores NaN (valores no convertibles)
dfsolar = dfsolar.dropna()

# Convertir a enteros después de eliminar NaN
dfsolar = dfsolar.astype({"year": int, "month": int, "day": int, "SSN": int})

# Crear columna de fecha
dfsolar["Date"] = pd.to_datetime(dfsolar[["year", "month", "day"]])

dfsolar = dfsolar[dfsolar["year"] <= 2011]




print('2.b.a')

año = np.array(dfsolar['year'], dtype=float)
datossolar = np.array(dfsolar['SSN'], dtype=float)
date = np.array(dfsolar['Date'])

# Calcula la diferencia de años para obtener el intervalo de muestreo
intervalo_tiempo = np.diff(año)  # Diferencias entre años consecutivos

# Realiza la FFT de los datos solares
f_fast = np.fft.rfft(datossolar)

# Calcula las frecuencias correspondientes a la FFT
freq = np.fft.rfftfreq(len(año), d=np.mean(intervalo_tiempo))  # Se usa el intervalo medio

# Buscar el pico dominante en la FFT (excluyendo la componente DC en freq[0])
idx_max = np.argmax(np.abs(f_fast[1:])) + 1  # Evitamos el índice 0 (componente de tendencia)
freq_dominante = freq[idx_max]  # Frecuencia dominante en ciclos por año

# Calculamos el período en años
P_solar = 1 / freq_dominante  

# Graficar la FFT en escala logarítmica
plt.figure(figsize=(10, 6))
plt.scatter(freq, np.abs(f_fast), color='green', s=5, label="FFT Magnitude")
plt.xscale('log')  # Escala logarítmica en el eje X
plt.yscale('log')  # Escala logarítmica en el eje Y
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
n_manchas_hoy = y(np.array([t_prediccion]), f_n, X_n)[0]

# Imprimir el resultado en el formato requerido
print(f'2.b.b) {{n_manchas_hoy = {n_manchas_hoy:.2f}}}')

t_pred = np.arange(0, t_prediccion + 10000)  # Ampliar predicción hasta 2027
manchas_pred = y(t_pred, f_n, X_n)

plt.figure(figsize=(10,5))
plt.plot(date, datossolar, label="Datos Originales", color="blue")
plt.plot(t_pred, manchas_pred, label="Predicción FFT", color="red", linestyle="dashed")
plt.xlabel("Días desde 2012")
plt.ylabel("Número de manchas solares")
plt.title("Predicción de Manchas Solares con FFT")
plt.savefig("2.b.pdf")
plt.legend()
plt.grid()
plt.show()
