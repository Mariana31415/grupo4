import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.integrate import simpson
import numpy as np
import sympy as sym

#punto 1

#1.a Limpiar los datos
print('1.a limpiar los datos')
df = pd.read_csv('Rhodium.csv')
x = np.array(df['Wavelength (pm)'], dtype=float)
y = np.array(df['Intensity (mJy)'], dtype=float)

# Función para obtener los parámetros del modelo de regresión polinomial
def GetFit(x, y, n=1):
    l = x.shape[0]
    b = y

    A = np.ones((l, n+1))

    for i in range(1, n+1):
        A[:,i] = x**i

    AT = np.dot(A.T, A)
    bT = np.dot(A.T, b)

    xsol = np.linalg.solve(AT, bT)

    return xsol

# Función para obtener el modelo ajustado (predicciones)
def GetModel(x, p):
    y = 0.
    for i in range(len(p)):
        y += p[i] * x**i
    return y

n = 40
param = GetFit(x, y, n)

# Calcular las predicciones del modelo ajustado
y_pred = GetModel(x, param)

# Calcular el error residual (diferencia entre los valores reales y los predichos)
error = np.abs(y - y_pred)
threshold = 3 * np.std(error)
y_clean = np.where(error > threshold, y_pred, y)

# Contar los datos eliminados o corregidos
n_eliminados = np.sum(error > threshold)
print(f'1.a) Número de datos eliminados: {n_eliminados}')

# Crear la gráfica antes y después de la limpieza
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# Antes de la limpieza (datos originales)
ax[0].scatter(x, y, color='b', label='Original Data')
ax[0].plot(x, y_pred, color='r', lw=2, label='Fitted Model')
ax[0].set_title('Datos Originales')
ax[0].set_xlabel('Wavelength (pm)')
ax[0].set_ylabel('Intensity (mJy)')
ax[0].plt.grid(True)
ax[0].legend()

# Después de la limpieza (datos corregidos)
ax[1].scatter(x, y_clean, color='g', label='Cleaned Data')
ax[1].plot(x, y_pred, color='r', lw=2, label='Fitted Model')
ax[1].set_title('Datos Limpiados')
ax[1].set_xlabel('Wavelength (pm)')
ax[1].set_ylabel('Intensity (mJy)')
ax[1].plt.grid(True)
ax[1].legend()

# Guardar la gráfica en un archivo PDF
plt.tight_layout()
plt.savefig('limpieza.pdf')

# Mostrar la gráfica
plt.show()


#1.b Hallar los dos picos de rayos X eliminando el espectro de fondo
print('1.b Hallar los dos picos de rayos X eliminando el espectro de fondo')
peaks, properties = find_peaks(y_clean, prominence=0.05, height=0.1) 

# Coordenadas de los picos detectados
picos_x = x[peaks]
picos_y = y_clean[peaks]

# Graficar los datos corregidos con los picos detectados
plt.figure(figsize=(10, 6))
plt.scatter(x, y_clean, color='green', s=5, label="Datos corregidos")  # Datos limpios
plt.scatter(picos_x, picos_y, color='red', s=50, label="Picos detectados")  # Picos detectados
plt.xlabel("Wavelength (pm)")
plt.ylabel("Intensity (mJy)")
plt.title("Espectro corregido con picos detectados")
plt.grid(True)
plt.savefig("picos.pdf")
plt.legend()
plt.show()

# Imprimir las posiciones de los picos detectados
print("Posiciones y valores de los picos detectados:")
for i, (px, py) in enumerate(zip(picos_x, picos_y)):
    print(f"Pico {i+1}: Wavelength = {px:.4f}, Intensity = {py:.4f}")

print('1.b) Método: {picos maximos hallados sin ruido}')
















#Punto 2
print('Punto 2')
file_path = "hysteresis.dat"

with open(file_path, "r") as file:
  lines = file.readlines()

# Limpieza de datos
cleaned_data = []

for line in lines:
  line = line.replace("-", " ", 1)
  parts = line.strip().split()

  # Caso 1. donde B y H estan concatenados con un (-)
  if len(parts) == 2 and "-" in parts[1]:
    t = parts[0]
    bh_split = parts[1].rsplit("-", 1)

    B = "-" + bh_split[0]
    H = "-" + bh_split[1]

    parts = [t, B, H]

  # Caso 2. B y H estan estan concatenados con un 0.
  if len(parts) == 2 and "0." in parts[1]:
      t = parts[0]
      bh_split = parts[1].rsplit("0.", 1)

      B = bh_split[0]
      H = "0." + bh_split[1]

      parts = [t, B, H]

  cleaned_data.append(parts)


# Convertir a dataframe
df_hysteresis = pd.DataFrame(cleaned_data, columns=["t", "B", "H"]).astype(float)

# 2.a) Graficar H y B en funcion de t
print('2.a')
plt.figure(figsize=(10, 5))
plt.plot(df_hysteresis["t"], df_hysteresis["B"], label="Campo externo B(mT)")
plt.plot(df_hysteresis["t"], df_hysteresis["H"], label="Densidad del campo interno H(A/m)")
plt.xlabel("Tiempo(ms)")
plt.ylabel("Intensidad del campo")
plt.legend()
plt.title("Histeris magnética")
plt.grid(True)
plt.savefig("histerico.pdf")
plt.show()

# 2b) Frecuencia de la señal
peaks, _ = find_peaks(df_hysteresis["B"])
peak_times = df_hysteresis["t"].iloc[peaks].values
periodos = np.diff(peak_times)
frecuencia = 1 / np.mean(periodos)
print(f"2.b) Frecuencia de la señal: {frecuencia:.3f} Hz")

print('Procedimiento para 2b: 1. Detectamos los picos de B(t)\n'
      '2. Calculamos los periodos entre picos consecutivos\n'
      '3. Sacamos el promedio de los periodos T(medio)\n'
      '4. Obtuvimos la frecuencia como 1/T(medio)')

# 2c)
y = df_hysteresis["H"] 
x = df_hysteresis["B"]
area_hysteresis = scipy.integrate.simpson(y=y,x=x, dx=1, axis=-1, even='avg') 
#Hacemos conversion de unidades para que nos quede en el sistema internacional.
area_hysteresis_conv = area_hysteresis*(10**(-3))
print(f"2.c) Energia perdida: {area_hysteresis_conv:.3f} J/m^3")

#Grafica histeresis H vs B
plt.figure(figsize=(8, 6))
plt.fill(df_hysteresis["B"], df_hysteresis["H"],color="blue", alpha=0.5, label="Energia perdida")
plt.xlabel("Campo externo B(mT)")
plt.ylabel("Densidad del campo interno H(A/m)")
plt.legend()
plt.title("Energia perdida")
plt.grid(True)
plt.savefig("energy.pdf")
plt.show()
