import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.integrate import simpson
from scipy.integrate import trapezoid
import numpy as np
import sympy as sym
import scipy
#punto 1

#1.a Limpiar los datos
print('1.a limpiar los datos')
df1 = pd.read_csv('Rhodium.csv')
x1 = np.array(df1['Wavelength (pm)'], dtype=float)
y1 = np.array(df1['Intensity (mJy)'], dtype=float)

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
param = GetFit(x1, y1, n)

# Calcular las predicciones del modelo ajustado
y_pred = GetModel(x1, param)

# Calcular el error residual (diferencia entre los valores reales y los predichos)
error = np.abs(y1 - y_pred)
threshold = 3 * np.std(error)
y_clean = np.where(error > threshold, y_pred, y1)

# Contar los datos eliminados o corregidos
n_eliminados = np.sum(error > threshold)
print(f'1.a) Número de datos eliminados: {n_eliminados}')

# Crear la gráfica antes y después de la limpieza
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# Antes de la limpieza (datos originales)
ax[0].scatter(x1, y1, color='b', label='Original Data')
ax[0].plot(x1, y_pred, color='r', lw=2, label='Fitted Model')
ax[0].set_title('Datos Originales')
ax[0].set_xlabel('Wavelength (pm)')
ax[0].set_ylabel('Intensity (mJy)')
ax[0].grid(True)
ax[0].legend()

# Después de la limpieza (datos corregidos)
ax[1].scatter(x1, y_clean, color='g', label='Cleaned Data')
ax[1].plot(x1, y_pred, color='r', lw=2, label='Fitted Model')
ax[1].set_title('Datos Limpiados')
ax[1].set_xlabel('Wavelength (pm)')
ax[1].set_ylabel('Intensity (mJy)')
ax[1].grid(True)
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
picos_x = x1[peaks]
picos_y = y_clean[peaks]

# Graficar los datos corregidos con los picos detectados
plt.figure(figsize=(10, 6))
plt.scatter(x1, y_clean, color='green', s=5, label="Datos corregidos")  # Datos limpios
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

print('1.c hallar la localización del máximo y el ancho a media altura (FWHM).')

#1.c Función para calcular el FWHM
def calculate_fwhm(x, y, peak_index):
    """Calcula el ancho a media altura (FWHM) de un pico en y ubicado en peak_index."""
    peak_height = y[peak_index]
    half_max = peak_height / 2

    # Buscar la izquierda del pico
    left_side = np.where(y[:peak_index] < half_max)[0]
    left_x = x[left_side[-1]] if len(left_side) > 0 else None

    # Buscar la derecha del pico
    right_side = np.where(y[peak_index:] < half_max)[0]
    right_x = x[peak_index + right_side[0]] if len(right_side) > 0 else None

    # Calcular FWHM
    if left_x is not None and right_x is not None:
        fwhm = right_x - left_x
    else:
        fwhm = None  # Si no se encuentra el FWHM

    return fwhm, left_x, right_x

# Detectar los picos
peaks, properties = find_peaks(y_clean, prominence=0.05, height=0.1)
picos_x = x1[peaks]
picos_y = y_clean[peaks]

# Guardar resultados
results = []
for i, peak_index in enumerate(peaks):
    fwhm, left_x, right_x = calculate_fwhm(x1, y_clean, peak_index)

    # Almacenar resultados en una lista
    results.append({
        "Pico": i + 1,
        "Wavelength (pm)": f"{picos_x[i]:.4f}",
        "Intensity (mJy)": f"{picos_y[i]:.4f}",
        "FWHM (pm)": f"{fwhm:.4f}" if fwhm else "No encontrado",
        "Left X (pm)": f"{left_x:.4f}" if left_x else "No encontrado",
        "Right X (pm)": f"{right_x:.4f}" if right_x else "No encontrado"
    })

# Convertir a DataFrame para mejor visualización
df_results = pd.DataFrame(results)

# Mostrar la tabla con tabulate
print(tabulate(df_results, headers="keys", tablefmt="grid"))













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
plt.scatter(df_hysteresis["t"], df_hysteresis["B"], label="Campo externo B(mT)")
plt.scatter(df_hysteresis["t"], df_hysteresis["H"], label="Densidad del campo interno H(A/m)")
plt.xlabel("Tiempo(ms)")
plt.ylabel("Intensidad del campo")
plt.legend()
plt.title("Histeris magnética")
plt.grid(True)
plt.savefig("histerico.pdf")
plt.show()

# 2b) Frecuencia de la señal
def func_seno(t, A, f, phi, C):
    return A * np.sin(2 * np.pi * f * t + phi) + C
popt, _ = curve_fit(func_seno,df_hysteresis["t"],df_hysteresis["H"], p0=[2.8, 1/2, 0, -0.10])
# p0 son los valores iniciales de amplitud, frecuencia y desplazamiento vertical y horizontal
# que estimo yo a partir de la grafica del primer punto.
# Extraer los parámetros ajustados
A_fit, f_fit, phi_fit, C_fit = popt


# Mostrar los resultados

print(f"Frecuencia nueva(f): {f_fit}")


print('Procedimiento para 2b: 1. Modelamos una funcion seno\n'
      '2. hacemos un ajuste de datos con esta funcion\n'
      '3. imprimimos la frecuencia\n')

# 2c)
y = df_hysteresis["B"]
x = df_hysteresis["H"]

B = df_hysteresis["B"].values
H = df_hysteresis["H"].values

# Implementar la fórmula para calcular el área dentro de la curva cerrada
def calculate_area(x, y):
    n = len(x)
    area = 0.0
    for i in range(n - 1):
        area += x[i] * y[i + 1] - x[i + 1] * y[i]
    # Cerrar el polígono sumando el último término
    area += x[-1] * y[0] - x[0] * y[-1]
    return 0.5 * abs(area)

# Calcular el área usando la fórmula
area_enclosed = calculate_area(H, B)
print(f"Área encerrada dentro del ciclo de histéresis: {area_enclosed*(10**-3):.4f} J/m^3")

#Grafica histeresis H vs B
plt.figure(figsize=(8, 6))
plt.fill(df_hysteresis["H"], df_hysteresis["B"],color="blue", alpha=0.5, label="Energia perdida")
plt.xlabel("Densidad del campo interno H(A/m)")
plt.ylabel("Campo externo B(mT)")
plt.legend()
plt.title("Energia perdida")
plt.grid(True)
plt.savefig("energy.pdf")
plt.show()