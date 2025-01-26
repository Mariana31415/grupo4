import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.integrate import simpson
import numpy as np
import sympy as sym


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
