import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from numba import njit

# Parámetros
alpha = 0.022
L = 2.0  # Longitud del dominio
dx = 0.02  # Paso espacial
dt = 0.00005  # Paso temporal
T_max = 2000  # Tiempo total de simulación
save_interval = 100  # Guardar cada 100 pasos para reducir uso de memoria

N = int(L / dx)  # Número de puntos en x
M = int(T_max / dt)  # Número de pasos de tiempo

# Malla espacial
x = np.arange(0, L, dx, dtype=np.float64)

# Condición inicial
psi = np.cos(np.pi * x).astype(np.float64)
psi_old = psi.copy()  # Para el esquema de segundo orden en el tiempo

@njit
def evolucion_kdv(psi, psi_old, N, M, dx, dt, alpha, save_interval):
    """Simula la ecuación de KdV con diferencias finitas según el esquema de la imagen."""
    num_saves = M // save_interval
    sol = np.zeros((num_saves, N), dtype=np.float64)
    sol[0, :] = psi.copy()
    
    save_idx = 1
    for t in range(1, M):
        psi_new = np.empty_like(psi)
        
        for j in range(N):
            jm2 = (j - 2) % N  # Índice j-2 con frontera periódica
            jm1 = (j - 1) % N  # Índice j-1 con frontera periódica
            jp1 = (j + 1) % N  # Índice j+1 con frontera periódica
            jp2 = (j + 2) % N  # Índice j+2 con frontera periódica
            
            term1 = (dt / (3 * dx)) * psi[j] * (psi[jp1] - psi[jm1])
            term2 = (alpha * dt / dx**3) * (psi[jp2] - 2 * psi[jp1] + 2 * psi[jm1] - psi[jm2])
            
            psi_new[j] = psi_old[j] - term1 - term2  # Esquema de segundo orden en el tiempo
        
        psi_old[:], psi[:] = psi[:], psi_new  # Evita copias innecesarias
        
        if t % save_interval == 0:
            sol[save_idx, :] = psi.copy()
            save_idx += 1
    
    return sol

# Ejecutar la simulación
sol = evolucion_kdv(psi, psi_old, N, M, dx, dt, alpha, save_interval)

# Visualización de la simulación
plt.imshow(sol[:].T, aspect='auto', cmap='magma')
plt.colorbar(label='$\psi(x, t)$')
plt.xlabel("Time [s]")
plt.ylabel("Angle x [m]")
plt.title("Simulación de KdV")
plt.savefig("kdv_simulation.png")
plt.show()

# Cálculo de cantidades conservadas
times = np.arange(0, T_max, save_interval * dt)
masa = np.trapz(sol, dx=dx, axis=1)
momento = np.trapz(sol**2, dx=dx, axis=1)
energia = np.trapz((sol*3) / 3 - (alpha * np.gradient(sol, dx, axis=1))*2, dx=dx, axis=1)

# Graficar cantidades conservadas
fig, axs = plt.subplots(3, 1, figsize=(8, 10))
axs[0].plot(times, masa, label="Masa")
axs[0].set_ylabel("Masa")
axs[0].legend()
axs[1].plot(times, momento, label="Momento")
axs[1].set_ylabel("Momento")
axs[1].legend()
axs[2].plot(times, energia, label="Energía")
axs[2].set_ylabel("Energía")
axs[2].legend()
axs[2].set_xlabel("Tiempo")

plt.tight_layout()
plt.savefig("3.b.pdf")
plt.show()

# Generar video de la simulación
fig, ax = plt.subplots()
cax = ax.imshow(sol.T, aspect='auto', cmap='magma', animated=True)
fig.colorbar(cax, label='$\psi(x, t)$')
ax.set_xlabel("Time [s]")
ax.set_ylabel("Angle x [m]")
ax.set_title("Simulación de KdV")

def update(frame):
    cax.set_data(sol[:frame].T)
    return cax,

ani = animation.FuncAnimation(fig, update, frames=len(sol), interval=50, blit=True)
ani.save("3.a.mp4", writer="ffmpeg", fps=30)
plt.close(fig)