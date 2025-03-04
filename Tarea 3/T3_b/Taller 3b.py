import numpy as np
import matplotlib.pyplot as plt
from numba import jit

import matplotlib.animation as animation
from IPython.display import HTML

#primer punto Poisson

# Parámetros de la malla
N = 100  # Número de puntos en cada dirección
L = 1.1  # Extensión de la malla (ligeramente mayor que el disco unitario)
dx = 2 * L / (N - 1)
x = np.linspace(-L, L, N)
y = np.linspace(-L, L, N)
X, Y = np.meshgrid(x, y)

# Inicialización de la función potencial
phi = np.random.rand(N, N) * 0.1  # Condiciones iniciales aleatorias
rho = -4 * np.pi * (-X - Y)  # Término fuente de la ecuación de Poisson

# Condición de frontera: φ(x, y) = sin(7θ) en el círculo x² + y² = 1
theta = np.arctan2(Y, X)
boundary_condition = np.sin(7 * theta)
mask = (X**2 + Y**2 >= 1)  # Máscara para la frontera
phi[mask] = boundary_condition[mask]

@jit(nopython=True)
def gauss_seidel(phi, rho, mask, boundary_condition, dx, tol=1e-4, max_iter=15000):
    error = tol + 1
    iteration = 0
    N = phi.shape[0]
    while error > tol and iteration < max_iter:
        phi_old = phi.copy()
        error = 0
        
        # Aplicamos la ecuación de Poisson en los puntos interiores del disco
        for i in range(1, N-1):
            for j in range(1, N-1):
                if X[i, j]**2 + Y[i, j]**2 < 1:  # Solo dentro del disco
                    phi_new = 0.25 * (phi[i+1, j] + phi[i-1, j] + phi[i, j+1] + phi[i, j-1] - dx**2 * rho[i, j])
                    error = max(error, abs(phi_new - phi[i, j]))
                    phi[i, j] = phi_new
        
        # Aplicamos la condición de frontera
        for i in range(N):
            for j in range(N):
                if mask[i, j]:
                    phi[i, j] = boundary_condition[i, j]
        
        iteration += 1
    
    return phi

# Ejecutar el método acelerado con numba
phi = gauss_seidel(phi, rho, mask, boundary_condition, dx)


#punto 2

import numpy as np
import matplotlib.pyplot as plt
from celluloid import Camera

# Parámetros
c = 1.0          # Velocidad de la onda
L = 2.0          # Longitud del dominio
Nx = 100         # Número de puntos espaciales
T = 2.0          # Duración total (segundos)
fps = 60         # Frames por segundo
Nt = int(T * fps)  # Número de pasos temporales
dx = L / (Nx - 1)
dt = T / (Nt - 1)
C = c * dt / dx   # Número de Courant
assert C <= 1, f"C = {C} > 1, el esquema no es estable"

# Malla espacial
x = np.linspace(0, L, Nx)

# Condición inicial
u0 = np.exp(-125 * (x - 0.5)**2)

# Función para aplicar condiciones de contorno
def apply_boundary_conditions(u, boundary_type):
    if boundary_type == 'dirichlet':
        u[0] = 0
        u[-1] = 0
    elif boundary_type == 'neumann':
        u[0] = u[1]
        u[-1] = u[-2]
    elif boundary_type == 'periodic':
        u[0] = u[-2]
        u[-1] = u[1]
    return u

# Simulación
def simulate(boundary_type):
    u = u0.copy()
    u_prev = u.copy()
    u_next = np.zeros_like(u)

    # Primer paso temporal
    u_next = u + 0.5 * (C**2) * (np.roll(u, -1) - 2*u + np.roll(u, 1))
    u_next = apply_boundary_conditions(u_next, boundary_type)

    # Configurar la figura para la animación
    fig, ax = plt.subplots()
    camera = Camera(fig)

    # Iterar sobre el tiempo
    for t in range(Nt):
        # Calcular el siguiente paso
        u_next = 2*u - u_prev + (C**2) * (np.roll(u, -1) - 2*u + np.roll(u, 1))
        u_next = apply_boundary_conditions(u_next, boundary_type)

        # Graficar cada frame
        ax.plot(x, u_next, 'b-')
        ax.set_ylim(-1, 1)
        ax.set_title(f'Tiempo: {t*dt:.2f}s, Condición: {boundary_type.capitalize()}')
        ax.set_xlabel('x')
        ax.set_ylabel('u(t, x)')
        camera.snap()

        # Actualizar pasos
        u_prev = u.copy()
        u = u_next.copy()

    # Guardar animación
    animation = camera.animate(interval=1000/fps)
    animation.save(f'2_{boundary_type}.mp4', writer='ffmpeg', fps=fps)
    plt.close()

# Simular para cada condición de contorno
for boundary_type in ['dirichlet', 'neumann', 'periodic']:
    print(f"Simulando con condición de contorno: {boundary_type}")
    simulate(boundary_type)
    
import os
for boundary_type in ['dirichlet', 'neumann', 'periodic']:
    print(f"Simulando con condición de contorno: {boundary_type}")
    simulate(boundary_type)
print("Videos guardados en:", os.getcwd())



#punto 4

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from tqdm import tqdm
import shutil

# Comprobamos la ruta de ffmpeg
print("FFmpeg path:", shutil.which("ffmpeg"))

# -----------------------------------------------------
# 1. PARÁMETROS DEL DOMINIO
# -----------------------------------------------------
Lx = 1.0   # Ancho en x (m)
Ly = 2.0   # Largo en y (m)

nx = 101   # Puntos en x
ny = 201   # Puntos en y

dx = Lx/(nx - 1)
dy = Ly/(ny - 1)

x_vals = np.linspace(0, Lx, nx)  # eje x (0..1)
y_vals = np.linspace(0, Ly, ny)  # eje y (0..2)

# -----------------------------------------------------
# 2. TIEMPO DE SIMULACIÓN
# -----------------------------------------------------
dt   = 0.0005   # Paso de tiempo
tmax = 2.0      # 2 s
Nt   = int(tmax/dt) + 1

# -----------------------------------------------------
# 3. MATRIZ DE VELOCIDADES c(x,y)
# -----------------------------------------------------
c_agua = 0.5

# Pared alrededor de y=1 con grosor total 0.04 => y en [0.98, 1.02].
y_pared_inf = 1.0 - 0.02  # 0.98
y_pared_sup = 1.0 + 0.02  # 1.02

# Elipse centrada en (x=0.5, y=1.0), semiejes (0.2, 0.2)
x_c, y_c = 0.5, 1.0
a_x, a_y = 0.2, 0.2

c_matrix = np.zeros((ny, nx))
for j in range(ny):
    for i in range(nx):
        xx = x_vals[i]
        yy = y_vals[j]
        # Verificamos si está en la franja de la "pared"
        if (y_pared_inf <= yy <= y_pared_sup):
            # Checamos si (xx,yy) está dentro de la elipse
            dentro_elipse = ((xx - x_c)**2 / a_x**2) + ((yy - y_c)**2 / a_y**2) <= 1.0
            if dentro_elipse:
                c_matrix[j, i] = c_agua  # Apertura elíptica => agua
            else:
                c_matrix[j, i] = 0.0     # Pared
        else:
            c_matrix[j, i] = c_agua     # Agua normal

# -----------------------------------------------------
# 4. ARREGLO 3D PARA u(t,y,x)
# -----------------------------------------------------
u = np.zeros((Nt, ny, nx))

# -----------------------------------------------------
# 5. FUENTE SINUSOIDAL
# -----------------------------------------------------
# Fuente en (x=0.5, y=0.5), amplitud 1 cm, freq=10 Hz
x_f, y_f = 0.5, 0.5
A = 0.01   # 1 cm
f = 10.0
omega = 2.0 * np.pi * f

# Hallar índices i_f, j_f más cercanos
i_f = np.argmin(np.abs(x_vals - x_f))
j_f = np.argmin(np.abs(y_vals - y_f))

# -----------------------------------------------------
# 6. CONDICIONES DE FRONTERA
# -----------------------------------------------------
def aplicar_condiciones_frontera(u2d):
    # Fijar u=0 en todos los bordes
    u2d[0, :]   = 0.0   # y=0
    u2d[-1, :]  = 0.0   # y=2
    u2d[:, 0]   = 0.0   # x=0
    u2d[:, -1]  = 0.0   # x=1
    return u2d

# -----------------------------------------------------
# 7. CONDICIONES INICIALES
# -----------------------------------------------------
u[0] = 0.0
u[1] = 0.0

u[0] = aplicar_condiciones_frontera(u[0])
u[1] = aplicar_condiciones_frontera(u[1])

# -----------------------------------------------------
# 8. BUCLE DE TIEMPO (ECUACIÓN DE ONDA 2D) + BARRA DE PROGRESO
# -----------------------------------------------------
for n in tqdm(range(1, Nt-1), desc="Simulación", ncols=80):
    for j in range(1, ny-1):
        for i in range(1, nx-1):
            d2x = (u[n, j, i+1] - 2.0*u[n, j, i] + u[n, j, i-1]) / dx**2
            d2y = (u[n, j+1, i] - 2.0*u[n, j, i] + u[n, j-1, i]) / dy**2
            c2  = c_matrix[j, i]**2

            u[n+1, j, i] = (2.0*u[n, j, i] - u[n-1, j, i]
                            + (dt**2)*c2*(d2x + d2y))

    # Inyectar la fuente sinusoidal en (j_f, i_f)
    t_n = n * dt
    u[n+1, j_f, i_f] = A * np.sin(omega * t_n)

    # Aplicar condiciones de frontera
    u[n+1] = aplicar_condiciones_frontera(u[n+1])

# -----------------------------------------------------
# 9. ANIMACIÓN
# -----------------------------------------------------
frames = []
# Tomamos un fotograma cada 10 pasos => ~200 fotogramas en total
for n in range(0, Nt, 10):
    frames.append(u[n].copy())

fig, ax = plt.subplots(figsize=(5,5))
# Ajustamos la escala de color para mostrar +/- 1 cm
im = ax.imshow(frames[0],
               extent=[0, Lx, 0, Ly],
               origin='lower', cmap='RdBu', aspect='auto',
               vmin=-0.01, vmax=0.01)

ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
ax.set_title('Onda 2D con pared y apertura elíptica (fuente sinusoidal)')

def update(frame):
    im.set_data(frame)
    return [im]

ani = animation.FuncAnimation(fig, update, frames=frames, interval=50)

# -----------------------------------------------------
# 10. GUARDAR VIDEO EN MP4 (requiere ffmpeg instalado)
# -----------------------------------------------------
# Ajusta fps=5 (o el que desees)
ani.save('onda_2D.mp4', writer='ffmpeg', fps=5)

plt.show()
