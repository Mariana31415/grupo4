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

# -----------------------------------------------------
# 1. PARÁMETROS DEL DOMINIO
# -----------------------------------------------------
Nx = 50       # Número de puntos en x
Ny = 50       # Número de puntos en y
Lx = 4.0      # Dominio en x: [0, 4]
Ly = 4.0      # Dominio en y: [0, 4]

x = np.linspace(0, Lx, Nx)
y = np.linspace(0, Ly, Ny)

dx = x[1] - x[0]
dy = y[1] - y[0]   # Suponemos dx = dy

# -----------------------------------------------------
# 2. TIEMPO DE SIMULACIÓN
# -----------------------------------------------------
Nt = 1000
t = np.linspace(0, 4, Nt)  # Simulación en [0,4] s
dt = t[1] - t[0]

v = 5.0  # Velocidad de la onda
lambda_ = v * dt / dx
print("lambda =", lambda_)  # Aproximadamente 0.245

gamma = 2 * dt   # Término de absorción

# -----------------------------------------------------
# 3. ARREGLO 3D PARA u(t,y,x)
# -----------------------------------------------------
# u tendrá dimensiones (Nt, Ny, Nx)
u = np.zeros((Nt, Ny, Nx))

# -----------------------------------------------------
# 4. CONDICIÓN INICIAL: PULSO GAUSSIANO 2D
# -----------------------------------------------------
# Usamos un pulso gaussiano centrado en (x0, y0) = (2,2)
x0, y0 = 2.0, 2.0
sigma = 0.2
for j in range(Ny):
    for i in range(Nx):
        u[0, j, i] = np.exp(-(((x[i]-x0)**2 + (y[j]-y0)**2) / (sigma**2)))

# Para velocidad inicial = 0, copiamos la condición inicial:
u[1] = u[0].copy()

# -----------------------------------------------------
# 5. CONDICIONES DE FRONTERA ABSORBENTES / FUJOSAS
# -----------------------------------------------------
def aplicar_condiciones_frontera(arr2d):
    # En este ejemplo, imponemos:
    #   - Borde izquierdo: inyección de fuente h2(t) (ver más abajo)
    #   - Borde derecho: copia (por ejemplo, u = u[neighbor])
    #   - Bordes superior e inferior: u = 0
    arr2d[:, -1] = arr2d[:, -2]   # Borde derecho: Neumann (copia)
    arr2d[0, :] = 0.0             # Borde inferior
    arr2d[-1, :] = 0.0            # Borde superior
    return arr2d

# Función fuente para el borde izquierdo (i = 0)
def h2(time):
    return 0.5 * np.cos(5 * np.pi * time)

# -----------------------------------------------------
# 6. ESQUEMA FINITO DIFERENCIAS EN 2D (CON ABSORCIÓN)
# -----------------------------------------------------
# El esquema para un punto interior (i,j) es:
# u[n+1,j,i] = 2*u[n,j,i] - u[n-1,j,i]
#              + dt^2 * v^2 * [ (u[n,j,i+1]-2*u[n,j,i]+u[n,j,i-1])/dx^2
#                               + (u[n,j+1,i]-2*u[n,j,i]+u[n,j-1,i])/dy^2 ]
#              - gamma*u[n-1,j,i] + gamma*u[n-2,j,i]
#
# Para el borde izquierdo (i = 0), forzamos u[n, j, 0] = h2(t[n]).

for n in tqdm(range(2, Nt)):
    # Borde izquierdo: inyectamos la fuente en todos los y
    for j in range(Ny):
        u[n, j, 0] = h2(t[n])
    
    # Recorremos el interior (i de 1 a Nx-2, j de 1 a Ny-2)
    for j in range(1, Ny-1):
        for i in range(1, Nx-1):
            d2x = (u[n-1, j, i+1] - 2*u[n-1, j, i] + u[n-1, j, i-1]) / dx**2
            d2y = (u[n-1, j+1, i] - 2*u[n-1, j, i] + u[n-1, j-1, i]) / dy**2
            # Usamos el mismo c (v) en todo el dominio; si se desea variar c(x,y), habría que multiplicar c^2
            u[n, j, i] = (2*u[n-1, j, i] - u[n-2, j, i]
                          + (dt**2)*v**2*(d2x + d2y)
                          - gamma*u[n-1, j, i] + gamma*u[n-2, j, i])
    
    # Para el borde derecho (i = Nx-1), lo tratamos con una diferencia unidireccional
    for j in range(1, Ny-1):
        i = Nx - 1
        d2x = (u[n-1, j, i] - 2*u[n-1, j, i-1] + u[n-1, j, i-2]) / dx**2
        d2y = (u[n-1, j+1, i] - 2*u[n-1, j, i] + u[n-1, j-1, i]) / dy**2
        u[n, j, i] = (2*u[n-1, j, i] - u[n-2, j, i]
                      + (dt**2)*v**2*(d2x + d2y)
                      - gamma*u[n-1, j, i] + gamma*u[n-2, j, i])
    
    # Aplicamos condiciones de frontera (superior e inferior)
    u[n] = aplicar_condiciones_frontera(u[n])

# -----------------------------------------------------
# 7. ANIMACIÓN
# -----------------------------------------------------
# Seleccionamos fotogramas cada 10 pasos para la animación
frames = []
for n in range(0, Nt, 10):
    frames.append(u[n].copy())

fig, ax = plt.subplots(figsize=(6,6))
im = ax.imshow(frames[0], extent=[0, Lx, 0, Ly],
               origin='lower', cmap='RdBu', aspect='auto')
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.set_title('Onda 2D con absorción (extensión 1D a 2D)')

def update(frame):
    im.set_data(frame)
    return [im]

ani = animation.FuncAnimation(fig, update, frames=frames, interval=50)
plt.show()



    

