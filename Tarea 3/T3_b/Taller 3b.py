import numpy as np
import matplotlib.pyplot as plt
from numba import jit
from celluloid import Camera
import matplotlib.animation as animation
from IPython.display import HTML
from scipy.integrate import solve_ivp

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



#Punto 2 ondas

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



#Punto 3

# Parámetros
alpha = 0.022
L = 2.0  # Longitud del dominio
dx = 0.02  # Paso espacial
dt = 0.00005  # Paso temporal
T_max = 2  # Tiempo total de simulación
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




