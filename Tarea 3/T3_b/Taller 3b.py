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
from numba import jit

# Parámetros iniciales y constantes
a0 = 1  # Radio de Bohr en unidades atómicas
e = 1   # Carga del electrón en unidades atómicas
m_e = 1 # Masa del electrón en unidades atómicas
alpha = 1/137.035999206  # Constante de estructura fina
EH = 1  # Energía de Hartree en unidades atómicas
hbar = 1  # Constante reducida de Planck en unidades atómicas

# Condiciones iniciales
x0 = 1
y0 = 0
vx0 = 0
vy0 = 1
dt = 0.01
tmax = 1000

@jit(nopython=True)
def coulomb_force(x, y):
    r = np.sqrt(x**2 + y**2)
    if r < 1e-6:  # Evitar singularidades numéricas
        return np.array([0.0, 0.0])
    return -np.array([x, y]) / r**3

@jit(nopython=True)
def derivs(state):
    x, y, vx, vy = state
    ax, ay = coulomb_force(x, y)
    return np.array([vx, vy, ax, ay])

@jit(nopython=True)
def rk4_step(state, dt):
    k1 = derivs(state)
    k2 = derivs(state + dt/2 * k1)
    k3 = derivs(state + dt/2 * k2)
    k4 = derivs(state + dt * k3)
    return state + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

@jit(nopython=True)
def simulate_orbit(dt, tmax, include_larmor, amplification=1.0):
    num_steps = int(tmax / dt)
    states = np.zeros((num_steps, 4))
    states[0] = np.array([x0, y0, vx0, vy0])
    radius = np.zeros(num_steps)
    energy = np.zeros(num_steps)

    for i in range(1, num_steps):
        state = states[i-1]
        state = rk4_step(state, dt)
        x, y, vx, vy = state
        r = np.sqrt(x**2 + y**2)
        radius[i] = r

        kinetic_energy = 0.5 * m_e * (vx**2 + vy**2)
        potential_energy = -e**2 / r if r > 1e-6 else -e**2 / 1e-6
        energy[i] = kinetic_energy + potential_energy

        if include_larmor:
            v2 = vx**2 + vy**2
            loss = amplification * (2/3) * alpha**3 * v2 / r**4 * dt

            if loss >= kinetic_energy:
                vx = 0
                vy = 0
                return states[:i], radius[:i], energy[:i], np.linspace(0, i*dt, i)
            else:
                v_new = np.sqrt(2 * (kinetic_energy - loss) / m_e)
                factor = v_new / np.sqrt(v2)
                vx *= factor
                vy *= factor

        states[i] = np.array([x, y, vx, vy])
        if r < 0.01:  # Evita caída indefinida
            return states[:i], radius[:i], energy[:i], np.linspace(0, i*dt, i)

    return states, radius, energy, np.linspace(0, tmax, num_steps)

# Simulación sin Larmor
states_no_larmor, radius_no_larmor, energy_no_larmor, times_no_larmor = simulate_orbit(dt, tmax, False)

# Simulación con Larmor (aumento de la amplificación)
states_larmor, radius_larmor, energy_larmor, times_larmor = simulate_orbit(dt, tmax, True, amplification=1e6)


print(f'2.b) t_fall = {times_larmor[-1]:.5f} en unidades de ℏ/EH')


















#punto 3

#a
# Parámetros físicos y orbitales
GM = 4.0 * np.pi**2
c = 63239.7263
a = 0.38709893
e = 0.20563069

x0 = a * (1.0 + e)
y0 = 0.0
r0 = x0
v0 = np.sqrt(GM * (2.0/r0 - 1.0/a))
vx0 = 0.0
vy0 = +v0
t_span = (0.0, 10.0)
dt = 1e-2
alpha = 1e-2

# Función de derivadas (Newton y Relativista)
def derivadas_newton(t, state):
    x, y, vx, vy = state
    r = np.sqrt(x*x + y*y)
    ax = -GM * x / (r**3)
    ay = -GM * y / (r**3)
    return np.array([vx, vy, ax, ay])

# Método RK4
def runge_kutta_4(derivs, y0, t_span, dt):
    t0, tf = t_span
    n_steps = int(np.floor((tf - t0)/dt))
    times = np.zeros(n_steps+1)
    sol = np.zeros((n_steps+1, len(y0)))

    times[0] = t0
    sol[0] = y0

    for i in range(n_steps):
        t = times[i]
        y = sol[i]

        k1 = derivs(t, y)
        k2 = derivs(t + dt/2, y + dt*k1/2)
        k3 = derivs(t + dt/2, y + dt*k2/2)
        k4 = derivs(t + dt, y + dt*k3)

        sol[i+1] = y + (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4)
        times[i+1] = t + dt

    return times, sol

estado_inicial = np.array([x0, y0, vx0, vy0])
t_newt, sol_newt = runge_kutta_4(derivadas_newton, estado_inicial, t_span, dt)
x_n, y_n = sol_newt[:, 0], sol_newt[:, 1]




plt.close()


#b















#Punto 4 Cuantización de la energía
# Parámetros
hbar = 1.0  # Constante de Planck reducida
m = 1.0     # Masa
omega = 1.0 # Frecuencia angular
L = 6.0     # Rango espacial: [-L, L]
N = 1000    # Número de puntos en la malla
x = np.linspace(-L, L, N)  # Malla espacial
h = x[1] - x[0]  # Tamaño del paso

# Construcción de la matriz Hamiltoniana
main_diag = (hbar**2 / (m * h**2)) + 0.5 * m * omega**2 * x**2  # Diagonal principal
off_diag = -0.5 * (hbar**2 / (m * h**2)) * np.ones(N-1)  # Diagonales adyacentes
H = np.diag(main_diag) + np.diag(off_diag, 1) + np.diag(off_diag, -1)

# Cálculo de valores y vectores propios
eigvals, eigvecs = np.linalg.eigh(H)  # Resuelve la matriz simétrica

# Seleccionar los primeros 5 niveles
n_levels = 5
energies = eigvals[:n_levels]
wavefunctions = eigvecs[:, :n_levels]

# Normalizar las funciones de onda
for i in range(n_levels):
    norm = np.sqrt(np.sum(wavefunctions[:, i]**2) * h)
    wavefunctions[:, i] /= norm

# Pozo de potencial parabólico (línea punteada)
V = 0.5 * m * omega**2 * x**2


# Factor de escala para reducir la amplitud de las funciones de onda
scale_factor = 0.5  # Puedes ajustar este valor según necesites

# Funciones de onda desplazadas por sus energías
for n in range(n_levels):
    psi = scale_factor * wavefunctions[:, n]  # Escalar la función de onda
    E_n = energies[n]
    

