import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from scipy.signal import argrelextrema
from numba import jit
import matplotlib.animation as animation
from IPython.display import HTML
from scipy.integrate import solve_ivp


# Parámetros del problema
g = 9.773  # Gravedad en Bogotá (m/s²)
m = 10  # Masa del proyectil (kg)
v0 = 20.0  # Velocidad inicial (m/s)
dt = 0.01  # Paso de tiempo
t_max = 10  # Tiempo máximo de simulación

def f_prime(state, beta):
    """Derivadas del sistema: (x, y, vx, vy) con fricción cuadrática"""
    x, y, vx, vy = state
    v = np.sqrt(vx**2 + vy**2)  # Magnitud de la velocidad
    ax = -beta/m * v * vx
    ay = -g - (beta/m) * v * vy
    return np.array([vx, vy, ax, ay])

def runge_kutta_4(state, dt, beta):
    """Un paso del método de Runge-Kutta de 4to orden"""
    k1 = f_prime(state, beta)
    k2 = f_prime(state + 0.5 * dt * k1, beta)
    k3 = f_prime(state + 0.5 * dt * k2, beta)
    k4 = f_prime(state + dt * k3, beta)
    return state + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)

def simulate(angle, beta):
    """Simula el movimiento del proyectil y calcula la energía perdida por fricción"""
    x, y = 0, 0
    vx, vy = v0 * np.cos(angle), v0 * np.sin(angle)
    state = np.array([x, y, vx, vy])
    
    E_perdida = 0  # Inicializar energía disipada

    for _ in range(int(t_max/dt)):
        v = np.sqrt(vx**2 + vy**2)  # Magnitud de la velocidad
        P_friccion = beta * v**3  # Potencia disipada
        E_perdida += P_friccion * dt  # Integración numérica
        
        state = runge_kutta_4(state, dt, beta)
        x, y, vx, vy = state
        if y < 0:  # Se detiene cuando toca el suelo
            break
    
    return x, E_perdida  # Retorna el alcance y la energía perdida

def find_best_angle(beta):
    """Encuentra el ángulo que maximiza el alcance usando optimización"""
    result = minimize_scalar(lambda angle: -simulate(angle, beta)[0], bounds=(0, np.pi/2), method='bounded')
    return np.degrees(result.x)  # Convertir a grados

# Valores de beta a probar
betas = np.linspace(0, 1.9, 20)
best_angles = []
lost_energies = []

for beta in betas:
    best_angle_deg = find_best_angle(beta)
    best_angles.append(best_angle_deg)
    
    _, E_perdida = simulate(np.radians(best_angle_deg), beta)  # Simular con ángulo óptimo
    lost_energies.append(E_perdida)





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

# Parámetros físicos y orbitales
GM = 4.0 * np.pi**2
a = 0.38709893
e = 0.20563069

x0 = a * (1.0 + e)
y0 = 0.0
vx0 = 0.0
vy0 = np.sqrt(GM * (2.0 / x0 - 1.0 / a))

# Función de derivadas (Newton)
def derivadas_newton(t, state):
    x, y, vx, vy = state
    r = np.sqrt(x*x + y*y)
    ax = -GM * x / r**3
    ay = -GM * y / r**3
    return [vx, vy, ax, ay]

# Resolver con RK45
t_span = (0, 10)  # 10 años
dt = 1e-3
t_eval = np.arange(t_span[0], t_span[1], dt)
sol = solve_ivp(derivadas_newton, t_span, [x0, y0, vx0, vy0], t_eval=t_eval, method='RK45')

x, y = sol.y[0], sol.y[1]
vx, vy = sol.y[2], sol.y[3]

# Encontrar periastro y apoastro
r = np.sqrt(x**2 + y**2)
periastro_idx = argrelextrema(r, np.less)[0]

# Calcular ángulos y aplicar np.unwrap para evitar saltos
angles_periastro = np.unwrap(np.arctan2(y[periastro_idx], x[periastro_idx]))

t_periastro = sol.t[periastro_idx]

# Convertir ángulos a grados y luego a arcosegundos
angles_periastro_deg = np.degrees(angles_periastro) % 360
angles_periastro_arcsec = angles_periastro_deg * 3600

# Calcular el período teórico
P_teo = np.sqrt(a**3)

# Calcular el período simulado (promedio de diferencias entre tiempos de periastro)
P_sim = np.mean(np.diff(t_periastro))

# Imprimir en el formato solicitado
print(f'2.a) {P_teo = :.5f}; {P_sim = :.5f}')




# Ajuste lineal con incertidumbre
(coefs, cov_matrix) = np.polyfit(t_periastro, angles_periastro_arcsec, 1, cov=True)  # Extraer coeficientes y covarianza
pendiente_año, intercepto = coefs
incertidumbre_pendiente = np.sqrt(cov_matrix[0, 0])  # Extraer incertidumbre de la pendiente

# Convertir pendiente a arcsec/siglo
pendiente_siglo = pendiente_año * 100
incertidumbre_pendiente_siglo = incertidumbre_pendiente * 100

#punto 3












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

