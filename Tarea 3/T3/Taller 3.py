import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

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


