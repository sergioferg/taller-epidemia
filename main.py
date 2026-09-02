import matplotlib.pyplot as plt
import numpy as np

# 1. Definición de Parámetros y Condiciones Iniciales
N = 3e7          # Población total
I0 = 100         # Infectados iniciales
E0 = 0           # Expuestos iniciales
R0 = 0           # Recuperados iniciales
S0 = N - I0      # Susceptibles iniciales

sigma = 0.2
gamma = 0.1

# Condiciones de simulación
t_start = 0
t_end = 200      # 200 días para visualizar el aplanamiento de la curva
h = 0.1          # Tamaño de paso
t_eval = np.arange(t_start, t_end + h, h)
num_pasos = len(t_eval)

# 2. Función del Sistema de Ecuaciones Diferenciales
def seir_derivadas(t, y, beta, sigma, gamma, N):
    S, E, I, R = y
    dSdt = -beta * S * I / N
    dEdt = (beta * S * I / N) - (sigma * E)
    dIdt = (sigma * E) - (gamma * I)
    dRdt = gamma * I
    return np.array([dSdt, dEdt, dIdt, dRdt])

# 3. Implementación Manual de Runge-Kutta de 4to Orden (RK4)
def rk4_paso(t, y, h, beta, sigma, gamma, N):
    k1 = seir_derivadas(t, y, beta, sigma, gamma, N)
    k2 = seir_derivadas(t + h/2, y + (h/2) * k1, beta, sigma, gamma, N)
    k3 = seir_derivadas(t + h/2, y + (h/2) * k2, beta, sigma, gamma, N)
    k4 = seir_derivadas(t + h, y + h * k3, beta, sigma, gamma, N)

    y_next = y + (h / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    return y_next

# 4. Inicialización de Vectores de Estado
# Matrices para almacenar resultados: filas = tiempo, columnas = [S, E, I, R]
resultados_A = np.zeros((num_pasos, 4))
resultados_B = np.zeros((num_pasos, 4))

y0 = np.array([S0, E0, I0, R0])
resultados_A[0] = y0
resultados_B[0] = y0

# 5. Bucle de Simulación Principal
for i in range(1, num_pasos):
    t_actual = t_eval[i-1]

    # Parámetro beta para el Escenario A (Constante)
    beta_A = 0.6

    # Parámetro beta para el Escenario B (Cambio en t=30)
    if t_actual < 30:
        beta_B = 0.6
    else:
        beta_B = 0.25

    # Calcular siguiente paso con RK4
    resultados_A[i] = rk4_paso(t_actual, resultados_A[i-1], h, beta_A, sigma, gamma, N)
    resultados_B[i] = rk4_paso(t_actual, resultados_B[i-1], h, beta_B, sigma, gamma, N)

# 6. Análisis de Conservación de la Población
# Suma de S + E + I + R en cada paso de tiempo
poblacion_total_A = np.sum(resultados_A, axis=1)
poblacion_total_B = np.sum(resultados_B, axis=1)

# Calcular el error absoluto máximo respecto a N
error_conservacion_A = np.max(np.abs(poblacion_total_A - N))
error_conservacion_B = np.max(np.abs(poblacion_total_B - N))

print("Análisis de Conservación")
print(f"Error máximo conservación Escenario A: {error_conservacion_A:.2e}")
print(f"Error máximo conservación Escenario B: {error_conservacion_B:.2e}")
print("Si los valores son cercanos a 0, se verifica la conservación.\n")

# 7. Generación de Gráficas
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Gráfica Escenario A
ax1.plot(t_eval, resultados_A[:, 0], label='Susceptibles (S)', color='blue')
ax1.plot(t_eval, resultados_A[:, 1], label='Expuestos (E)', color='orange')
ax1.plot(t_eval, resultados_A[:, 2], label='Infectados (I)', color='red')
ax1.plot(t_eval, resultados_A[:, 3], label='Recuperados (R)', color='green')
ax1.set_title('Escenario A: Propagación Libre ($\\beta = 0.6$)')
ax1.set_xlabel('Días')
ax1.set_ylabel('Población')
ax1.grid(True)
ax1.legend()

# Gráfica Escenario B
ax2.plot(t_eval, resultados_B[:, 0], label='Susceptibles (S)', color='blue')
ax2.plot(t_eval, resultados_B[:, 1], label='Expuestos (E)', color='orange')
ax2.plot(t_eval, resultados_B[:, 2], label='Infectados (I)', color='red')
ax2.plot(t_eval, resultados_B[:, 3], label='Recuperados (R)', color='green')
ax2.axvline(x=30, color='gray', linestyle='--', label='Inicio Intervención (t=30)')
ax2.set_title('Escenario B: Con Intervención en t=30 ($\\beta = 0.25$)')
ax2.set_xlabel('Días')
ax2.set_ylabel('Población')
ax2.grid(True)
ax2.legend()

plt.tight_layout()
plt.show()
