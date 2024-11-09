import numpy as np
import matplotlib.pyplot as plt

# Definir parámetros comunes
np.random.seed(42)
n_steps = 100  # Número de pasos en el tiempo
t = np.arange(n_steps)

# Proceso de Martingala (Ejemplo: Random Walk)
martingala = np.cumsum(np.random.randn(n_steps))

# Proceso de Markov (Ejemplo: Proceso de Markov simple con dos estados)
markov = np.zeros(n_steps)
state = 0  # Estado inicial
for i in range(1, n_steps):
    if np.random.rand() > 0.5:
        state = 1 - state  # Cambiar de estado con probabilidad 0.5
    markov[i] = markov[i-1] + (1 if state == 1 else -1)

# Crear gráficos en un solo subplot
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Gráfico de la Martingala
axes[0].plot(t, martingala, label='Martingala', color='blue')
axes[0].set_title('Evolución de un Proceso de Martingala')
axes[0].set_xlabel('Tiempo')
axes[0].set_ylabel('Precio del activo $X_t$')
axes[0].legend()

# Gráfico del Proceso de Markov
axes[1].plot(t, markov, label='Markov', color='green')
axes[1].set_title('Evolución de un Proceso de Markov')
axes[1].set_xlabel('Tiempo')
axes[1].set_ylabel('Precio del activo $X_t$')
axes[1].legend()

plt.tight_layout()
plt.show()
