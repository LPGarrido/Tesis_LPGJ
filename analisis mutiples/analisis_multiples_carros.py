import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import scipy.stats as stats

# Archivos proporcionados
archivos = {
    'carro1': {
        'posiciones': r"C:\Users\lpgar\Documents\5 Año\10 Semestre\Diseño e Innovación en Ingeniería 2\pyRobot3\analisis mutiples\carro_1_PID_posiciones.csv",
        'velocidades': r"C:\Users\lpgar\Documents\5 Año\10 Semestre\Diseño e Innovación en Ingeniería 2\pyRobot3\analisis mutiples\carro_1_PID_velocidades.csv"
    },
    'carro2': {
        'posiciones': r"C:\Users\lpgar\Documents\5 Año\10 Semestre\Diseño e Innovación en Ingeniería 2\pyRobot3\analisis mutiples\carro_2_PID_posiciones.csv",
        'velocidades': r"C:\Users\lpgar\Documents\5 Año\10 Semestre\Diseño e Innovación en Ingeniería 2\pyRobot3\analisis mutiples\carro_2_PID_velocidades.csv"
    },
    'carro3': {
        'posiciones': r"C:\Users\lpgar\Documents\5 Año\10 Semestre\Diseño e Innovación en Ingeniería 2\pyRobot3\analisis mutiples\carro_3_PID_posiciones.csv",
        'velocidades': r"C:\Users\lpgar\Documents\5 Año\10 Semestre\Diseño e Innovación en Ingeniería 2\pyRobot3\analisis mutiples\carro_3_PID_velocidades.csv"
    },
    'carro4': {
        'posiciones': r"C:\Users\lpgar\Documents\5 Año\10 Semestre\Diseño e Innovación en Ingeniería 2\pyRobot3\analisis mutiples\carro_4_PID_posiciones.csv",
        'velocidades': r"C:\Users\lpgar\Documents\5 Año\10 Semestre\Diseño e Innovación en Ingeniería 2\pyRobot3\analisis mutiples\carro_4_PID_velocidades.csv"
    }
}

# Definir nombres de columnas para los archivos de posiciones y velocidades
columnas_posiciones = ['Posición Deseada en X', 'Posición Deseada en Y', 'Posición Real en X', 'Posición Real en Y']
columnas_velocidades = ['velocidad lineal', 'velocidad angular']

# Variables para almacenar datos de todos los carros
todos_errores = []
todos_vel_lineal = []
todos_vel_angular = []
todos_tiempos = []

resultados_errores = {}
resultados_vel_lineal = {}
resultados_vel_angular = {}
resultados_tiempos = {}

# Procesar cada carro
for carro, paths in archivos.items():
    # Leer archivos CSV
    posiciones_df = pd.read_csv(paths['posiciones'], header=None, names=columnas_posiciones)
    velocidades_df = pd.read_csv(paths['velocidades'], header=None, names=columnas_velocidades)
    
    # Cálculo de ECM y semejanza
    ecm_X = mean_squared_error(posiciones_df['Posición Deseada en X'], posiciones_df['Posición Real en X'])
    ecm_Y = mean_squared_error(posiciones_df['Posición Deseada en Y'], posiciones_df['Posición Real en Y'])
    similaridad_X = (1 - np.sqrt(ecm_X) / (np.max(posiciones_df['Posición Deseada en X']) - np.min(posiciones_df['Posición Deseada en X']))) * 100
    similaridad_Y = (1 - np.sqrt(ecm_Y) / (np.max(posiciones_df['Posición Deseada en Y']) - np.min(posiciones_df['Posición Deseada en Y']))) * 100
    similaridad_total = (similaridad_X + similaridad_Y) / 2

    # Estadísticas de velocidad lineal
    stats_lineal = velocidades_df['velocidad lineal'].describe()
    cv_lineal = stats_lineal['std'] / stats_lineal['mean']
    kurtosis_lineal = stats.kurtosis(velocidades_df['velocidad lineal'])
    skewness_lineal = stats.skew(velocidades_df['velocidad lineal'])

    # Estadísticas de velocidad angular
    stats_angular = velocidades_df['velocidad angular'].describe()
    cv_angular = stats_angular['std'] / stats_angular['mean']
    kurtosis_angular = stats.kurtosis(velocidades_df['velocidad angular'])
    skewness_angular = stats.skew(velocidades_df['velocidad angular'])
    
    # Estudio de tiempos
    tiempo_total_segundos = len(velocidades_df) * 0.1
    tiempo_quieto_lineal = (velocidades_df['velocidad lineal'] == 0).sum() * 0.1
    tiempo_quieto_angular = (velocidades_df['velocidad angular'] == 0).sum() * 0.1
    tiempo_quieto_general = ((velocidades_df['velocidad lineal'] == 0) & (velocidades_df['velocidad angular'] == 0)).sum() * 0.1
    porcentaje_quieto_lineal = (tiempo_quieto_lineal / tiempo_total_segundos) * 100
    porcentaje_quieto_angular = (tiempo_quieto_angular / tiempo_total_segundos) * 100
    porcentaje_quieto_total = (tiempo_quieto_general / tiempo_total_segundos) * 100
    tiempo_dentro_vel_max = (velocidades_df['velocidad lineal'] <= 0.13).sum() * 0.1
    porcentaje_dentro_vel_max = (tiempo_dentro_vel_max / tiempo_total_segundos) * 100

    # Guardar resultados para cada carro
    resultados_errores[carro] = [ecm_X, ecm_Y, similaridad_X, similaridad_Y, similaridad_total]
    resultados_vel_lineal[carro] = [stats_lineal['mean'], stats_lineal['std'], stats_lineal['min'], stats_lineal['max'], cv_lineal, kurtosis_lineal, skewness_lineal]
    resultados_vel_angular[carro] = [stats_angular['mean'], stats_angular['std'], stats_angular['min'], stats_angular['max'], cv_angular, kurtosis_angular, skewness_angular]
    resultados_tiempos[carro] = [tiempo_total_segundos, porcentaje_quieto_lineal, porcentaje_quieto_angular, porcentaje_quieto_total, porcentaje_dentro_vel_max]

    # Agregar a listas generales
    todos_errores.append([ecm_X, ecm_Y, similaridad_X, similaridad_Y, similaridad_total])
    todos_vel_lineal.append(velocidades_df['velocidad lineal'])
    todos_vel_angular.append(velocidades_df['velocidad angular'])
    todos_tiempos.append([tiempo_total_segundos, porcentaje_quieto_lineal, porcentaje_quieto_angular, porcentaje_quieto_total, porcentaje_dentro_vel_max])

# Calcular promedios generales
promedio_errores = np.mean(todos_errores, axis=0)
vel_lineal_general = pd.concat(todos_vel_lineal).describe()
cv_lineal_general = vel_lineal_general['std'] / vel_lineal_general['mean']
kurtosis_lineal_general = stats.kurtosis(pd.concat(todos_vel_lineal))
skewness_lineal_general = stats.skew(pd.concat(todos_vel_lineal))

vel_angular_general = pd.concat(todos_vel_angular).describe()
cv_angular_general = vel_angular_general['std'] / vel_angular_general['mean']
kurtosis_angular_general = stats.kurtosis(pd.concat(todos_vel_angular))
skewness_angular_general = stats.skew(pd.concat(todos_vel_angular))

promedio_tiempos = np.mean(todos_tiempos, axis=0)

# Añadir promedios a los resultados
resultados_errores['General'] = promedio_errores
resultados_vel_lineal['General'] = [vel_lineal_general['mean'], vel_lineal_general['std'], vel_lineal_general['min'], vel_lineal_general['max'], cv_lineal_general, kurtosis_lineal_general, skewness_lineal_general]
resultados_vel_angular['General'] = [vel_angular_general['mean'], vel_angular_general['std'], vel_angular_general['min'], vel_angular_general['max'], cv_angular_general, kurtosis_angular_general, skewness_angular_general]
resultados_tiempos['General'] = promedio_tiempos

# Convertir resultados a DataFrames para mostrar en la consola
df_errores = pd.DataFrame(resultados_errores, index=['ECM X', 'ECM Y', 'Simil X (%)', 'Simil Y (%)', 'Simil Total (%)']).T
df_vel_lineal = pd

# Convertir resultados a DataFrames para mostrar en la consola
df_errores = pd.DataFrame(resultados_errores, index=['ECM X', 'ECM Y', 'Simil X (%)', 'Simil Y (%)', 'Simil Total (%)']).T
df_vel_lineal = pd.DataFrame(resultados_vel_lineal, index=['Media', 'Desviación', 'Mín', 'Máx', 'CV', 'Curtosis', 'Asimetría']).T
df_vel_angular = pd.DataFrame(resultados_vel_angular, index=['Media', 'Desviación', 'Mín', 'Máx', 'CV', 'Curtosis', 'Asimetría']).T
df_tiempos = pd.DataFrame(resultados_tiempos, index=['Tiempo Total (s)', '% Tiempo Quieto Lineal', '% Tiempo Quieto Angular', '% Tiempo Quieto Total', '% Tiempo Vel Máx']).T

# Desplegar resultados en la consola
print("Tabla de Errores y Similitudes:")
print(df_errores)
print("\nTabla de Estadísticas Descriptivas de la Velocidad Lineal:")
print(df_vel_lineal)
print("\nTabla de Estadísticas Descriptivas de la Velocidad Angular:")
print(df_vel_angular)
print("\nTabla de Tiempos y Porcentajes:")
print(df_tiempos)

# Visualización en subplots para posiciones con escala de color basada en la velocidad
fig, axs = plt.subplots(2, 2, figsize=(15, 7.5))
fig.suptitle('Posiciones con velocidad (Lyapunov)')

# Para las posiciones coloreadas según la velocidad
for i, (carro, paths) in enumerate(archivos.items()):
    posiciones_df = pd.read_csv(paths['posiciones'], header=None, names=columnas_posiciones)
    velocidades_df = pd.read_csv(paths['velocidades'], header=None, names=columnas_velocidades)
    
    # Cálculo de la magnitud de la velocidad (combinación de lineal y angular)
    velocidad_total = np.sqrt(velocidades_df['velocidad lineal']**2 + velocidades_df['velocidad angular']**2)
    
    # Seleccionar el subplot adecuado
    ax = axs[i // 2, i % 2]
    
    # Gráfica de posiciones reales vs deseadas con color basado en la velocidad
    scatter = ax.scatter(posiciones_df['Posición Real en X'], posiciones_df['Posición Real en Y'], c=velocidad_total, cmap='plasma')
    ax.plot(posiciones_df['Posición Deseada en X'], posiciones_df['Posición Deseada en Y'], 'r--', label='Posición Deseada')
    ax.set_title(f"Carro {i+1}: Posiciones vs Velocidad")
    ax.set_xlabel('Posición en X')
    ax.set_ylabel('Posición en Y')
    fig.colorbar(scatter, ax=ax, label='Velocidad (m/s)')

# Ajustar el layout
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# Visualización en subplots para posiciones con escala de color basada en el error de posición
fig, axs = plt.subplots(2, 2, figsize=(15, 7.5))
fig.suptitle('Posiciones con error (Lyapunov)')

# Para las posiciones coloreadas según el error
for i, (carro, paths) in enumerate(archivos.items()):
    posiciones_df = pd.read_csv(paths['posiciones'], header=None, names=columnas_posiciones)
    
    # Cálculo del error entre la posición deseada y la real
    error = np.sqrt((posiciones_df['Posición Real en X'] - posiciones_df['Posición Deseada en X'])**2 + 
                    (posiciones_df['Posición Real en Y'] - posiciones_df['Posición Deseada en Y'])**2)
    
    # Seleccionar el subplot adecuado
    ax = axs[i // 2, i % 2]
    
    # Gráfica de posiciones reales vs deseadas con color basado en el error
    scatter = ax.scatter(posiciones_df['Posición Real en X'], posiciones_df['Posición Real en Y'], c=error, cmap='viridis')
    ax.plot(posiciones_df['Posición Deseada en X'], posiciones_df['Posición Deseada en Y'], 'r--', label='Posición Deseada')
    ax.set_title(f"Carro {i+1}: Posiciones vs Error")
    ax.set_xlabel('Posición en X')
    ax.set_ylabel('Posición en Y')
    fig.colorbar(scatter, ax=ax, label='Error (m)')

# Ajustar el layout
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
