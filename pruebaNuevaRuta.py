"""
self.xi = np.array([0, 0, math.radians(90)])  # Punto 1 con ángulo de 90 grados
self.pos_futuro_obs = np.array([0, 1])  # Punto 4

self.xi = np.array([0, 0, math.radians(-90)])  # Punto 1 con ángulo de 90 grados
self.pos_futuro_obs = np.array([0, -1])  # Punto 4

self.xi = np.array([0, 0, math.radians(0)])  # Punto 1 con ángulo de 90 grados
self.pos_futuro_obs = np.array([1, 0])  # Punto 4

self.xi = np.array([0, 0, math.radians(180)])  # Punto 1 con ángulo de 90 grados
self.pos_futuro_obs = np.array([-1, 0])  # Punto 4

"""

import numpy as np
import matplotlib.pyplot as plt
import math

class ConexionPuntos:
    def __init__(self):
        self.xi = np.array([0, 0, math.radians(90)])  # Punto 1 con ángulo de 90 grados
        self.pos_futuro_obs = np.array([0, 1])  # Punto 4

        #self.xi = np.array([0, 0, math.radians(-90)])  # Punto 1 con ángulo de 90 grados
        #self.pos_futuro_obs = np.array([0, -1])  # Punto 4

        #self.xi = np.array([0, 0, math.radians(0)])  # Punto 1 con ángulo de 90 grados
        #self.pos_futuro_obs = np.array([1, 0])  # Punto 4

        #self.xi = np.array([0, 0, math.radians(180)])  # Punto 1 con ángulo de 90 grados
        #self.pos_futuro_obs = np.array([-1, 0])  # Punto 4

    def conectar_puntos(self):
        # Calcular punto 1
        x1, y1 = self.xi[0], self.xi[1]

        # Ángulo de 45 grados desde el punto 1 para el punto 2
        angulo_punto2 = self.xi[2] + math.radians(45)
        distancia = (0.20**2 + 0.20**2)**0.5
        x2 = x1 + distancia * math.cos(angulo_punto2)
        y2 = y1 + distancia * math.sin(angulo_punto2)

        # Punto 4 es el punto futuro conocido
        x4, y4 = self.pos_futuro_obs

        # Opciones para el punto 3
        opciones = [
            (x4 - 0.20, y4 - 0.20),
            (x4 - 0.20, y4 + 0.20),
            (x4 + 0.20, y4 - 0.20),
            (x4 + 0.20, y4 + 0.20)
        ]

        # Vector de la línea 1 a 4 extendido a 3D
        vector_14 = np.array([x4 - x1, y4 - y1, 0])

        # Elegir la opción para el punto 3 que sea casi paralela o antiparalela a 1 a 4
        mejor_opcion = None
        paralelismo_tol = 0.01  # Tolerancia para considerar vectores casi paralelos
        for opcion in opciones:
            vector_23 = np.array([opcion[0] - x2, opcion[1] - y2, 0])
            # Calcular el producto cruzado para ver si los vectores son casi paralelos
            producto_cruz = np.linalg.norm(np.cross(vector_14, vector_23))
            if producto_cruz < paralelismo_tol:
                # Comprobar el ángulo entre las líneas 2-3 y 3-4 para asegurar que sea obtuso
                vector_43 = np.array([x4 - opcion[0], y4 - opcion[1], 0])
                cos_theta = np.dot(vector_23[:-1], -vector_43[:-1]) / (np.linalg.norm(vector_23[:-1]) * np.linalg.norm(vector_43[:-1]))
                theta = np.arccos(np.clip(cos_theta, -1.0, 1.0)) * 180 / math.pi  # Convertir a grados
                if theta > 90:
                    mejor_opcion = opcion
                    break

        if mejor_opcion is None:
            raise ValueError("No se encontró una opción válida para el punto 3 que sea casi paralela.")

        x3, y3 = mejor_opcion

        # Retornar las coordenadas de los puntos
        return np.array([(x1, y1), (x2, y2), (x3, y3), (x4, y4)])

# Crear instancia de la clase y obtener puntos
conexion = ConexionPuntos()
puntos = conexion.conectar_puntos()
print(puntos)
# Gráfica
plt.figure()
plt.plot([puntos[0][0], puntos[1][0]], [puntos[0][1], puntos[1][1]], 'b-', linewidth=1, label='De 1 a 2')  # Línea de 1 a 2
plt.plot([puntos[1][0], puntos[2][0]], [puntos[1][1], puntos[2][1]], 'b-', linewidth=1, label='De 2 a 3')  # Línea de 2 a 3
plt.plot([puntos[2][0], puntos[3][0]], [puntos[2][1], puntos[3][1]], 'b-', linewidth=1, label='De 3 a 4')  # Línea de 3 a 4
plt.scatter(*zip(*puntos), color='blue', s=100)  # Marcar los puntos
plt.axis('equal')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Ruta - Caso 01')
plt.legend()
plt.show()


# Gráfica
plt.figure()
plt.plot([puntos[0][0], puntos[1][0]], [puntos[0][1], puntos[1][1]], 'b-', linewidth=1, label='De 1 a 2')  # Línea de 1 a 2
plt.plot([puntos[1][0], puntos[2][0]], [puntos[1][1], puntos[2][1]], 'b-', linewidth=1, label='De 2 a 3')  # Línea de 2 a 3
plt.plot([puntos[2][0], puntos[3][0]], [puntos[2][1], puntos[3][1]], 'b-', linewidth=1, label='De 3 a 4')  # Línea de 3 a 4
plt.scatter(*zip(*puntos), color='blue', s=100)  # Marcar los puntos
plt.axis('equal')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Ruta Suavizada en el Grafo')
plt.legend()
plt.show()




"""
self.xi = np.array([0, 0, math.radians(90)])  # Punto 1 con ángulo de 90 grados
self.pos_futuro_obs = np.array([0, 1])  # Punto 4

self.xi = np.array([0, 0, math.radians(-90)])  # Punto 1 con ángulo de 90 grados
self.pos_futuro_obs = np.array([0, -1])  # Punto 4

self.xi = np.array([0, 0, math.radians(0)])  # Punto 1 con ángulo de 90 grados
self.pos_futuro_obs = np.array([1, 0])  # Punto 4

self.xi = np.array([0, 0, math.radians(180)])  # Punto 1 con ángulo de 90 grados
self.pos_futuro_obs = np.array([-1, 0])  # Punto 4

from funciones import *

# Cargar datos desde un archivo CSV
ruta_smooth = pd.read_csv('ruta_smooth.csv').values

plt.figure()
plt.plot([0, 3.8], [0, 4.8], 'r--', linewidth=1, label='Referencia')
plt.plot(ruta_smooth[:, 0], ruta_smooth[:, 1], 'b', linewidth=1, label='Ruta Suavizada')
plt.axis('equal')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Ruta Suavizada en el Grafo')
plt.legend()
plt.show()
"""