import numpy as np
from funciones import *
import time
import matplotlib.pyplot as plt

class CarroAutonomo:
    def __init__(self, senales, num_robot, vel, experimental, PID, dt, K, path, xi, vehiculo_path, h_robot, robotat, robot, tiempo, flag_graficar, delta, numParqueo):
        self.no_robot = num_robot
        self.v_max = vel
        self.path = path
        self.newpath = None
        self.senales = senales
        self.done = False
        self.done2 = False
        self.flag_graficar = flag_graficar
        self.delta = delta
        self.PID = PID
        self.experimental = experimental
        self.FlagSend = False
        self.wL = 0
        self.wR = 0
        self.numParqueo = numParqueo
        self.parqueos = np.array([[0.65, 0.65], [0.65 + 0.40, 0.65]])
        self.change = False
        self.errores = np.zeros((K, 2))
    
        if self.PID:
            self.diferencial = 50
        else:
            deltas = np.diff(path, axis=0)
            distancias = np.sqrt(np.sum(deltas**2, axis=1))
            self.v_mean = np.mean(distancias) / dt
            self.diferencial = round(self.v_max / self.v_mean)
        
        self.tramo = self.diferencial
        # PID Controller constants
        self.alpha = 50
        self.kpO = 2
        self.kiO = 0.0001
        self.kdO = -1
        self.EO = 0
        self.eO_1 = 0
        self.alto_cercano = False
        self.bandera = False
        self.senal_inactiva = 0
        self.idx_almacenar = 0
        self.recorrido = np.zeros((K + 1, 2))
        self.tiempo = tiempo
        self.a = 0.08

        # Inicialización y condiciones iniciales
        xi0 = np.array([xi[0], xi[1], xi[2]])
        u0 = np.array([0, 0])
        self.xi = xi0  # vector de estado
        self.u = u0    # vector de entradas

        self.XI = np.zeros((3, K))
        self.U = np.zeros((2, K))

        self.vehiculo_path = vehiculo_path
        self.h_robot = h_robot
        self.robotat = robotat
        self.robot = robot
        self.time_inicial = 0
        self.time_inicial2 = 0
        self.time_alto = 0
        self.tramo_deseado = 0
        self.FalgObstaculos = False
        self.timeDone = 0
        self.FlagParqueo1 = False
        self.FlagParqueo2 = False
        self.FlagParqueo3 = False
        self.factorExtraVel = 1
        

    
def method1(self, k, dt, K, carros, G, node_coordinates):
    # Planificación Global: Determinación de la ruta ideal
    if self.FlagParqueo3 and (not self.FlagParqueo2):
        posicion_d = self.parqueos[self.numParqueo-1]
        xg = posicion_d[0]
        yg = posicion_d[1]
    else:
        xg = self.path[self.tramo, 0]
        yg = self.path[self.tramo, 1]

    xpos = self.xi[0]
    ypos = self.xi[1]
    phi = self.xi[2]

    self.recorrido[self.idx_almacenar, :] = [xpos, ypos]

    # Solo senales activas
    senales_activas = [senal for senal in self.senales if senal['activo']]
    
    # Planificación de Comportamiento: Determinación del comportamiento basado en el entorno
    semaforo_cercano, _, color = hay_semaforo_cerca(xpos, ypos, senales_activas)
    self.senales, _ = actualizar_senales_desde_csv('semaforos.csv', self.senales)

    # Cambios en la velocidad
    factor_velocidad, dentro = hay_senal_de_bajar_velocidad(xpos, ypos, senales_activas)
    if dentro:
        v_max_modificada = self.v_max * factor_velocidad
    else:
        v_max_modificada = self.v_max

    # Evasión de Colisiones: Lógica de evitar colisiones con otros vehículos
    factor_velocidad1 = evitar_colision(self, carros, dt)
    v_max_modificada = v_max_modificada * factor_velocidad1 * self.factorExtraVel

    # Respeto de Señales de Tránsito: Manejo de señales de tránsito
    if not self.alto_cercano:
        self.alto_cercano, id_senal, self.time_alto = hay_senal_de_alto_cerca(xpos, ypos, senales_activas)
        if self.alto_cercano:
            self.time_inicial = self.tiempo[k]
            for senal in self.senales:
                if senal['id'] == id_senal:
                    senal['activo'] = False
                    break
            self.senal_inactiva = id_senal
    else:
        if self.tiempo[k] - self.time_inicial >= self.time_alto:
            self.alto_cercano = False
            self.bandera = True
            self.time_inicial2 = self.tiempo[k]

    if self.bandera and ((self.tiempo[k] - self.time_inicial2) > 6):
        for senal in self.senales:
            if senal['id'] == self.senal_inactiva:
                senal['activo'] = True
                break
        self.bandera = False

    # Planificación Local: Cálculo de la trayectoria específica
    trayectoria = calcular_trayectoria_segura(self.path, xpos, ypos, xg, yg, phi)

    # Cálculo del Controlador: Ajuste de la posición y velocidad del carro
    ajustar_velocidad_y_posicion(self, trayectoria, v_max_modificada, dt)
    
    # Verificar final de ruta para iniciar el parqueo automático
    if self.tramo >= len(self.path) - 1:
        self.FlagParqueo2 = True
        if self.numParqueo == 0:
            self.numParqueo = 1

    # Parqueo Automático
    if self.FlagParqueo2:
        realizar_parqueo(self, self.parqueos[self.numParqueo-1], dt)

    # Actualización de índice de almacenamiento
    self.idx_almacenar += 1
