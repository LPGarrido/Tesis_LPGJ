import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import socket
from scipy.spatial.transform import Rotation as R

from matplotlib.patches import Polygon, Circle, Rectangle
import struct
from math import atan2, sin, cos, radians, degrees, pi
import json
import time
import math

def crear_grafo(lineas):
    """
    Crea un grafo dirigido a partir de una lista de líneas representadas por sus puntos extremos y calcula sus componentes conectadas.

    Parámetros:
    lineas (ndarray): Un arreglo numpy de forma (N, 4), donde cada fila representa una línea con las coordenadas (x1, y1, x2, y2) de sus puntos extremos.

    Retorno:
    G (networkx.DiGraph): Un grafo dirigido donde los nodos representan puntos únicos y las aristas están ponderadas según la distancia euclidiana entre los puntos.
    node_coordinates (ndarray): Un arreglo numpy de forma (M, 2) con las coordenadas de los nodos únicos presentes en el grafo.
    """
    # Crear lista de puntos únicos.
    puntos_unicos = np.unique(np.vstack((lineas[:, :2], lineas[:, 2:])), axis=0)
    n = puntos_unicos.shape[0]

    # Crear un diccionario para mapear las coordenadas a índices de nodos.
    coordenadas_a_nodos = {tuple(coordenada): i for i, coordenada in enumerate(puntos_unicos)}

    # Preparar listas de nodos inicial y final.
    s = []
    t = []
    weights = []

    # Llenar las listas con los índices correspondientes.
    for linea in lineas:
        s.append(coordenadas_a_nodos[tuple(linea[:2])])
        t.append(coordenadas_a_nodos[tuple(linea[2:])])
        weights.append(np.linalg.norm(linea[:2] - linea[2:]))

    # Crear el grafo dirigido.
    G = nx.DiGraph()
    for i in range(len(s)):
        G.add_edge(s[i], t[i], weight=weights[i])
    
    node_coordinates = puntos_unicos

    # Identificar componentes conectadas
    componentes = list(nx.connected_components(G.to_undirected()))
    # componentes_dict = {nodo: i + 1 for i, componente in enumerate(componentes) for nodo in componente}

    if len(componentes) > 1:
        print('El grafo NO está completamente conectado.')
        print(f'Número de componentes conectadas: {len(componentes)}')
    else:
        print('El grafo está completamente conectado.')

    return G, node_coordinates

def find_closest_node(node_coordinates, x, y):
    """
    Encuentra el nodo más cercano a un punto dado dentro de un conjunto de coordenadas.

    Parámetros:
    node_coordinates (ndarray): Un arreglo numpy de forma (N, 2) que contiene las coordenadas de los nodos, donde cada fila representa un nodo en el plano (x, y).
    x (float): La coordenada x del punto de referencia.
    y (float): La coordenada y del punto de referencia.

    Retorno:
    int: El índice del nodo más cercano al punto (x, y) dentro del arreglo `node_coordinates`.
    """
    diffs = node_coordinates - np.array([x, y])
    dists = np.linalg.norm(diffs, axis=1)
    return np.argmin(dists)

def calcular_ruta(G, node_coordinates, Ax, Ay, Bx, By):
    """
    Calcula la ruta más corta entre dos puntos dados en un grafo utilizando las coordenadas más cercanas en el grafo.

    Parámetros:
    G (networkx.Graph): Un grafo dirigido que contiene los nodos y aristas con pesos asociados.
    node_coordinates (ndarray): Un arreglo numpy de forma (N, 2) que contiene las coordenadas de los nodos, donde cada fila representa un nodo en el plano (x, y).
    Ax (float): La coordenada x del punto de partida.
    Ay (float): La coordenada y del punto de partida.
    Bx (float): La coordenada x del punto de destino.
    By (float): La coordenada y del punto de destino.

    Retorno:
    ruta (ndarray): Un arreglo numpy que contiene las coordenadas de los nodos en la ruta más corta desde el punto A al punto B.
    d (float): La distancia total de la ruta más corta. Si no existe ruta, devuelve infinito.
    """
    nodoA = find_closest_node(node_coordinates, Ax, Ay)
    nodoB = find_closest_node(node_coordinates, Bx, By)

    try:
        path = nx.shortest_path(G, source=nodoA, target=nodoB, weight='weight')
        d = nx.shortest_path_length(G, source=nodoA, target=nodoB, weight='weight')
        ruta = node_coordinates[path, :]
    except nx.NetworkXNoPath:
        print('No se encontró ruta.')
        return [], float('inf')

    return ruta, d

def f(xi, u):
    """
    Calcula la derivada del estado para un sistema de movimiento en el plano, utilizando un vector de estado y un vector de control.

    Parámetros:
    xi (ndarray): Un arreglo numpy de forma (3,) que representa el estado del sistema, donde `xi[0]` y `xi[1]` son las coordenadas (x, y) y `xi[2]` es la orientación (ángulo) en radianes.
    u (ndarray): Un arreglo numpy de forma (2,) que representa el vector de control, donde `u[0]` es la velocidad lineal y `u[1]` es la velocidad angular.

    Retorno:
    ndarray: Un arreglo numpy de forma (3,) que contiene la derivada del estado, representando el cambio en las coordenadas (x, y) y en la orientación.
    """
    return np.array([u[0] * np.cos(xi[2]),
                     u[0] * np.sin(xi[2]),
                     u[1]])

def calcularVertices(x, y, phi):
    """
    Calcula las coordenadas de los vértices de un triángulo equilátero centrado en un punto dado con una orientación específica.

    Parámetros:
    x (float): La coordenada x del centro del triángulo.
    y (float): La coordenada y del centro del triángulo.
    phi (float): El ángulo de orientación del triángulo en radianes.

    Retorno:
    ndarray: Un arreglo numpy de forma (3, 2) que contiene las coordenadas de los tres vértices del triángulo, donde cada fila representa un vértice con sus coordenadas (x, y).
    """
    return np.array([[x + 0.1 * np.cos(phi), y + 0.1 * np.sin(phi)],
                     [x + 0.1 * np.cos(phi + 2 * np.pi / 3), y + 0.1 * np.sin(phi + 2 * np.pi / 3)],
                     [x + 0.1 * np.cos(phi + 4 * np.pi / 3), y + 0.1 * np.sin(phi + 4 * np.pi / 3)]])

def bresenham(x0, y0, x1, y1):
    """
    Implementa el algoritmo de Bresenham para generar los puntos de una línea entre dos coordenadas enteras.

    Parámetros:
    x0 (int): La coordenada x del punto inicial.
    y0 (int): La coordenada y del punto inicial.
    x1 (int): La coordenada x del punto final.
    y1 (int): La coordenada y del punto final.

    Retorno:
    list: Una lista de tuplas donde cada tupla representa un punto (x, y) en la línea entre el punto inicial y el punto final.
    """
    points = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    while True:
        points.append((x0, y0))
        if x0 == x1 and y0 == y1:
            break
        e2 = err * 2
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy

    return points

def occupancy_grid(lines, resolucion):
    """
    Genera una grilla de ocupación (occupancy grid) a partir de un conjunto de líneas, marcando las celdas por donde pasan dichas líneas.

    Parámetros:
    lines (ndarray): Un arreglo numpy de forma (N, 4), donde cada fila representa una línea con las coordenadas (x1, y1, x2, y2) de sus puntos de inicio y fin.
    resolucion (float): La resolución del grid, que determina el tamaño de cada celda en las unidades del sistema de coordenadas.

    Retorno:
    ndarray: Una matriz 2D de numpy que representa el occupancy grid, donde las celdas marcadas con 1 indican las áreas ocupadas por las líneas y las celdas con 0 representan áreas libres.
    """
    # Determinar los límites del occupancy grid
    maxX = np.max(lines[:, [0, 2]])
    maxY = np.max(lines[:, [1, 3]])
    minX = np.min(lines[:, [0, 2]])
    minY = np.min(lines[:, [1, 3]])

    # Calcular el tamaño del grid
    numCols = int(np.ceil((maxX - minX) / resolucion))
    numRows = int(np.ceil((maxY - minY) / resolucion))

    # Inicializar la matriz del grid
    grid = np.zeros((numRows, numCols))

    # Iterar a través de cada línea
    for i in range(lines.shape[0]):
        # Extraer los puntos de inicio y fin de la línea
        xi = lines[i, 0]
        yi = lines[i, 1]
        xf = lines[i, 2]
        yf = lines[i, 3]

        # Convertir las coordenadas de las líneas a índices del grid
        xi_grid = int(np.floor((xi - minX) / resolucion))
        yi_grid = int(np.floor((yi - minY) / resolucion))
        xf_grid = int(np.floor((xf - minX) / resolucion))
        yf_grid = int(np.floor((yf - minY) / resolucion))

        # Usar la función bresenham para obtener los puntos de la línea
        pts = bresenham(xi_grid, yi_grid, xf_grid, yf_grid)

        # Marcar las celdas del grid por donde pasa la línea
        for (col, row) in pts:
            if 0 <= row < numRows and 0 <= col < numCols:
                grid[row, col] = 1  # Marcar la celda como ocupada

    return grid

def actualizar_senales_desde_csv(archivo_csv, senales0):
    """
    Actualiza los valores de una lista de señales a partir de un archivo CSV, utilizando un ID para identificar cada señal y asignarle un nuevo valor.

    Parámetros:
    archivo_csv (str): La ruta al archivo CSV que contiene los datos de actualización. El CSV debe tener columnas 'ID' y 'Color'.
    senales0 (list): Una lista de diccionarios, donde cada diccionario representa una señal con al menos una clave 'id' y 'valor'.

    Retorno:
    tuple: Una tupla que contiene:
        - senales (list): La lista actualizada de señales, con los valores modificados según el contenido del archivo CSV.
        - flag (bool): Un indicador que es `True` si alguna señal fue modificada, de lo contrario `False`.
    """
    # Leer el archivo CSV a un DataFrame
    datos_csv = pd.read_csv(archivo_csv)
    
    # Copiar las señales a una nueva variable para trabajar
    senales = senales0.copy()

    # Crear un diccionario para mapear IDs a índices en la estructura `senales`
    id_a_indice = {senal['id']: idx for idx, senal in enumerate(senales)}

    # Iterar sobre cada fila del CSV
    for i in range(len(datos_csv)):
        id_csv = datos_csv.loc[i, 'ID']
        color_csv = datos_csv.loc[i, 'Color']
        
        # Verificar si el ID existe en el diccionario
        if id_csv in id_a_indice:
            # Obtener el índice directamente del diccionario
            indice = id_a_indice[id_csv]
            
            # Actualizar el valor de 'valor' con el color del CSV
            senales[indice]['valor'] = color_csv

    # Verificar si se ha modificado `senales` respecto a `senales0`
    flag = senales != senales0
    
    return senales, flag

def hay_semaforo_cerca(x, y, senales):
    """
    Verifica si hay un semáforo cercano a una posición dada y retorna información sobre el semáforo encontrado.

    Parámetros:
    x (float): La coordenada x de la posición a verificar.
    y (float): La coordenada y de la posición a verificar.
    senales (list): Una lista de diccionarios que representan señales, donde cada diccionario debe contener al menos las claves 'tipo', 'activo', 'posicion', 'id' y 'valor'.

    Retorno:
    tuple: Una tupla que contiene:
        - flag (bool): `True` si hay un semáforo cercano dentro del umbral de distancia, `False` en caso contrario.
        - id_senal (int): El identificador del semáforo más cercano si se encuentra, `0` si no hay semáforo cercano.
        - color (str): El color del semáforo encontrado, o `'Verde'` si no hay semáforo cercano.
    """
    flag = False
    umbral_distancia = 0.08
    id_senal = 0  # Inicializar id_senal en 0, que indica ninguna señal detectada
    color = 'Verde'
    
    for senal in senales:
        if (senal['tipo'] == 'semaforo') and (senal['activo']):
            distancia = np.sqrt((senal['posicion'][0] - x)**2 + (senal['posicion'][1] - y)**2)
            if distancia < umbral_distancia:
                flag = True
                id_senal = senal['id']
                color = senal['valor']
                break
    
    return flag, id_senal, color

def hay_senal_de_bajar_velocidad(x, y, senales):
    """
    Determina si la posición dada (x, y) está dentro del área de influencia de alguna señal de "bajar velocidad" y devuelve el factor de reducción de velocidad correspondiente.

    Parámetros:
    x (float): La coordenada x de la posición a verificar.
    y (float): La coordenada y de la posición a verificar.
    senales (list): Una lista de diccionarios que representan señales, donde cada diccionario contiene al menos las claves 'activo', 'tipo', 'posicion', y 'valor'. La clave 'posicion' debe ser un arreglo numpy que define los límites del área de la señal.

    Retorno:
    float: El factor de reducción de velocidad correspondiente a la señal activa más restrictiva en la posición dada. Si no hay ninguna señal de "bajar velocidad" en la posición, devuelve 1.
    """
    # Inicializar factorVelocidad como infinito para facilitar la búsqueda del mínimo
    factor_velocidad = float('inf')
    # Inicializar 'dentro' como False, asumiendo que (x, y) no está dentro de ninguna señal inicialmente
    dentro = False

    # Iterar sobre cada señal en la lista de senales
    for senal in senales:
        # Verificar si la señal está activa y es de tipo 'bajarVelocidad'
        if senal['activo'] and senal['tipo'] == 'bajarVelocidad':
            # Convertir la posición en un array de Numpy
            posicion = np.array(senal['posicion'])
            # Obtener los límites de la señal
            x_min = np.min(posicion[:, 0])
            x_max = np.max(posicion[:, 0])
            y_min = np.min(posicion[:, 1])
            y_max = np.max(posicion[:, 1])

            # Verificar si (x, y) está dentro de la señal
            if x_min <= x <= x_max and y_min <= y <= y_max:
                if factor_velocidad > senal['valor']:
                    factor_velocidad = senal['valor']

    # Si después de revisar todas las señales, 'factor_velocidad' sigue siendo infinito,
    # significa que (x, y) no estaba dentro de ninguna señal de 'bajarVelocidad'.
    # Por lo tanto, establecer factor_velocidad a 1 y 'dentro' permanece como estaba.
    if factor_velocidad == float('inf'):
        factor_velocidad = 1

    return factor_velocidad


def hay_senal_de_alto_cerca(x, y, senales):
    """
    Verifica si hay una señal de alto cercana a una posición dada y devuelve información sobre la señal encontrada.

    Parámetros:
    x (float): La coordenada x de la posición a verificar.
    y (float): La coordenada y de la posición a verificar.
    senales (list): Una lista de diccionarios que representan señales, donde cada diccionario debe contener al menos las claves 'tipo', 'activo', 'posicion', 'id' y 'valor'.

    Retorno:
    tuple: Una tupla que contiene:
        - flag (bool): `True` si hay una señal de alto cercana dentro del umbral de distancia, `False` en caso contrario.
        - id_senal (int): El identificador de la señal de alto más cercana si se encuentra, `0` si no hay ninguna señal cercana.
        - time_alto (float): El tiempo asociado a la señal de alto encontrada, `0` si no hay ninguna señal cercana.
    """
    flag = False
    umbral_distancia = 0.06
    id_senal = 0  # Inicializar id_senal en 0, que indica ninguna señal detectada
    time_alto = 0
    
    for senal in senales:
        if (senal['tipo'] == 'alto') and (senal['activo']):
            distancia = np.sqrt((senal['posicion'][0] - x)**2 + (senal['posicion'][1] - y)**2)
            if distancia < umbral_distancia:
                flag = True
                id_senal = senal['id']
                time_alto = senal['valor']
                break
    
    return flag, id_senal, time_alto

def verifica_alto(filename):
    """
    Verifica si la palabra 'alto' está presente en el contenido de un archivo dado.

    Parámetros:
    filename (str): La ruta al archivo que se desea verificar.

    Retorno:
    bool: `False` si la palabra 'alto' se encuentra en el contenido del archivo (sin importar mayúsculas o minúsculas), `True` en caso contrario.

    Excepciones:
    Exception: Se lanza si no se puede abrir el archivo debido a un error de E/S (IOError).
    """
    try:
        with open(filename, 'r') as file:
            contenido = file.read()
    except IOError:
        raise Exception('No se pudo abrir el archivo')
    
    # Busca la palabra 'alto' en el contenido
    if 'alto' in contenido.lower():
        return False
    else:
        return True


def evitar_obstaculos(robot, pos_obstaculos, lines, dt):
    """
    Determina y ajusta la trayectoria del robot para evitar obstáculos y encontrar una nueva ruta segura si es necesario.

    Parámetros:
    robot (object): Un objeto que representa al robot, el cual debe tener atributos como `xi`, `path`, `tramo`, `PID`, `newpath`, `v_mean`, `v_max` y `diferencial`.
    pos_obstaculos (list): Una lista de posiciones de obstáculos, donde cada posición es un arreglo numpy que representa las coordenadas (x, y) del obstáculo.
    lines (list): No utilizado en esta implementación, pero representa un conjunto de líneas relacionadas con el entorno (potencialmente para futuros usos).
    dt (float): El paso de tiempo utilizado para los cálculos de velocidad del robot.

    Retorno:
    bool: `True` si se detectó un obstáculo y fue necesario recalcular la trayectoria, `False` si no se detectaron obstáculos.
    """
    def distancia(p1, p2):
        return np.sqrt(np.sum((p1 - p2)**2))

    def esta_en_obstaculo(punto, obstaculos):
        for obs in obstaculos:
            if distancia(punto, obs) < 0.26 / 2:
                return True
        return False

    def interpolar_puntos(p1, p2, distancia_max):
        vector = p2 - p1
        longitud = np.sqrt(np.sum(vector**2))
        num_puntos = int(np.ceil(longitud / distancia_max))
        return np.linspace(p1, p2, num_puntos)

    # Paso 1: Evaluar los obstáculos
    punto_futuro = robot.path[robot.tramo, :]
    if esta_en_obstaculo(punto_futuro, pos_obstaculos):
        pos_inicial = np.array([robot.xi[0], robot.xi[1]])
        FlagObstaculo = True
    else:
        return False

    # Paso 3: Evaluar tramo
    pos_deseado = None
    for i in range(robot.tramo + 1, robot.path.shape[0]):
        if not esta_en_obstaculo(robot.path[i, :], pos_obstaculos):
            pos_deseado = robot.path[i, :]
            robot.tramo_deseado = i
            break

    if pos_deseado is None:
        return False

    # Paso 4: Determinar el new_path
    new_path = []
    puntos_interpolados = interpolar_puntos(pos_inicial, pos_deseado, 0.0003)

    # Filtra los puntos interpolados para que estén dentro de los límites y no pasen por obstáculos
    for punto in puntos_interpolados:
        if 0.25 <= punto[0] <= 3.55 and 0.25 <= punto[1] <= 4.55 and not esta_en_obstaculo(punto, pos_obstaculos):
            new_path.append(punto)

    # Si algún punto está en un obstáculo, usa un algoritmo de pathfinding
    if len(new_path) != len(puntos_interpolados):
        new_path = []
        # Crear un grid de ocupación con menor precisión
        x = np.linspace(0.25, 3.55, int((3.55 - 0.25) / 0.1) + 1)
        y = np.linspace(0.25, 4.55, int((4.55 - 0.25) / 0.1) + 1)
        grid_x, grid_y = np.meshgrid(x, y)
        grid = np.c_[grid_x.ravel(), grid_y.ravel()]

        # Crear un grid de ocupación, marcando los puntos en los obstáculos como inválidos
        valid_points = []
        for punto in grid:
            if not esta_en_obstaculo(punto, pos_obstaculos):
                valid_points.append(punto)
        valid_points = np.array(valid_points)

        # Implementar Dijkstra u otro algoritmo de pathfinding
        import networkx as nx

        G = nx.Graph()
        for punto in valid_points:
            vecinos = valid_points[np.linalg.norm(valid_points - punto, axis=1) < 0.014]  # Aumentar el umbral de vecinos para el nuevo grid
            for vecino in vecinos:
                G.add_edge(tuple(punto), tuple(vecino), weight=distancia(punto, vecino))

        start = tuple(pos_inicial)
        end = tuple(pos_deseado)
        try:
            path = nx.shortest_path(G, source=start, target=end, weight='weight')
            new_path = np.array(path)
        except nx.NetworkXNoPath:
            # Manejar el caso donde no se encuentra un camino válido
            pass
    print(new_path)
    # Si new_path está vacío, puede significar que no se encontró una trayectoria válida. Puedes manejar este caso según sea necesario.
    if new_path.size == 0:
        # Manejar el caso donde no se encontró una trayectoria válida
        pass
    else:
        # Ajustar new_path para que la distancia entre puntos sea 0.0003
        final_new_path = []
        for i in range(len(new_path) - 1):
            puntos_interpolados = interpolar_puntos(new_path[i], new_path[i + 1], 0.0003)
            for punto in puntos_interpolados:
                if 0.25 <= punto[0] <= 3.55 and 0.25 <= punto[1] <= 4.55:
                    final_new_path.append(punto)

        new_path = np.array(final_new_path)

    new_path = np.array(new_path)

    # Si new_path está vacío, puede significar que no se encontró una trayectoria válida. Puedes manejar este caso según sea necesario.
    if new_path.size == 0:
        # Manejar el caso donde no se encontró una trayectoria válida
        pass


    # Paso 5: Ordenar nuevo path
    robot.newpath = new_path
    print(new_path)
    if robot.PID:
        robot.diferencial = 50
    else:
        deltas = np.diff(new_path, axis=0)
        distancias = np.sqrt(np.sum(deltas**2, axis=1))
        robot.v_mean = np.mean(distancias) / dt
        robot.diferencial = round(robot.v_max / robot.v_mean)
    robot.tramo = robot.diferencial

    return FlagObstaculo


def robotat_3pi_connect(agent_id):
    """
    Conecta con un robot específico utilizando su ID y devuelve un diccionario con la información del robot y la conexión establecida.

    Parámetros:
    agent_id (int): El identificador del agente (robot) al cual se desea conectar. Debe estar en el rango de 0 a 19.

    Retorno:
    dict: Un diccionario que contiene la información del robot, incluyendo:
        - 'id' (int): El ID del robot.
        - 'ip' (str): La dirección IP del robot.
        - 'port' (int): El puerto utilizado para la conexión (siempre 9090).
        - 'tcpsock' (socket or None): El objeto socket para la conexión TCP si se establece correctamente, o `None` si no se puede conectar.
    
    Excepciones:
    ValueError: Se lanza si el `agent_id` está fuera del rango permitido (0-19).
    """
    agent_id = round(agent_id)
    if agent_id < 0 or agent_id > 19:
        raise ValueError('Invalid agent ID. Allowed IDs: 0 - 19.')
    
    robot = {'id': agent_id}

    if agent_id > 9:
        ip = '192.168.50.1'
    else:
        ip = '192.168.50.10'

    ip = ip + str(agent_id)
    robot['ip'] = ip
    robot['port'] = 9090

    try:
        tcpsock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        tcpsock.connect((robot['ip'], robot['port']))
        robot['tcpsock'] = tcpsock
    except Exception as e:
        print('ERROR: Could not connect to the robot.')
        robot['tcpsock'] = None

    return robot

def robotat_3pi_disconnect(robot):
    """
    Desconecta el robot específico cerrando la conexión TCP y deteniendo cualquier acción en curso.

    Parámetros:
    robot (dict): Un diccionario que contiene la información del robot, incluyendo el objeto socket en 'tcpsock' que representa la conexión TCP activa.

    Retorno:
    None: La función no retorna ningún valor, pero cierra la conexión y detiene el robot.

    Acciones:
    - Detiene cualquier acción en curso del robot utilizando `robotat_3pi_force_stop`.
    - Cierra el socket TCP para desconectar el robot.
    - Imprime un mensaje indicando que se ha desconectado del robot.
    """
    robotat_3pi_force_stop(robot)
    robot['tcpsock'].close()
    print('Disconnected from robot.')

def robotat_3pi_force_stop(robot):
    """
    Envía un comando al robot para detener ambos motores inmediatamente.

    Parámetros:
    robot (dict): Un diccionario que contiene la información del robot, incluyendo el objeto socket en 'tcpsock' que representa la conexión TCP activa.

    Retorno:
    None: La función no retorna ningún valor, pero envía un comando a través del socket para detener los motores.
    """
    dphiL = 0
    dphiR = 0

    # Encode to a simple CBOR array
    cbormsg = bytearray(11)
    cbormsg[0] = 130  # 82 = array(2)
    cbormsg[1] = 250  # FA = single-precision float
    cbormsg[2:6] = struct.pack('>f', dphiL)
    cbormsg[6] = 250  # FA = single-precision float
    cbormsg[7:11] = struct.pack('>f', dphiR)
    
    robot['tcpsock'].sendall(cbormsg)

def robotat_3pi_set_wheel_velocities(robot, dphiL, dphiR):
    """
    Establece las velocidades angulares de las ruedas del robot, asegurándose de que los valores estén dentro del rango permitido.

    Parámetros:
    robot (dict): Un diccionario que contiene la información del robot, incluyendo el objeto socket en 'tcpsock' que representa la conexión TCP activa.
    dphiL (float): La velocidad angular deseada para la rueda izquierda en revoluciones por minuto (rpm).
    dphiR (float): La velocidad angular deseada para la rueda derecha en revoluciones por minuto (rpm).

    Retorno:
    None: La función no retorna ningún valor, pero envía un comando codificado al robot para establecer las velocidades de las ruedas.

    Advertencias:
    - Si `dphiL` o `dphiR` superan los límites permitidos (800 rpm y -800 rpm), se ajustan al límite y se muestra una advertencia.
    """
    wheel_maxvel_rpm = 800  # 850
    wheel_minvel_rpm = -800  # -850
    
    if dphiL > wheel_maxvel_rpm:
        print(f'Warning: Left wheel speed saturated to {wheel_maxvel_rpm} rpm')
        dphiL = wheel_maxvel_rpm

    if dphiR > wheel_maxvel_rpm:
        print(f'Warning: Right wheel speed saturated to {wheel_maxvel_rpm} rpm')
        dphiR = wheel_maxvel_rpm

    if dphiL < wheel_minvel_rpm:
        print(f'Warning: Left wheel speed saturated to {wheel_minvel_rpm} rpm')
        dphiL = wheel_minvel_rpm

    if dphiR < wheel_minvel_rpm:
        print(f'Warning: Right wheel speed saturated to {wheel_minvel_rpm} rpm')
        dphiR = wheel_minvel_rpm

    # Encode to a simple CBOR array
    cbormsg = bytearray(11)
    cbormsg[0] = 130  # 82 = array(2)
    cbormsg[1] = 250  # FA = single-precision float
    cbormsg[2:6] = struct.pack('>f', dphiL)
    cbormsg[6] = 250  # FA = single-precision float
    cbormsg[7:11] = struct.pack('>f', dphiR)
    
    robot['tcpsock'].sendall(cbormsg)

def robotat_connect():
    """
    Establece una conexión TCP con el servidor Robotat y devuelve el objeto de conexión.

    Parámetros:
    None

    Retorno:
    socket or None: Devuelve el objeto socket que representa la conexión TCP si la conexión es exitosa, o `None` si no se pudo establecer la conexión.
    
    Acciones:
    - Intenta conectar con el servidor Robotat en la dirección IP '192.168.50.200' y puerto 1883.
    - Imprime un mensaje de error si no se puede establecer la conexión.
    """
    ip = '192.168.50.200'
    port = 1883
    try:
        tcp_obj = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        tcp_obj.connect((ip, port))
        return tcp_obj
    except Exception as e:
        print('ERROR: Could not connect to Robotat server.')
        return None

def robotat_disconnect(tcp_obj):
    """
    Desconecta el socket del servidor Robotat enviando un comando de salida y cerrando la conexión.

    Parámetros:
    tcp_obj (socket): El objeto socket que representa la conexión TCP establecida con el servidor Robotat.

    Retorno:
    None: La función no retorna ningún valor, pero cierra la conexión y maneja posibles errores al desconectar.

    Acciones:
    - Envía el comando 'EXIT' al servidor para indicar el cierre de la conexión.
    - Cierra el socket TCP y maneja cualquier excepción que ocurra durante el proceso de desconexión.
    """
    try:
        tcp_obj.send(b'EXIT')
        tcp_obj.close()
        print('Disconnected from Robotat server.')
    except Exception as e:
        print('Error while disconnecting:', e)

def robotat_get_pose0(tcp_obj, agent_id):
    """
    Obtiene la posición y orientación de los agentes especificados desde el servidor Robotat.

    Parámetros:
    tcp_obj (socket): El objeto socket que representa la conexión TCP establecida con el servidor Robotat.
    agent_id (list or int): Identificador o lista de identificadores de los agentes cuyos datos de pose se solicitan.

    Retorno:
    ndarray or None: Un arreglo numpy de forma (n, 7), donde cada fila representa la pose de un agente en términos de posición y orientación. Devuelve `None` si ocurre un error o si se recibe una respuesta vacía.

    Acciones:
    - Envía una solicitud al servidor en formato JSON para obtener la pose de los agentes.
    - Recibe y decodifica la respuesta en formato JSON.
    - Procesa la respuesta para extraer los datos de pose y los organiza en un arreglo numpy.
    - Maneja cualquier excepción que ocurra durante el proceso de solicitud y respuesta.
    """
    try:
        # Clear any existing data
        #probar
        tcp_obj.settimeout(0.01)
        try:
            tcp_obj.recv(4096)
        except:
            pass
        #tcp_obj.recv(2048)
        #tcp_obj.recv(1024)
        tcp_obj.settimeout(None)
        # Prepare the request payload
        request_payload = {
            "dst": 1,   # DST_ROBOTAT
            "cmd": 1,   # CMD_GET_POSE
            "pld": agent_id
        }

        # Send the request as JSON-encoded data
        tcp_obj.send(json.dumps(request_payload).encode())

        # Receive the response
        response_data = tcp_obj.recv(4096)  # Adjust buffer size as needed

        # Decode and process the response
        if response_data:
            pose_data = json.loads(response_data)
            #new experiment
            n = len(agent_id)
            pose = np.array(pose_data).reshape(n,7)
            return pose
        else:
            print("Received empty response from server.")
            return None
    except Exception as e:
        print("An error occurred:", e)
        return None

def quat2eul(position_quaternion_array, euler_angles_order):
    """
    Convierte un arreglo de posiciones y cuaterniones a posiciones y ángulos de Euler en el orden especificado.

    Parámetros:
    position_quaternion_array (ndarray): Un arreglo numpy de forma (N, 7), donde cada fila representa la posición (x, y, z) y el cuaternión (w, x, y, z).
    euler_angles_order (str): El orden de los ángulos de Euler a convertir, especificado como una cadena (por ejemplo, 'xyz', 'zyx').

    Retorno:
    ndarray: Un nuevo arreglo numpy de forma (N, 6), donde cada fila contiene la posición (x, y, z) y los correspondientes ángulos de Euler (en grados) para el cuaternión dado.
    """
    # Ensure the input is a NumPy array
    position_quaternion_array = np.array(position_quaternion_array)

    # Extract position and quaternion components for all rows
    positions = position_quaternion_array[:, :3]
    quaternions = position_quaternion_array[:, 3:]

    # Roll the quaternions to change their order
    quaternions_eq = np.roll(quaternions, -1, axis=1)

    # Convert quaternions to Euler angles for all rows
    rotations = R.from_quat(quaternions_eq)
    euler_angles = rotations.as_euler(euler_angles_order, degrees=True)

    # Create a new array with positions and Euler angles
    new_array = np.hstack((positions, euler_angles))

    return new_array

def robotat_get_pose(tcp_obj, agent_id, secuencia):
    """
    Obtiene la pose del agente desde el servidor Robotat y la convierte de cuaternión a ángulos de Euler.

    Parámetros:
    tcp_obj (socket): El objeto socket que representa la conexión TCP establecida con el servidor Robotat.
    agent_id (list or int): Identificador o lista de identificadores de los agentes cuyos datos de pose se solicitan.
    secuencia (str): El orden de los ángulos de Euler para la conversión (por ejemplo, 'xyz', 'zyx').

    Retorno:
    ndarray: Un arreglo numpy de forma (N, 6), donde cada fila contiene la posición (x, y, z) y los correspondientes ángulos de Euler (en grados) para el cuaternión obtenido del agente.
    """
    new_array = quat2eul(robotat_get_pose0(tcp_obj, agent_id),secuencia)
    return new_array

def wrap_to_pi(angles):
    """
    Envuelve los ángulos dados en radianes al rango [-π, π].

    Parámetros:
    angles (ndarray or float): Un ángulo o un arreglo de ángulos en radianes que se desea envolver dentro del rango [-π, π].

    Retorno:
    ndarray or float: Los ángulos envueltos en el rango [-π, π], con el mismo formato (escalar o arreglo) que el parámetro de entrada.
    """
    return (angles + np.pi) % (2 * np.pi) - np.pi

def obtener_pose(robotat, numMarker, delta):
    """
    Calcula la posición y orientación de un marcador en el sistema de referencia del mapa, aplicando una transformación desde el sistema de referencia del robot.

    Parámetros:
    robotat (object): Objeto que representa la conexión con el sistema Robotat para obtener las poses de los marcadores.
    numMarker (list): Lista de identificadores de los marcadores cuya pose se desea obtener.
    delta (float): Valor adicional para ajustar el ángulo de orientación del marcador.

    Retorno:
    tuple: Una tupla que contiene:
        - x_objeto (float): La coordenada x del objeto en el sistema de referencia del mapa.
        - y_objeto (float): La coordenada y del objeto en el sistema de referencia del mapa.
        - phi (float): El ángulo de orientación del objeto en grados, ajustado con el valor de `delta`.
    """
    x_m1, y_m1 = 2.9, 0.5
    x_m2, y_m2 = 2, 0.5
    x_m3, y_m3 = 2.9, 1.4

    M1 = np.array([x_m1, y_m1])
    M2 = np.array([x_m2, y_m2])
    M3 = np.array([x_m3, y_m3])

    VO_M = (M1 + M2 + M3) / 3

    VR_M = np.array([[-1, 0], [0, -1]])
    VT_M = np.vstack((np.hstack((VR_M, VO_M.reshape(2, 1))), [0, 0, 1]))

    MT_V = np.linalg.inv(VT_M)

    x_i = robotat_get_pose(robotat, [23], "ZYX")
    x_i = x_i[0]
    theta = atan2(sin(radians(x_i[3])), cos(radians(x_i[3])))

    OT_M = np.array([
        [cos(theta), -sin(theta), x_i[0]],
        [sin(theta), cos(theta), x_i[1]],
        [0, 0, 1]
    ])

    OT_V = np.dot(OT_M, MT_V)

    VT_O = np.linalg.inv(OT_V)

    objeto = robotat_get_pose(robotat, numMarker, "XYZ")
    objeto = objeto[0]
    pos_objeto = objeto[:2]

    objeto_mapa = np.dot(VT_O, np.append(pos_objeto, 1))

    x_objeto = objeto_mapa[0]
    y_objeto = objeto_mapa[1]

    phi1 = degrees(wrap_to_pi(atan2(sin(radians(objeto[5])), cos(radians(objeto[5])))))
    phi2 = phi1 + delta
    phi = degrees(atan2(sin(radians(phi2)), cos(radians(phi2))))

    return x_objeto, y_objeto, phi



def crear_grafo2(lineas, rectangulos_excluir):
    """
    Crea un grafo a partir de líneas especificadas, excluyendo las que se encuentran dentro de ciertos rectángulos, y conecta nodos adicionales si están lo suficientemente cerca dentro de áreas específicas.

    Parámetros:
    lineas (ndarray): Un arreglo numpy de forma (N, 4), donde cada fila representa una línea con coordenadas (x1, y1, x2, y2) de los puntos de inicio y fin.
    rectangulos_excluir (list or ndarray): Una lista o arreglo numpy que contiene los rectángulos que deben excluirse al crear el grafo. Cada rectángulo se define por las coordenadas de dos esquinas opuestas (x1, y1, x2, y2).

    Retorno:
    tuple: Una tupla que contiene:
        - G (networkx.Graph): Un grafo no dirigido creado a partir de las líneas especificadas, excluyendo aquellas dentro de los rectángulos y conectando nodos dentro de ciertas áreas.
        - node_coordinates (ndarray): Un arreglo numpy que contiene las coordenadas de los nodos únicos presentes en el grafo.
    """
    # Definir los rectángulos y la distancia máxima para la conectividad especial
    rectangulos_conectar = np.array([
        [0.50, 0.00, 2.98, 0.40],
        [2.10, 0.50, 2.53, 0.80],
        [0.50, 0.87, 2.00, 1.30],
        [2.53, 0.80, 2.99, 1.24],
        [3.04, 1.34, 3.41, 2.23],
        [3.03, 2.60, 3.41, 3.21],
        [0.82, 2.02, 1.75, 2.40],
        [0.02, 0.44, 0.40, 0.79],
    ])
    max_distancia = 0.25  # Distancia máxima para conectar nodos

    # Filtrar líneas que están dentro de los rectángulos especificados para excluir
    lineas_filtradas = lineas.copy()
    for rect in rectangulos_excluir:
        min_x = min(rect[0], rect[2])
        max_x = max(rect[0], rect[2])
        min_y = min(rect[1], rect[3])
        max_y = max(rect[1], rect[3])

        # Excluir líneas con cualquier punto dentro del rectángulo
        lineas_filtradas = lineas_filtradas[~(
            (lineas_filtradas[:, 0] >= min_x) & (lineas_filtradas[:, 0] <= max_x) &
            (lineas_filtradas[:, 1] >= min_y) & (lineas_filtradas[:, 1] <= max_y) |
            (lineas_filtradas[:, 2] >= min_x) & (lineas_filtradas[:, 2] <= max_x) &
            (lineas_filtradas[:, 3] >= min_y) & (lineas_filtradas[:, 3] <= max_y)
        )]

    # Extraer los puntos únicos de las líneas filtradas
    puntos_unicos = np.unique(np.vstack((lineas_filtradas[:, :2], lineas_filtradas[:, 2:])), axis=0)
    n = len(puntos_unicos)

    # Mapa de coordenadas a índices de nodos
    coordenadas_a_nodos = {tuple(coord): i for i, coord in enumerate(puntos_unicos)}

    # Preparar vectores de nodos inicial y final
    s = []
    t = []

    # Conectar nodos que están en las líneas
    for linea in lineas_filtradas:
        s.append(coordenadas_a_nodos[tuple(linea[:2])])
        t.append(coordenadas_a_nodos[tuple(linea[2:])])

    # Conectar nodos dentro de cada rectángulo si están lo suficientemente cerca
    for rect in rectangulos_conectar:
        indices_dentro = np.where(
            (puntos_unicos[:, 0] >= min(rect[0], rect[2])) & (puntos_unicos[:, 0] <= max(rect[0], rect[2])) &
            (puntos_unicos[:, 1] >= min(rect[1], rect[3])) & (puntos_unicos[:, 1] <= max(rect[1], rect[3]))
        )[0]
        for i in range(len(indices_dentro)):
            for j in range(i + 1, len(indices_dentro)):
                if np.linalg.norm(puntos_unicos[indices_dentro[i]] - puntos_unicos[indices_dentro[j]]) <= max_distancia:
                    s.append(indices_dentro[i])
                    t.append(indices_dentro[j])

    # Crear el grafo no dirigido
    G = nx.Graph()
    G.add_edges_from(zip(s, t))
    node_coordinates = puntos_unicos
    return G, node_coordinates

def actualizar_semaforo(carros, ax, canvas):
    """
    Actualiza los estados de los semáforos para los carros en la simulación, basándose en un archivo CSV que contiene los colores de los semáforos, y visualiza los cambios en la gráfica.

    Parámetros:
    carros (list): Lista de objetos `carro`, donde cada objeto tiene un atributo `senales` que contiene la información sobre los semáforos asociados al carro.
    ax (matplotlib.axes._axes.Axes): El objeto de ejes de matplotlib en el cual se dibujan los semáforos.
    canvas (matplotlib.backends.backend_agg.FigureCanvasAgg): El lienzo de matplotlib que se usa para actualizar la visualización.

    Retorno:
    None: La función no retorna ningún valor, pero actualiza el estado de los semáforos de los carros y los visualiza en el gráfico.
    
    Acciones:
    - Lee un archivo CSV llamado 'semaforos.csv' para obtener los estados de los semáforos (ID y color).
    - Actualiza los estados de los semáforos para cada carro en la lista de `carros` si el estado ha cambiado.
    - Dibuja un círculo que representa el semáforo con el color actualizado en la gráfica `ax`.
    """
    caso = 0
    datos_csv = pd.read_csv('semaforos.csv')

    for carro in carros:
        # senales = carro.senales.copy()
        flag = False
        senales = carro.senales
        id_a_indice = {senal['id']: idx for idx, senal in enumerate(senales)}
        
        for i in range(len(datos_csv)):
            id_csv = datos_csv.loc[i, 'ID']
            color_csv = datos_csv.loc[i, 'Color']

        if id_csv in id_a_indice:
            indice = id_a_indice[id_csv]
            colorAnt = senales[indice]['valor']
            senales[indice]['valor'] = color_csv
            if colorAnt != color_csv:
                flag = True
                color_senal = color_csv.lower()
                if color_senal == "verde":
                    color = 'green'
                elif color_senal == "amarillo":
                    color = 'yellow'
                else:
                    color = 'red'
                # Reducir el tamaño del círculo
                ax.add_patch(Circle(senales[indice]['posicion'], 0.05, color=color))
            
        carro.senales = senales

