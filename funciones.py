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
    diffs = node_coordinates - np.array([x, y])
    dists = np.linalg.norm(diffs, axis=1)
    return np.argmin(dists)

def calcular_ruta(G, node_coordinates, Ax, Ay, Bx, By):
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
    return np.array([u[0] * np.cos(xi[2]),
                     u[0] * np.sin(xi[2]),
                     u[1]])

def calcularVertices(x, y, phi):
    return np.array([[x + 0.1 * np.cos(phi), y + 0.1 * np.sin(phi)],
                     [x + 0.1 * np.cos(phi + 2 * np.pi / 3), y + 0.1 * np.sin(phi + 2 * np.pi / 3)],
                     [x + 0.1 * np.cos(phi + 4 * np.pi / 3), y + 0.1 * np.sin(phi + 4 * np.pi / 3)]])

def bresenham(x0, y0, x1, y1):
    """Implementación del algoritmo de Bresenham para generar los puntos de una línea."""
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
    Esta función abre y lee un archivo de texto.
    Devuelve False si encuentra la palabra 'alto', de lo contrario True.
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
    robotat_3pi_force_stop(robot)
    robot['tcpsock'].close()
    print('Disconnected from robot.')

def robotat_3pi_force_stop(robot):
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
    try:
        tcp_obj.send(b'EXIT')
        tcp_obj.close()
        print('Disconnected from Robotat server.')
    except Exception as e:
        print('Error while disconnecting:', e)

def robotat_get_pose0(tcp_obj, agent_id):
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
    new_array = quat2eul(robotat_get_pose0(tcp_obj, agent_id),secuencia)
    return new_array

def wrap_to_pi(angles):
    return (angles + np.pi) % (2 * np.pi) - np.pi

def obtener_pose(robotat, numMarker, delta):
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

