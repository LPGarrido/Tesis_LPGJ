import time
from carro import *
import concurrent.futures


lines = pd.read_csv('Lines.csv').values  # Leer las líneas del archivo CSV
G, node_coordinates = crear_grafo(lines) # Crear el grafo
Mapa = pd.read_csv('Mapa.csv').values

experimental = False

PID = False
vMax = 0.10


dt = 0.1
t0 = 0
tf = 1000
K = int((tf - t0) / dt)
tiempo = np.arange(t0, tf + dt, dt)

if experimental:
    robotat = robotat_connect()
else:
    robotat = None

# Carro 1 
noRobot1 = 2
delta1 = 141.9342
if experimental:
    robot1 = robotat_3pi_connect(noRobot1)
    x_inicial, y_inicial, phi0 = obtener_pose(robotat, [noRobot1], delta1)
    x0 = np.array([x_inicial, y_inicial])
    phi_inicial = np.deg2rad(phi0)
else:
    robot1 = None
    x0 = node_coordinates[find_closest_node(node_coordinates, 1.3, 1),:]
    phi_inicial = np.deg2rad(0.0)

xi1 = np.array([x0[0], x0[1], phi_inicial])

A1 = [x0[0], x0[1]]
B1 = [3.15,1.59]
path1, _ = calcular_ruta(G, node_coordinates, A1[0], A1[1], B1[0], B1[1])


grid0 = occupancy_grid(Mapa, 0.01)
# Configuración de la figura
plt.figure()  # Configuración de la figura
extent = [0, 3.8, 0, 4.8]  # Mostrar la imagen de la cuadrícula de ocupación
plt.imshow(grid0, cmap=plt.cm.binary, origin='lower', extent=extent) # cmap=plt.cm.gray
plt.plot(path1[:, 0], path1[:, 1], 'r--', linewidth=1)  # Trazar el camino | Línea roja discontinua
# Inicialización de la figura y ejes
vehiculoPath1, = plt.plot(xi1[0], xi1[1], 'b-', linewidth=2)
vertices = calcularVertices(xi1[0], xi1[1], xi1[2])
hRobot1 = plt.fill(vertices[:, 0], vertices[:, 1], 'b')
plt.axis('equal')

numParqueo1 = 2

carro1 = carro(vMax, xi1, dt, K, tiempo, path1, experimental, delta1, numParqueo1, robot1, PID, vehiculoPath1, hRobot1, robotat, G, node_coordinates)
carros = [carro1]
k1_final = 0
k2_final = 0

pos_obstaculos = np.array([[8.0,8.0]])
for k in range(K):
    tic = time.time()
    # carro1.method1(k, dt, K, carros, pos_obstaculos, lines)
    if carro1.experimental:
        xpos, ypos, phi0 = obtener_pose(carro1.robotat, [carro1.no_robot], carro1.delta)
        phi = np.deg2rad(phi0)
        carro1.xi[0] = xpos
        carro1.xi[1] = ypos
        carro1.xi[2] = phi

    carro1.method1(k,carros)

    correr = verifica_alto('archivo.txt')
    if (not correr and experimental):
        robotat_3pi_force_stop(carro1.robot)
        break

    time_ejecucion = time.time() - tic
    # print(time_ejecucion)
    if time_ejecucion < dt:
        plt.pause(dt - time_ejecucion)


plt.show()


