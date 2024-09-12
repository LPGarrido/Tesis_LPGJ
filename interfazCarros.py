import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle, Rectangle
import numpy as np
import pandas as pd
import concurrent.futures
import time
import threading
import json
from carro import *  

carros = []

dt = 0.10
tf = 1000
K = int((tf) / dt)
tiempo = np.arange(0, tf + dt, dt)
timeMAX = 0
tiemposEjecucion = np.zeros((K,1))

experimental = True
if experimental:
    robotat0 = robotat_connect()
else:
    robotat0 = None

# Variable global para controlar la cancelación de la simulación
simulacion_cancelada = False
señales_mostradas = False
simulacion_en_curso = False

def cargar_senales():
    with open('senales.json', 'r') as file:
        return json.load(file)

def ocultar_senales(ax):
    global señales_mostradas
    while ax.patches:
        ax.patches[-1].remove()  # Elimina cada parche del final de la lista
    canvas.draw()
    señales_mostradas = False

def toggle_senales(ax):
    global señales_mostradas, simulacion_en_curso
    if not simulacion_en_curso:
        if señales_mostradas:
            ocultar_senales(ax)
        else:
            dibujar_senales(ax)
            señales_mostradas = True

def dibujar_senales(ax):
    senales = cargar_senales()

    for senal in senales:
        tipo = senal['tipo']
        posicion = senal['posicion']
        valor = senal['valor']

        if tipo == "bajarVelocidad":
            color = ''
            if valor <= 0.25:
                color = 'red'
            elif valor <= 0.5:
                color = 'orange'
            elif valor <= 0.75:
                color = 'yellow'
            else:
                color = 'green'

            # Crear un rectángulo usando los puntos extremos
            x_min, y_min = posicion[0]
            x_max, y_max = posicion[1]
            width = x_max - x_min
            height = y_max - y_min

            rectangulo = Rectangle((x_min, y_min), width, height, color=color, alpha=0.2)
            ax.add_patch(rectangulo)

        elif tipo == "alto":
            # Reducir el tamaño del hexágono
            hexagono = Polygon([(posicion[0] + 0.05 * np.cos(np.deg2rad(60 * i)),
                                 posicion[1] + 0.05 * np.sin(np.deg2rad(60 * i)))
                                for i in range(6)], color='red')
            ax.add_patch(hexagono)

        elif tipo == "semaforo":
            color_senal = valor.lower()
            if color_senal == "verde":
                color = 'green'
            elif color_senal == "amarillo":
                color = 'yellow'
            else:
                color = 'red'
            
            # Reducir el tamaño del círculo
            ax.add_patch(Circle(posicion, 0.05, color=color))

    obstaculos = np.array([[1.02, 0.04, 1.37, 0.23], [8,8, 9, 9]])

    for obs in obstaculos:
        x_min, y_min, x_max, y_max = obs
        width = x_max - x_min
        height = y_max - y_min

        rectangulo = Rectangle((x_min, y_min), width, height, color='black', alpha=0.2)
        ax.add_patch(rectangulo)
    
    canvas.draw()

def crear_nuevo_carro(root, ax, canvas, dt0, K0, tf, G, node_coordinates):
    def solicitar_punto(title, point_var):
        solicitud = tk.Toplevel(root)
        solicitud.title(title)
        label = ttk.Label(solicitud, text="Haga clic en la gráfica para seleccionar el punto.")
        label.pack()
        
        def obtener_punto(event):
            point_var.set((event.xdata, event.ydata))
            solicitud.destroy()
            print(f"{title} seleccionado: ({event.xdata}, {event.ydata})")
            canvas.mpl_disconnect(cid)
        
        cid = canvas.mpl_connect('button_press_event', obtener_punto)

    def guardar_datos():
        global robotat0
        no_robot = int(entry_no_robot.get())
        v_max = float(entry_v_max.get())
        experimental = entry_experimental.get() == "True"
        pid = entry_pid.get() == "True"
        num_parqueo = int(entry_num_parqueo.get())
        deltas = {
            8: 173.4735,
            5: -89.8008,
            2: 142.8531,
            10:310.0049,
            3: 183.7171
        }
        delta1 = deltas.get(no_robot,None)
        
        ventana_nuevo_carro.destroy()

        if not experimental:
            solicitar_punto("Seleccione el punto inicial", point_var_inicial)
            root.wait_variable(point_var_inicial)
            x0 = np.array(point_var_inicial.get())
            phi_inicial = np.deg2rad(0.0)
            robot1 = None
            print("no experimental")
        else:
            print("Experimental")
            x_inicial, y_inicial, phi0 = obtener_pose(robotat0, [no_robot], delta1)
            x0 = np.array([x_inicial, y_inicial])
            phi_inicial = np.deg2rad(phi0)
            robot1 = robotat_3pi_connect(no_robot)

        xi1 = np.array([x0[0], x0[1], phi_inicial])

        solicitar_punto("Seleccione el punto final", point_var_final)
        root.wait_variable(point_var_final)
        
        B = point_var_final.get()

        # Ploteo del vehículo
        vehiculo_path, = ax.plot(xi1[0], xi1[1], 'b-', linewidth=2)
        vertices = calcularVertices(xi1[0], xi1[1], xi1[2])
        h_robot = ax.fill(vertices[:, 0], vertices[:, 1], 'b')
        canvas.draw()
        
        carroAutonomo = carro(
            numero = no_robot,
            vMax=v_max,
            xi=xi1,
            dt=dt0, 
            tf=tf, 
            B=B, 
            FlagExperimental=experimental, 
            delta=delta1,
            noParqueo=num_parqueo, 
            robot=robot1, 
            FlagControlador=pid, 
            vehiculo_path=vehiculo_path, 
            h_robot=h_robot, 
            robotat=robotat0, 
            grafo=G, 
            coordenadasNodos=node_coordinates
        )
        
        # Añadir el carro a la lista global
        carros.append(carroAutonomo)
        print("Carro creado:", carroAutonomo)

    ventana_nuevo_carro = tk.Toplevel(root)
    ventana_nuevo_carro.title("Crear nuevo carro")

    ttk.Label(ventana_nuevo_carro, text="Número de robot:").pack()
    entry_no_robot = ttk.Entry(ventana_nuevo_carro)
    entry_no_robot.pack()

    ttk.Label(ventana_nuevo_carro, text="Velocidad máxima:").pack()
    entry_v_max = ttk.Entry(ventana_nuevo_carro)
    entry_v_max.pack()

    ttk.Label(ventana_nuevo_carro, text="¿Es experimental (True/False)?:").pack()
    entry_experimental = ttk.Entry(ventana_nuevo_carro)
    entry_experimental.pack()

    ttk.Label(ventana_nuevo_carro, text="¿Controlador PID (True/False)?:").pack()
    entry_pid = ttk.Entry(ventana_nuevo_carro)
    entry_pid.pack()

    ttk.Label(ventana_nuevo_carro, text="Número de parqueo:").pack()
    entry_num_parqueo = ttk.Entry(ventana_nuevo_carro)
    entry_num_parqueo.pack()

    point_var_inicial = tk.Variable()
    point_var_final = tk.Variable()

    btn_guardar = ttk.Button(ventana_nuevo_carro, text="Guardar", command=guardar_datos)
    btn_guardar.pack(pady=5)

num_terminado = 1

def iniciar_simulacion(canvas, root, label_tiempo):
    global K, carros, G, node_coordinates, dt, tiempo, simulacion_cancelada, lines

    simulacion_cancelada = False
    simulacion_en_curso = True
    
    # Función para manejar la detección de obstáculos
    def manejar_obstaculo(carro):
        if carro.obstaculoDetectado1 and (carro.caseObstaculos==0):
            carro.caseObstaculos = 1
            threading.Thread(target=carro.ruta_obstaculo2).start()

    def run_simulacion():
        global ax, canvas, num_terminado, timeMAX
        for k in range(K):

            tic = time.time()

            if simulacion_cancelada or (num_terminado == len(carros)+1):
                print("Simulación cancelada durante el ciclo.")
                print(np.max(tiemposEjecucion))
                print(np.mean(tiemposEjecucion[:k]))
                break

            actualizar_semaforo(carros,ax,canvas)

            for carro in carros:
                if carro.experimental:
                    xpos, ypos, phi0 = obtener_pose(carro.robotat, [carro.no_robot], carro.delta)
                    phi = np.deg2rad(phi0)
                    carro.xi[0] = xpos
                    carro.xi[1] = ypos
                    carro.xi[2] = phi

            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [executor.submit(carro.method1, k, carros) for carro in carros]
                concurrent.futures.wait(futures)


            # Crear hilos para ruta_obstaculo si se detecta un obstáculo
            for carro in carros:
                manejar_obstaculo(carro)
                
            correr = verifica_alto('archivo.txt')
            if not correr and any(carro.experimental for carro in carros):
                for carro in carros:
                    if carro.experimental:
                        robotat_3pi_force_stop(carro.robot)
                break
            if not correr:
                break
            
            for carro in carros:
                if carro.done2 and (not carro.guardado):
                    if carro.PID:
                        nameControl = "PID"
                    else:
                        nameControl = "Lyapunov"
                    nameArchivo1 = f"carro_{num_terminado}"+"_"+nameControl+"_posiciones.csv"
                    nameArchivo2 = f"carro_{num_terminado}"+"_"+nameControl+"_velocidades.csv"
                    np.savetxt(nameArchivo1, carro.posiciones[:carro.k_final, :], delimiter=',', fmt='%.6f')
                    np.savetxt(nameArchivo2, np.transpose(carro.U[:, :carro.k_final]), delimiter=',', fmt='%.6f')
                    carro.guardado = True
                    num_terminado = num_terminado + 1
            
            # Actualizar la gráfica en el hilo principal
            root.after(0, canvas.draw)
            # Actualizar la etiqueta de tiempo en el hilo principal
            root.after(0, lambda: label_tiempo.config(text=f"Tiempo de simulación: {tiempo[k]:.2f} s"))

            time_ejecucion = time.time() - tic
            tiemposEjecucion[k] = time_ejecucion
            if time_ejecucion < dt:
                time.sleep(dt - time_ejecucion)  

        print("Simulación terminada.")
        for carro in carros:
            carro.frenar()

    # Ejecutar la simulación en un hilo separado
    threading.Thread(target=run_simulacion).start()

def cancelar_simulacion():
    global simulacion_cancelada
    simulacion_cancelada = True

def guardar_mundo():
    if not carros:
        print("No hay carros para guardar.")
        return

    # Convertir los carros a un formato serializable
    carros_data = [carro.to_dict() for carro in carros]

    # Abrir la ventana de diálogo para guardar el archivo
    filepath = filedialog.asksaveasfilename(
        defaultextension=".json",
        filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        title="Guardar Mundo"
    )
    if not filepath:
        return

    # Guardar los datos en el archivo
    with open(filepath, 'w') as file:
        json.dump(carros_data, file, indent=4)
    print(f"Mundo guardado en {filepath}")

def cargar_mundo(ax, canvas, robotat, G, node_coordinates):
    global carros

    # Abrir la ventana de diálogo para abrir el archivo
    filepath = filedialog.askopenfilename(
        defaultextension=".json",
        filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        title="Cargar Mundo"
    )
    if not filepath:
        return

    # Cargar los datos desde el archivo
    with open(filepath, 'r') as file:
        carros_data = json.load(file)

    # Crear los objetos carro a partir de los datos
    carros = [carro.from_dict(data, ax, canvas, robotat, G, node_coordinates) for data in carros_data]
    print(f"Mundo cargado desde {filepath}")

# Configuración de la interfaz gráfica
root = tk.Tk()
root.title("Simulación vehículo autónomo a escala en la Ciudad de Guatemala")

# Crear un frame para la gráfica y los botones
frame = ttk.Frame(root)
frame.pack(fill=tk.BOTH, expand=True)

# Crear un canvas para la gráfica
fig, ax = plt.subplots()
canvas = FigureCanvasTkAgg(fig, master=frame)
canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

# Crear un frame para los botones
button_frame = ttk.Frame(frame)
button_frame.pack(side=tk.RIGHT, fill=tk.Y)

# Leer el mapa desde un archivo CSV
Mapa = pd.read_csv('Mapa.csv').values
grid0 = occupancy_grid(Mapa, 0.01)
extent = [0, 3.8, 0, 4.8]
ax.imshow(grid0, cmap=plt.cm.binary, origin='lower', extent=extent)
canvas.draw()

# Inicializar los nodos y el grafo
lines = pd.read_csv('Lines.csv').values  # Leer las líneas del archivo CSV
G, node_coordinates = crear_grafo(lines) # Crear el grafo

# Crear botones
btn_crear_carro = ttk.Button(button_frame, text="Crear nuevo carro", command=lambda: crear_nuevo_carro(root, ax, canvas, dt, K, tf, G, node_coordinates))
btn_crear_carro.pack(pady=5)

# Crear una etiqueta para mostrar el tiempo de simulación
label_tiempo = ttk.Label(button_frame, text="Tiempo de simulación: 0.00 s")
label_tiempo.pack(pady=5)

btn_iniciar_simulacion = ttk.Button(button_frame, text="Iniciar simulación", command=lambda: iniciar_simulacion(canvas, root, label_tiempo))
btn_iniciar_simulacion.pack(pady=5)

btn_cancelar_simulacion = ttk.Button(button_frame, text="Cancelar", command=cancelar_simulacion)
btn_cancelar_simulacion.pack(pady=5)

# Añadir botones para guardar y cargar el mundo
btn_guardar_mundo = ttk.Button(button_frame, text="Guardar mundo", command=guardar_mundo)
btn_guardar_mundo.pack(pady=5)

btn_cargar_mundo = ttk.Button(button_frame, text="Cargar mundo", command=lambda: cargar_mundo(ax, canvas, robotat0, G, node_coordinates))
btn_cargar_mundo.pack(pady=5)

btn_toggle_senales = ttk.Button(button_frame, text="Mostrar/Ocultar señales", command=lambda: toggle_senales(ax))
btn_toggle_senales.pack(pady=5)



# Iniciar el bucle principal de la interfaz gráfica
root.mainloop()
