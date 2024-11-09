from funciones import *


class carro:
    def __init__(self, numero, vMax, xi, dt, tf, B, FlagExperimental, delta, noParqueo, robot, FlagControlador, vehiculo_path, h_robot, robotat, grafo, coordenadasNodos):
        """
    Inicializa una instancia de un carro autónomo con todos los parámetros necesarios para la simulación.

    Parámetros:
    numero (int): Identificador único del robot.
    vMax (float): Velocidad máxima del carro.
    xi (array-like): Estado inicial del carro en forma [x, y, orientación].
    dt (float): Paso de tiempo para la simulación.
    tf (float): Tiempo final de la simulación.
    B (array-like): Coordenadas del punto objetivo (destino) del carro.
    FlagExperimental (bool): Indicador de si el carro está en modo experimental.
    delta (float): Valor de ajuste de orientación.
    noParqueo (int): Número del espacio de parqueo asignado.
    robot (object or None): Objeto del robot, si está en modo experimental (conectado a hardware).
    FlagControlador (bool): Indicador de si el carro usa un controlador PID (`True`) o Lyapunov (`False`).
    vehiculo_path (matplotlib.lines.Line2D): Objeto de línea en Matplotlib para visualizar el trayecto del carro.
    h_robot (matplotlib.patches.Polygon): Objeto de Matplotlib para visualizar la forma del carro.
    robotat (object): Objeto de conexión a Robotat.
    grafo (networkx.Graph): El grafo de caminos para la planificación de rutas.
    coordenadasNodos (ndarray): Coordenadas de los nodos del grafo.

    Retorno:
    None: Este es un constructor, por lo tanto, no retorna ningún valor, pero inicializa todos los atributos del carro para ser utilizado en la simulación.

    Acciones:
    - Carga las señales de tráfico desde un archivo JSON.
    - Establece los estados iniciales del carro, las banderas de control, los parámetros del controlador, y las condiciones de frenado.
    - Realiza la planificación global de la ruta desde el estado inicial hasta el punto objetivo utilizando el grafo dado.
    - Configura las propiedades de graficación para la visualización del carro en la simulación.
    - Inicializa estructuras para almacenar información de la simulación como errores, posiciones y controles.
    """
        self.no_robot = numero
        # ----------- Senales -------------
        with open('senales.json', 'r') as archivo_json:
            self.senales = json.load(archivo_json) 
        self.senal_inactiva = 0
        # ---------- Entradas -------------
        self.u = np.array([0, 0])
        self.wL = 0
        self.wR = 0
        # ---------- Salidas --------------
        self.xi = np.array([xi[0], xi[1], xi[2]])
        # ----------- Banderas ------------
        self.experimental = FlagExperimental
        self.done = False
        self.done2 = False
        self.FlagParqueo1 = False
        self.FlagParqueo2 = False
        self.FlagParqueo3 = False
        self.alto_cercano = False
        self.bandera = False
        self.delta = delta
        self.obstaculoDetectado1 = False
        self.banderaObs = False
        self.change = False
        # ----- Parametros de tiempo ------
        self.dt = dt
        self.K = int((tf) / dt)
        self.tf = tf
        self.tiempo = np.arange(0, tf + dt, dt)
        self.time_inicial = 0
        self.time_inicial2 = 0
        self.time_alto = 0
        self.time_fueraObs = 0
        # --------- Controlador -----------
        self.PID = FlagControlador
        # -------- Controlador PID --------
        self.alpha = 50
        self.kpO = 2
        self.kiO = 0.0001
        self.kdO = -1
        self.EO = 0
        self.eO_1 = 0
        # ----------- Errores -------------
        self.e = np.array([0, 0])
        # ------ Posiciones deseadas ------
        self.xg = 0
        self.yg = 0
        # ---------- Ruta Ideal -----------
        """ ---------------- GLOBAL PLANNING ---------------- """
        self.B = B
        pathIdeal, _ = calcular_ruta(grafo, coordenadasNodos, self.xi[0], self.xi[1], B[0], B[1])  # Ajustar según sea necesario
        self.path = pathIdeal
        """ ------------------------------------------------- """
        self.puntoFinal = self.path[-1]
        # --------- Velocidades -----------
        self.v_max = vMax
        self.v_modificada = vMax
        deltas = np.diff(self.path, axis=0)
        distancias = np.sqrt(np.sum(deltas**2, axis=1))
        self.v_mean = np.mean(distancias) / dt
        # ---- Diferencial Controlador ----
        if self.PID:
            self.diferencial = 50
        else:
            self.diferencial = round(self.v_max / self.v_mean)
        self.tramo = self.diferencial
        # ----- Elementos Graficar ---------
        self.vehiculo_path = vehiculo_path
        self.h_robot = h_robot
        # ----------- Robotat --------------
        self.robotat = robotat
        self.robot = robot
        # ------ Almacenar variables -------
        self.idx_almacenar = 0
        self.errores = np.zeros((self.K, 2))
        self.XI = np.zeros((3, self.K))
        self.U = np.zeros((2, self.K))
        self.recorrido = np.zeros((self.K, 2))
        self.posiciones = np.zeros((self.K, 4))
        self.k_final = 0
        self.guardado = False
        # ---------- Parqueo ---------------
        self.parqueos = np.array([[0.65, 0.65], [0.65 + 0.40, 0.65], [0.65 + 2*0.40, 0.65], [0.65 + 3*0.40, 0.65]])
        self.numParqueo = noParqueo
        # ---------- Obstaculos ------------
        self.obstaculos = np.array([[0.95, 0.04, 0.95+0.7, 0.23], [8,8, 9, 9]])
        self.caseObstaculos = 0
        self.pos_futuro_obs = np.array([0, 0])
        # -------- Nodos y grafo -----------
        self.G = grafo
        self.node_coordinates = coordenadasNodos
        # ------ Factor Velocidad ----------
        self.factorExtraVel = 1
        self.factorBajarVel = 1
        self.factorVelObst = 1
        # ------ Condiciones de Frenar ----- 
        self.cond1_frenar = False
        self.cond2_frenar = False
        self.cond3_frenar = False
        self.cond4_frenar = False
        self.cond5_frenar = False
        self.cond_frenar = False

    def method1(self,k,carros):
        """
        Ejecuta el ciclo de planificación de comportamiento, planificación local, y control para un carro autónomo durante un paso de la simulación.

        Parámetros:
        k (int): El índice de tiempo actual en la simulación.
        carros (list): Lista de objetos `carro` que representan otros vehículos en la simulación, utilizada para evitar colisiones.

        Retorno:
        None: La función no retorna ningún valor, pero actualiza el estado del carro, incluyendo su posición, velocidad, y comportamiento basado en el entorno.

        Acciones:
        - Planificación del comportamiento (fase 1 y 2): Calcula el punto deseado, evalúa señales de tráfico (alto, semáforo, reducir velocidad), y ajusta la velocidad.
        - Planificación local: Realiza la planificación de rutas para evitar colisiones con otros carros y obstáculos.
        - Control: Aplica el controlador adecuado (PID o Lyapunov) para seguir la trayectoria deseada y envía los comandos de velocidad al carro.
        - Actualiza la gráfica: Actualiza la visualización de la trayectoria y la posición del carro en el gráfico.
        - Almacena datos de la simulación: Guarda los errores, estados, y posiciones del carro para su posterior análisis.
        - Determina un nuevo punto objetivo si es necesario y maneja la detección de obstáculos y el parqueo automático.
        """
        """ ---------------- BEHAVIOR PLANNING (1/3) ---------------- """
        self.calucar_punto_deseado()
        self.recorrido[self.idx_almacenar, :] = [self.xi[0], self.xi[1]]
        # --------- Senales ---------
        self.senal_alto(k)
        self.senal_semaforo()
        self.senal_bajarVel()

        """ ---------------- LOCAL PLANNING ---------------- """
        # -------- Colision ---------
        self.evitarChoqueCarro(carros)

        """ ---------------- BEHAVIOR PLANNING (2/3) ---------------- """
        # -------- Velocidades --------
        factorVelocidad_completo = self.factorExtraVel*self.factorBajarVel*self.factorEvitarCarro*self.factorVelObst
        self.v_modificada = self.v_max*factorVelocidad_completo
        # -------- Cond Frenar --------
        self.cond3_frenar = (self.v_modificada == 0.0 and (not (self.FlagParqueo3 and (not self.FlagParqueo2))))
        self.cond4_frenar = self.done or self.done2
        self.cond5_frenar = (self.obstaculoDetectado1 and (self.caseObstaculos != 3))
        self.cond_frenar = self.cond1_frenar or self.cond2_frenar or self.cond3_frenar or self.cond4_frenar or self.cond5_frenar

        if self.cond_frenar:
            # -------- Frenar --------
            self.frenar()
            """ --------------------------------------------------------- """
        else:
            """ ---------------- CONTROL ---------------- """
            self.controlador()
            self.enviarVelocidades()
            """ ----------------------------------------- """
            
        # --------- Graficar ----------
        self.vehiculo_path.set_data(self.recorrido[:self.idx_almacenar, 0], self.recorrido[:self.idx_almacenar, 1])
        vertices = calcularVertices(self.xi[0], self.xi[1], self.xi[2])
        self.h_robot[0].set_xy(vertices)
        plt.draw()
        # --------- Almacenar ---------
        self.errores[k, :] = self.e
        if self.idx_almacenar < self.K:
            self.idx_almacenar += 1
        self.XI[:, k] = self.xi
        self.U[:, k] = self.u
        self.posiciones[k, :] = np.array([self.xg, self.yg, self.xi[0], self.xi[1]])

        """ ---------------- BEHAVIOR PLANNING (3/3) ---------------- """
        # ------- Nuevo Punto ---------
        self.determinar_nuevoPunto(k)

        # Evitar obstaculo
        if (not self.obstaculoDetectado1) and (self.caseObstaculos == 0):
            self.checkProxObstaculos()
        
        if self.banderaObs and (self.caseObstaculos==3) and ((self.tiempo[k] - self.time_fueraObs) > 3):
            self.caseObstaculos = 0
            self.banderaObs = False
            self.obstaculoDetectado1 = False
        """ ---------------- LOCAL PLANNING ---------------- """
        # ---- Parqueo automático -----
        self.parqueoAutomatico(k)


    def senal_alto(self,k):
        """
        Maneja el comportamiento del carro cuando se detecta una señal de alto cercana.

        Parámetros:
        k (int): El índice de tiempo actual en la simulación.

        Retorno:
        None: La función no retorna ningún valor, pero actualiza el estado del carro dependiendo de la proximidad y tiempo de la señal de alto.

        Acciones:
        - Detecta si hay una señal de alto cercana y, de ser así, inicia el tiempo de parada.
        - Desactiva temporalmente la señal de alto para evitar múltiples paradas innecesarias.
        - Después de que el tiempo de alto haya transcurrido, reactiva la señal y permite que el carro continúe su movimiento.
        - Actualiza la condición de frenado (`cond1_frenar`) dependiendo de la presencia de la señal de alto.
        """
        if not self.alto_cercano:
            self.alto_cercano, id_senal, self.time_alto = hay_senal_de_alto_cerca(self.xi[0], self.xi[1], self.senales)
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
        self.cond1_frenar = self.alto_cercano
    
    def senal_semaforo(self):
        """
        Maneja el comportamiento del carro cuando se detecta un semáforo cercano.

        Parámetros:
        None

        Retorno:
        None: La función no retorna ningún valor, pero actualiza la condición de frenado del carro (`cond2_frenar`) dependiendo del color del semáforo detectado.

        Acciones:
        - Detecta si hay un semáforo cercano.
        - Si el semáforo es de color rojo o amarillo, activa la condición de frenado (`cond2_frenar`).
        """
        semaforo_cercano, _, color = hay_semaforo_cerca(self.xi[0], self.xi[1], self.senales)
        self.cond2_frenar = (semaforo_cercano and (color == 'Rojo' or color == 'Amarillo'))

    def senal_bajarVel(self):
        """
        Ajusta el factor de reducción de velocidad del carro si se detecta una señal de "bajar velocidad" cercana.

        Parámetros:
        None

        Retorno:
        None: La función no retorna ningún valor, pero ajusta `factorBajarVel` según la señal de reducción de velocidad detectada.

        Acciones:
        - Determina si hay una señal de "bajar velocidad" cerca de la posición actual del carro.
        - Ajusta el factor de velocidad (`factorBajarVel`) para reducir la velocidad del carro de acuerdo con el valor de la señal detectada.
        """
        self.factorBajarVel = hay_senal_de_bajar_velocidad(self.xi[0], self.xi[1], self.senales)

    def calucar_punto_deseado(self):
        """
        Calcula el próximo punto deseado al cual el carro debe dirigirse, dependiendo de si el carro está en modo de parqueo o siguiendo su trayectoria normal.

        Parámetros:
        None

        Retorno:
        None: La función no retorna ningún valor, pero actualiza los atributos `xg` y `yg` para establecer las coordenadas del próximo punto objetivo.

        Acciones:
        - Si el carro está en el modo de parqueo (indicador `FlagParqueo3` activado y `FlagParqueo2` desactivado), se establece el punto de parqueo como el objetivo.
        - En caso contrario, se establece el siguiente punto en la trayectoria (`path`) como el objetivo.
        """
        if self.FlagParqueo3 and (not self.FlagParqueo2):
            posicion_d = self.parqueos[self.numParqueo-1]
            self.xg = posicion_d[0]
            self.yg = posicion_d[1]
        else:
            self.xg = self.path[self.tramo, 0]
            self.yg = self.path[self.tramo, 1]

    def determinar_nuevoPunto(self,k):
        """
        Determina el próximo tramo de la trayectoria que el carro debe seguir y actualiza el punto objetivo si se cumplen ciertas condiciones.

        Parámetros:
        k (int): El índice de tiempo actual en la simulación.

        Retorno:
        None: La función no retorna ningún valor, pero actualiza el atributo `tramo` y recalcula la ruta si se han evitado obstáculos.

        Acciones:
        - Incrementa el tramo a seguir en la trayectoria si la distancia al objetivo actual es suficientemente pequeña y no se requiere frenar.
        - Si el carro alcanza el final de la trayectoria, marca el estado como terminado (`done`).
        - En caso de haber evitado obstáculos (`caseObstaculos == 3`), recalcula la trayectoria hacia el punto final utilizando la planificación de rutas.
        - Ajusta la velocidad media y el diferencial si no se está utilizando el controlador PID.
        """
        if not (self.caseObstaculos == 3):
            if (np.linalg.norm(self.e) < 0.3) and (not self.cond2_frenar):
                self.tramo = min(self.tramo + self.diferencial, len(self.path) - 1)
                if self.tramo == len(self.path) - 1 and np.linalg.norm(self.e) < 0.05:
                        if not self.done:
                            self.timeDone = self.tiempo[k]
                            self.done = True
        else:
            if np.linalg.norm(self.e) < 0.16:
                self.tramo = min(self.tramo + self.diferencial, len(self.path) - 1)
                if self.tramo == len(self.path) - 1 and np.linalg.norm(self.e) < 0.035:
                        A1 = self.path[-1]
                        B1 = self.puntoFinal
                        path, _ = calcular_ruta(self.G, self.node_coordinates, A1[0], A1[1], B1[0], B1[1])
                        self.path = path
                        if not self.PID:
                            deltas = np.diff(path, axis=0)
                            distancias = np.sqrt(np.sum(deltas**2, axis=1))
                            self.v_mean = np.mean(distancias) / self.dt
                            self.diferencial = round(self.v_max / self.v_mean)
                        self.tramo = self.diferencial
                        self.time_fueraObs = self.tiempo[k]
                        self.banderaObs = True
                        

    def frenar(self):
        """
        Detiene el movimiento del carro estableciendo las velocidades de las ruedas y el control a cero.

        Parámetros:
        None

        Retorno:
        None: La función no retorna ningún valor, pero actualiza los atributos del carro para detenerlo.

        Acciones:
        - Establece las velocidades angulares de las ruedas (`wL`, `wR`) y el vector de control (`u`) a cero.
        - Si el carro está en modo experimental, envía un comando al robot físico para detenerlo utilizando `robotat_3pi_force_stop`.
        """
        self.wL = 0
        self.wR = 0
        self.u = np.array([0.0, 0.0])
        if self.experimental:
            robotat_3pi_force_stop(self.robot)
    
    def enviarVelocidades(self):
        """
        Envía los comandos de velocidad a las ruedas del carro o actualiza su estado utilizando el método Runge-Kutta de cuarto orden.

        Parámetros:
        None

        Retorno:
        None: La función no retorna ningún valor, pero actualiza el estado del carro (`xi`) y envía los comandos de velocidad.

        Acciones:
        - Si el carro está en modo experimental, envía las velocidades de las ruedas (`wL`, `wR`) al robot físico mediante `robotat_3pi_set_wheel_velocities`.
        - Si el carro no está en modo experimental, utiliza el método Runge-Kutta de cuarto orden para actualizar el estado (`xi`) del carro en función del control aplicado (`u`).
        """
        if self.experimental:
            robotat_3pi_set_wheel_velocities(self.robot, self.wL, self.wR)
        else:
            k1 = f(self.xi, self.u)
            k2 = f(self.xi + (self.dt / 2) * k1, self.u)
            k3 = f(self.xi + (self.dt / 2) * k2, self.u)
            k4 = f(self.xi + self.dt * k3, self.u)
            self.xi += (self.dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    def controlador(self):
        """
        Aplica el controlador adecuado (PID o Lyapunov) para calcular las velocidades de referencia del carro basadas en su posición y objetivo actual.

        Parámetros:
        None

        Retorno:
        None: La función no retorna ningún valor, pero actualiza el vector de control `u` y las velocidades de las ruedas `wL` y `wR` del carro.

        Acciones:
        - Calcula el error de posición entre el objetivo actual (`xg`, `yg`) y la posición actual (`xi`).
        - Si se usa el controlador PID:
            - Calcula el error de orientación y aplica el control proporcional-integral-derivativo para obtener la velocidad angular (`w`).
            - Ajusta la velocidad lineal (`v`) en función del error de posición.
            - Combina ambos controles para establecer los comandos de velocidad (`u`).
        - Si se usa el controlador Lyapunov:
            - Calcula el ajuste diferencial para la velocidad y aplica un control basado en matrices (`Kf`) y la cinemática del vehículo.
            - Determina las velocidades de referencia (`uRef`, `wRef`) usando la inversa del Jacobiano.
        - Calcula las velocidades de las ruedas (`wL`, `wR`) basándose en los comandos de velocidad obtenidos.
        """
        self.e = np.array([self.xg - self.xi[0], self.yg - self.xi[1]])
        eP = np.linalg.norm(self.e)

        if self.PID:
            thetag = np.arctan2(self.e[1], self.e[0])
            eO = thetag - self.xi[2]
            eO = np.arctan2(np.sin(eO), np.cos(eO))
            # Linear velocity control
            kP = self.v_modificada * (1 - np.exp(-self.alpha * eP**2)) / eP
            v = kP * eP
            # Angular velocity control
            eO_D = eO - self.eO_1
            self.EO += eO
            w = self.kpO * eO + self.kiO * self.EO + self.kdO * eO_D
            self.eO_1 = eO
            # Combine controllers
            uRef = v
            wRef = w
            self.u = np.array([uRef, wRef])
        else:
            self.diferencial = round(self.v_modificada / self.v_mean)
            Kf = 2 * np.eye(2)  # Control
            J = np.array([[np.cos(self.xi[2]), -0.08 * np.sin(self.xi[2])],
                          [np.sin(self.xi[2]),  0.08 * np.cos(self.xi[2])]])
            safe_index = min(self.tramo + self.diferencial, len(self.path) - 1)
            beta = np.arctan2(self.path[safe_index, 1] - self.yg, self.path[safe_index, 0] - self.xg)
            pdp = np.array([self.v_modificada * np.cos(beta),
                            self.v_modificada * np.sin(beta)])
            
            lambda_ = 0.01
            Jinv = np.linalg.pinv(J)
            qpRef = Jinv @ (pdp + Kf @ self.e)
            uRef = qpRef[0]
            wRef = qpRef[1]
            self.u = np.array([uRef, wRef])
        self.wL = min(max((self.u[0] - 39.5 / 1000 * self.u[1]) / (16 / 1000) * 60 / (2 * np.pi), -800), 800)
        self.wR = min(max((self.u[0] + 39.5 / 1000 * self.u[1]) / (16 / 1000) * 60 / (2 * np.pi), -800), 800)
        
    def parqueoAutomatico(self,k):
        """
        Gestiona el proceso de parqueo automático del carro, incluyendo fases intermedias y el ajuste de control.

        Parámetros:
        k (int): El índice de tiempo actual en la simulación.

        Retorno:
        None: La función no retorna ningún valor, pero actualiza el estado del carro y su ruta para realizar el parqueo automático.

        Acciones:
        - Si se completa la trayectoria inicial y se espera un tiempo, inicia el proceso de parqueo.
        - Ajusta el control y las banderas relacionadas con el proceso de parqueo.
        - Verifica si el carro está alineado con el objetivo para proceder con la segunda etapa de parqueo.
        - Marca el proceso de parqueo como completo una vez que el carro llega al objetivo.
        """
        if self.done and ((self.tiempo[k]-self.timeDone)>10) and (not self.FlagParqueo1):
            self.parqueo()
            self.done = False

        if self.FlagParqueo1 and self.done and (not self.FlagParqueo3):
            self.done = False
            if not self.PID:
                self.PID = True
                self.change = True
            self.factorExtraVel = 0
            self.FlagParqueo3 = True
        
        if self.FlagParqueo3 and (not self.FlagParqueo2):
            eO = np.arctan2(self.e[1], self.e[0]) - self.xi[2]
            eO = np.arctan2(np.sin(eO), np.cos(eO))

            if np.deg2rad(10) - np.abs(eO) > 0:
                if self.change:
                    self.PID = False
                else: 
                    self.PID = True
                self.factorExtraVel = 1.0
                self.parqueo2()

        if self.FlagParqueo2 and (self.tramo == len(self.path) - 1 and np.linalg.norm(self.e) < 0.05) and (not self.done2):
            self.done2 = True
            self.k_final = k
    
    def parqueo(self):
        """
        Calcula y establece una nueva ruta hacia el punto de parqueo designado.

        Parámetros:
        None

        Retorno:
        None: La función no retorna ningún valor, pero actualiza la ruta (`path`) del carro hacia el espacio de parqueo especificado.

        Acciones:
        - Determina la ruta desde la posición actual del carro hasta el punto de parqueo.
        - Ajusta la velocidad y el tramo a seguir en función del tipo de controlador utilizado.
        - Activa la bandera de parqueo para indicar que el proceso ha comenzado.
        """
        if not self.FlagParqueo1:
            A1 = [self.xi[0], self.xi[1]]
            B1 = self.parqueos[self.numParqueo-1]
            path, _ = calcular_ruta(self.G, self.node_coordinates, A1[0], A1[1], B1[0], B1[1])
            self.path = path
            if not self.PID:
                deltas = np.diff(path, axis=0)
                distancias = np.sqrt(np.sum(deltas**2, axis=1))
                self.v_mean = np.mean(distancias) / self.dt
                self.diferencial = round(self.v_max / self.v_mean)
                self.tramo = self.diferencial
            else:
                self.tramo = 50
            self.FlagParqueo1 = True


    def parqueo2(self):
        """
        Ajusta la trayectoria del carro para acercarse más al espacio de parqueo y suavizar la llegada.

        Parámetros:
        None

        Retorno:
        None: La función no retorna ningún valor, pero actualiza la trayectoria (`path`) del carro y establece el proceso de ajuste fino en el parqueo.

        Acciones:
        - Genera una trayectoria suave desde la posición actual hasta el espacio de parqueo.
        - Ajusta la velocidad y los parámetros de control para finalizar el parqueo.
        - Marca el proceso de parqueo como completado y actualiza el estado.
        """
        if not self.FlagParqueo2:
            A1 = np.array([self.xi[0], self.xi[1]])
            B1 = self.parqueos[self.numParqueo-1]
            difDistancia = 0.0003
            num_puntos = int(np.linalg.norm(B1 - A1) / difDistancia) + 1
            x_values = np.linspace(A1[0], B1[0], num_puntos)
            y_values = np.linspace(A1[1], B1[1], num_puntos)
            camino = np.column_stack((x_values, y_values))
            self.path = camino
            if not self.PID:
                deltas = np.diff(camino, axis=0)
                distancias = np.sqrt(np.sum(deltas**2, axis=1))
                self.v_mean = np.mean(distancias) / self.dt
                self.diferencial = round(self.v_max / self.v_mean)
                self.tramo = self.diferencial
            else:
                self.tramo = 60
            self.FlagParqueo2 = True
            self.done = False

    def evitar_colision(self, CheckCarros):
        """
        Evalúa la proximidad de otros carros y ajusta el factor de velocidad para evitar colisiones.

        Parámetros:
        CheckCarros (list): Lista de otros carros a verificar para evaluar posibles colisiones.

        Retorno:
        float: El factor de velocidad ajustado en función de la proximidad de otros carros. Disminuye si hay un carro cercano.

        Acciones:
        - Verifica la posición relativa de otros carros en el mismo carril para determinar si están demasiado cerca.
        - Ajusta el factor de velocidad para evitar colisiones.
        """
        # Inicializar el flag y el factor de velocidad
        flag_proximidad = False
        factor_velocidad = 1.0  # Comenzamos asumiendo que no hay proximidad
        inicial0 = self.v_max * self.dt * 1.7 + 0.1
        diferencia = self.v_max * self.dt

        # Verificar si la lista de carros a revisar está vacía
        if CheckCarros:
            # Iterar a través del array de objetos VA para revisar la proximidad
            for objeto_revision in CheckCarros:
                # Asegurarse de no comparar el objeto con sí mismo
                if self != objeto_revision:
                    # Verificar si el objeto de revisión está en el mismo grupo
                    inicial = inicial0 + 0.1

                    # Calcular la posición relativa del carro de revisión desde la perspectiva del carro actual
                    dx = objeto_revision.xi[0] - self.xi[0]
                    dy = objeto_revision.xi[1] - self.xi[1]

                    # Rotar el sistema de coordenadas para alinear con la dirección de movimiento del carro actual
                    theta = self.xi[2]
                    dx_rot = dx * np.cos(theta) + dy * np.sin(theta)
                    dy_rot = -dx * np.sin(theta) + dy * np.cos(theta)

                    # Solo considerar carros que están adelante y en el carril
                    if dx_rot > 0 and abs(dy_rot) <= 0.15:
                        # Evaluar la distancia y asignar factor_velocidad
                        for j in range(10):
                            distancia_evaluar = inicial + diferencia * j
                            velocidad_evaluar = 0.10 * j
                            if dx_rot <= distancia_evaluar:
                                factor_velocidad = min(factor_velocidad, velocidad_evaluar)
                                flag_proximidad = True
                                break  # Salir del bucle ya que encontramos el rango relevante
        return factor_velocidad
    
    def evitarChoqueCarro(self,carros):
        """
        Ajusta el factor de velocidad del carro para evitar colisiones con otros carros en la simulación.

        Parámetros:
        carros (list): Lista de todos los carros en la simulación.

        Retorno:
        None: La función no retorna ningún valor, pero ajusta `factorEvitarCarro` basado en la proximidad a otros carros.

        Acciones:
        - Llama a `evitar_colision` para evaluar y ajustar la velocidad en función de los carros cercanos.
        """
        self.factorEvitarCarro = self.evitar_colision(carros)

    def checkProxObstaculos(self):
        """
        Verifica si hay obstáculos en el camino del carro y ajusta el factor de velocidad y la detección de obstáculos en consecuencia.

        Parámetros:
        None

        Retorno:
        None: La función no retorna ningún valor, pero actualiza `factorVelObst` y la bandera `obstaculoDetectado1` si hay un obstáculo cercano.

        Acciones:
        - Recorre los próximos puntos en la trayectoria del carro para identificar la presencia de obstáculos.
        - Ajusta el factor de velocidad según la proximidad del obstáculo.
        - Marca un nuevo punto de ruta fuera del obstáculo si es necesario.
        """
        self.factorVelObst = 1
        self.obstaculoDetectado1 = False
        # Recorrer los siguientes 25 puntos en el camino
        val_min = 40
        dif = 10
        val_max = val_min + 21
        for n in range(val_max):
            indice_futuro = min(self.tramo + n, len(self.path) - 1)
            posicion_futura = self.path[indice_futuro]
            x, y = posicion_futura

            # Verificar cada obstáculo
            for obstaculo in self.obstaculos:
                x1, y1, x2, y2 = obstaculo

                if x1 <= x <= x2 and y1 <= y <= y2:
                    # Determinar el factor de velocidad basado en la proximidad
                    rangos_factores = {
                        (0, val_min): 0,
                        (val_min+1, val_min+dif): 0.20,
                        (val_min+dif+1, val_min+2*dif): 0.40,
                        (val_min+2*dif+1, val_min+3*dif): 0.60,
                        (val_min+3*dif+1, val_min+4*dif): 0.80
                    }
                    for rango, factor in rangos_factores.items():
                        if rango[0] <= n <= rango[1]:
                            self.factorVelObst = factor
                            break  # Break del ciclo de rangos_factores

                    # Si el obstáculo está muy cerca, activamos la detección y buscamos el siguiente punto fuera del obstáculo
                    if n <= val_min:
                        self.obstaculoDetectado1 = True
                        for m in range(n, 100000):
                            indice_futuro_sig = min(self.tramo + m, len(self.path) - 1)
                            posicion_futura_sig = self.path[indice_futuro_sig]
                            x_sig, y_sig = posicion_futura_sig
                            
                            # Si el siguiente punto futuro no está dentro del obstáculo, salir del ciclo
                            distance = math.sqrt((x_sig-self.xi[0])**2+(y_sig-self.xi[1])**2)
                            if (not (x1 <= x_sig <= x2 and y1 <= y_sig <= y2)) and (distance > 2.5*0.28284271247461906):
                                self.pos_futuro_obs = posicion_futura_sig
                                return  # Salir de la función al encontrar el siguiente punto fuera del obstáculo

                    break  # Break del ciclo de obstáculos

            if self.factorVelObst != 1:
                break  # Break del ciclo de posiciones futuras

        if self.factorVelObst == 0:
            self.obstaculoDetectado1 = True

    def ruta_obstaculo2(self):
        """
        Recalcula la trayectoria del carro para evitar obstáculos utilizando un enfoque de planificación con esquinas y rutas alternativas.

        Parámetros:
        None

        Retorno:
        None: La función no retorna ningún valor, pero actualiza la trayectoria (`path`) del carro para esquivar los obstáculos detectados.

        Acciones:
        - Identifica posibles rutas alternativas alrededor del obstáculo utilizando puntos intermedios.
        - Aplica una estrategia de planificación de esquinas para evitar colisiones y continuar hacia el objetivo.
        """
        rectangulos = np.array([
            [0.50, 0.00, 2.98, 0.40],
            [2.10, 0.50, 2.53, 0.80],
            [0.50, 0.87, 2.00, 1.30],
            [2.53, 0.80, 2.99, 1.24],
            [3.04, 1.34, 3.41, 2.23],
            [3.03, 2.60, 3.41, 3.21],
            [0.82, 2.02, 1.75, 2.40],
            [0.02, 0.44, 0.40, 0.79],
        ])
        x4, y4 = self.pos_futuro_obs
        x1, y1 = self.xi[0], self.xi[1]

        for rect in rectangulos:
            x1_r, y1_r, x2_r, y2_r = rect
            if x1_r <= x4 <= x2_r and y1_r <= y4 <= y2_r:
                distancia = math.sqrt(0.18**2 + 0.18**2)
                if math.sqrt((x4-x1)**2+(y4-y1)**2) > 2*distancia:
                    angulo_punto2 = self.xi[2] + math.radians(45)
                    x2 = x1 + distancia * math.cos(angulo_punto2)
                    y2 = y1 + distancia * math.sin(angulo_punto2)
                    opciones = [
                        (x4 - 0.20, y4 - 0.20),
                        (x4 - 0.20, y4 + 0.20),
                        (x4 + 0.20, y4 - 0.20),
                        (x4 + 0.20, y4 + 0.20)
                    ]
                    vector_14 = np.array([x4 - x1, y4 - y1])
                    mejor_opcion = None
                    paralelismo_tol = 0.04  # Tolerancia para considerar vectores casi paralelos
                    for opcion in opciones:
                        vector_23 = np.array([opcion[0] - x2, opcion[1] - y2])
                        # Calcular el producto cruzado para ver si los vectores son casi paralelos
                        producto_cruz = np.abs(np.cross(vector_14, vector_23))
                        if producto_cruz < paralelismo_tol:
                            # Comprobar el ángulo entre las líneas 2-3 y 3-4 para asegurar que sea obtuso
                            vector_43 = np.array([x4 - opcion[0], y4 - opcion[1]])
                            cos_theta = np.dot(vector_23, -vector_43) / (np.linalg.norm(vector_23) * np.linalg.norm(vector_43))
                            theta = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))  # Convertir a grados
                            if theta > 90:
                                mejor_opcion = opcion
                                break

                    if not(mejor_opcion is None):
                        x3, y3 = mejor_opcion
                        points = [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
                        ruta_nueva = []
                        espaciado = 0.0003
                        # Calcular secuencias entre puntos consecutivos
                        for i in range(len(points) - 1):
                            # Punto actual y siguiente
                            p1 = points[i]
                            p2 = points[i + 1]

                            # Distancia entre los puntos
                            distancia = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
                            num_puntos = max(int(distancia / espaciado), 1)  # Asegurarse de tener al menos un punto

                            # Generar puntos
                            valores_x = np.linspace(p1[0], p2[0], num_puntos + 1)  # +1 para incluir el punto final
                            valores_y = np.linspace(p1[1], p2[1], num_puntos + 1)  # +1 para incluir el punto final

                            # Combinar coordenadas x e y
                            nuevos_puntos = list(zip(valores_x, valores_y))
                            if i != 0:
                                nuevos_puntos = nuevos_puntos[1:]  # Evitar duplicar puntos que ya están en ruta_nueva
                            ruta_nueva.extend(nuevos_puntos)
                        
                        ruta_smooth = np.array(ruta_nueva)  # Convertir la lista de tuplas a un array de NumPy
                        # Aplicar suavizado usando rolling window con pandas
                        # x_smooth = pd.Series(ruta_smooth[:, 0]).rolling(window=int(0.90 * len(ruta_smooth)), min_periods=1, center=True).mean()
                        # y_smooth = pd.Series(ruta_smooth[:, 1]).rolling(window=int(0.90 * len(ruta_smooth)), min_periods=1, center=True).mean()
                        # ruta_smooth = np.column_stack((x_smooth, y_smooth))

                        self.caseObstaculos = 3
                        self.factorVelObst = 1
                        self.e = np.array([0, 0])
                        #np.savetxt('ruta_smooth.csv', ruta_smooth, delimiter=',', fmt='%.6f')
                        if not self.PID:
                            deltas = np.diff(ruta_smooth, axis=0)
                            distancias = np.sqrt(np.sum(deltas**2, axis=1))
                            self.v_mean = np.mean(distancias) / self.dt
                            self.diferencial = round(self.v_max / self.v_mean)
                        self.tramo = self.diferencial
                        self.path = ruta_smooth
        
        if self.caseObstaculos != 3:
            print("No se pudo")
            self.caseObstaculos = 2

    def ruta_obstaculo(self):
        """
        Genera una nueva ruta hacia el objetivo evitando los obstáculos presentes en el grafo cargado.

        Parámetros:
        None

        Retorno:
        None: La función no retorna ningún valor, pero recalcula y suaviza la trayectoria (`path`) del carro para esquivar obstáculos.

        Acciones:
        - Utiliza un grafo de rutas predefinidas y genera una nueva ruta que evita obstáculos.
        - Suaviza la nueva ruta generada para asegurar una trayectoria más eficiente.
        """
        lines = pd.read_csv('Lines.csv').values
        G, node_coordinates = crear_grafo2(lines, self.obstaculos)
        ruta, _ = calcular_ruta(G, node_coordinates, self.xi[0], self.xi[1], 1.7, 0.96)
        if ruta is not None:
            nueva_ruta = [ruta[0, :]]
            for i in range(1, len(ruta)):
                punto_actual = ruta[i, :]
                punto_anterior = ruta[i - 1, :]
                vector = punto_actual - punto_anterior
                distancia = np.linalg.norm(vector)

                if distancia > 0.0003:
                    # Calcular cuántos puntos adicionales son necesarios
                    num_puntos = int(np.ceil(distancia / 0.0003)) - 1
                    # Crear puntos adicionales
                    for j in range(1, num_puntos + 1):
                        punto_nuevo = punto_anterior + (vector * j / (num_puntos + 1))
                        nueva_ruta.append(punto_nuevo)
                nueva_ruta.append(punto_actual)  # Asegurar incluir el punto actual

            nueva_ruta = np.array(nueva_ruta)
            x_smooth = pd.Series(nueva_ruta[:, 0]).rolling(window=int(0.90 * len(nueva_ruta)), min_periods=1, center=True).mean()
            y_smooth = pd.Series(nueva_ruta[:, 1]).rolling(window=int(0.90 * len(nueva_ruta)), min_periods=1, center=True).mean()
            ruta_smooth = np.column_stack((x_smooth, y_smooth))
            
            if not np.array_equal(ruta_smooth[-1, :], ruta[-1, :]):
                ruta_smooth = np.vstack([ruta_smooth, ruta[-1, :]])
            if not np.array_equal(ruta_smooth[0, :], ruta[0, :]):
                distancia = np.linalg.norm(ruta[0, :] - ruta_smooth[0, :])
                x_values = np.linspace(ruta[0, 0], ruta_smooth[0, 0], int(np.ceil(distancia / 0.0003)) - 1)
                y_values = np.linspace(ruta[0, 1], ruta_smooth[0, 1], int(np.ceil(distancia / 0.0003)) - 1)
                nuevosValores = np.column_stack((x_values, y_values))
                ruta_smooth = np.vstack([nuevosValores, ruta_smooth])
                
            print("Sí se pudo")
            self.caseObstaculos = 3
            self.factorVelObst = 1
            self.e = np.array([0, 0])
            np.savetxt('ruta_smooth.csv', ruta_smooth, delimiter=',', fmt='%.6f')
            if not self.PID:
                deltas = np.diff(ruta_smooth, axis=0)
                distancias = np.sqrt(np.sum(deltas**2, axis=1))
                self.v_mean = np.mean(distancias) / self.dt
                self.diferencial = round(self.v_max / self.v_mean)
            self.tramo = self.diferencial
            self.path = ruta_smooth
        else:
            print("No se pudo")
            self.caseObstaculos = 2
    
    def to_dict(self):
        """
        Convierte el objeto carro en un diccionario serializable para guardarlo en un archivo.

        Parámetros:
        None

        Retorno:
        dict: Un diccionario que contiene todos los atributos relevantes del carro para ser serializados y guardados.

        Acciones:
        - Convierte los atributos del carro a un formato adecuado para ser almacenado en un archivo JSON.
        """
        # Convierte el objeto carro en un diccionario serializable
        return {
            'num': self.no_robot,
            'vMax': self.v_max,
            'xi': self.xi.tolist(),
            'dt': self.dt,
            'tf': self.tf,
            'B': self.B,
            'FlagExperimental': self.experimental,
            'delta': self.delta,
            'noParqueo': self.numParqueo,
            'robot': self.robot,
            'FlagControlador': self.PID,
            # Asegúrate de incluir cualquier otro atributo que desees serializar
        }

    @classmethod
    def from_dict(cls, data, ax, canvas, robotat, G, node_coordinates):
        """
        Crea un nuevo objeto `carro` a partir de un diccionario previamente serializado.

        Parámetros:
        data (dict): Diccionario que contiene los atributos del carro.
        ax (matplotlib.axes._axes.Axes): Objeto de ejes de Matplotlib para graficar el carro.
        canvas (matplotlib.backends.backend_agg.FigureCanvasAgg): Lienzo de Matplotlib para actualizar la visualización.
        robotat (object): Objeto de conexión con Robotat.
        G (networkx.Graph): Grafo de caminos utilizado para la planificación de la ruta.
        node_coordinates (ndarray): Coordenadas de los nodos del grafo.

        Retorno:
        cls: Una nueva instancia del objeto `carro` con los atributos restaurados.

        Acciones:
        - Crea instancias de los elementos gráficos (`vehiculo_path`, `h_robot`) y restaura los atributos del carro desde el diccionario `data`.
        """
        # Crea un nuevo objeto carro a partir de un diccionario
        xi = np.array(data['xi'])
        
        # Crea elementos gráficos para el nuevo carro
        vehiculo_path, = ax.plot(xi[0], xi[1], 'b-', linewidth=2)
        vertices = calcularVertices(xi[0], xi[1], xi[2])
        h_robot = ax.fill(vertices[:, 0], vertices[:, 1], 'b')
        canvas.draw()

        return cls(
            numero = data['num'],
            vMax=data['vMax'],
            xi=xi,
            dt=data['dt'],
            tf=data['tf'],
            B=np.array(data['B']),
            FlagExperimental=data['FlagExperimental'],
            delta=data['delta'],
            noParqueo=data['noParqueo'],
            robot=data['robot'],
            FlagControlador=data['FlagControlador'],
            vehiculo_path=vehiculo_path,
            h_robot=h_robot,
            robotat=robotat,
            grafo=G,
            coordenadasNodos=node_coordinates
        )