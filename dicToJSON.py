import json

# Ejemplo de diccionario
mi_diccionario = [
    {'id': 1, 'activo': True, 'tipo': 'alto', 'posicion': [1.70, 2.1135], 'detenido': False, 'valor': 3},
    {'id': 2, 'activo': True, 'tipo': 'alto', 'posicion': [2.28, 3.0], 'detenido': False, 'valor': 3},
    {'id': 3, 'activo': True, 'tipo': 'alto', 'posicion': [2.90, 2.40074], 'detenido': False, 'valor': 3},
    {'id': 4, 'activo': True, 'tipo': 'alto', 'posicion': [2.37561, 1.81], 'detenido': False, 'valor': 3},
    {'id': 5, 'activo': True, 'tipo': 'alto', 'posicion': [0.124549, 1.34617], 'detenido': False, 'valor': 3},
    {'id': 6, 'activo': True, 'tipo': 'alto', 'posicion': [0.473758, 1.15894], 'detenido': False, 'valor': 3},
    {'id': 7, 'activo': True, 'tipo': 'alto', 'posicion': [2.18442, 1.33198], 'detenido': False, 'valor': 3},
    {'id': 8, 'activo': True, 'tipo': 'alto', 'posicion': [2.03329, 0.966932], 'detenido': False, 'valor': 3},
    {'id': 9, 'activo': True, 'tipo': 'alto', 'posicion': [2.37559, 0.779572], 'detenido': False, 'valor': 3},
    {'id': 10, 'activo': True, 'tipo': 'alto', 'posicion': [2.1845, 0.512079], 'detenido': False, 'valor': 3},
    {'id': 11, 'activo': True, 'tipo': 'alto', 'posicion': [2.00984, 0.125002], 'detenido': False, 'valor': 3},
    {'id': 12, 'activo': True, 'tipo': 'bajarVelocidad', 'posicion': [[0, 3], [1.75, 4.5]], 'detenido': False, 'valor': 0.90},
    {'id': 13, 'activo': True, 'tipo': 'bajarVelocidad', 'posicion': [[1.7, 1.81], [2.9, 3.02]], 'detenido': False, 'valor': 0.5},
    {'id': 14, 'activo': True, 'tipo': 'semaforo', 'posicion': [3.44708, 3.43426], 'detenido': False, 'valor': 'Verde'},
    {'id': 15, 'activo': True, 'tipo': 'bajarVelocidad', 'posicion': [[2.1, 1.63], [2.26, 1.5]], 'detenido': False, 'valor': 0.5},
    {'id': 16, 'activo': True, 'tipo': 'bajarVelocidad', 'posicion': [[2.28, 0.78], [2.45, 0.63]], 'detenido': False, 'valor': 0.2},
    {'id': 17, 'activo': True, 'tipo': 'bajarVelocidad', 'posicion': [[2.28, 0.63], [2.45, 0.48]], 'detenido': False, 'valor': 0.5},
    {'id': 18, 'activo': True, 'tipo': 'bajarVelocidad', 'posicion': [[2.18, 3.02], [2.36, 3.17]], 'detenido': False, 'valor': 0.2},
    {'id': 19, 'activo': True, 'tipo': 'bajarVelocidad', 'posicion': [[2.18, 3.17], [2.36, 3.32]], 'detenido': False, 'valor': 0.5},
    {'id': 20, 'activo': True, 'tipo': 'bajarVelocidad', 'posicion': [[1.71, 0.2], [1.86, 0.04]], 'detenido': False, 'valor': 0.5},
    {'id': 21, 'activo': True, 'tipo': 'bajarVelocidad', 'posicion': [[1.86, 0.2], [2.01, 0.04]], 'detenido': False, 'valor': 0.2},
    {'id': 22, 'activo': True, 'tipo': 'bajarVelocidad', 'posicion': [[0.03, 1.35], [0.2, 1.5]], 'detenido': False, 'valor': 0.2},
    {'id': 23, 'activo': True, 'tipo': 'bajarVelocidad', 'posicion': [[0.03, 1.5], [0.2, 1.65]], 'detenido': False, 'valor': 0.5},
    {'id': 24, 'activo': True, 'tipo': 'bajarVelocidad', 'posicion': [[2.03, 1.04], [1.88, 0.88]], 'detenido': False, 'valor': 0.2},
    {'id': 25, 'activo': True, 'tipo': 'bajarVelocidad', 'posicion': [[1.88, 1.04], [1.73, 0.88]], 'detenido': False, 'valor': 0.5},
    {'id': 26, 'activo': True, 'tipo': 'bajarVelocidad', 'posicion': [[2.1, 0.52], [2.26, 0.67]], 'detenido': False, 'valor': 0.2},
    {'id': 27, 'activo': True, 'tipo': 'bajarVelocidad', 'posicion': [[2.1, 0.67], [2.26, 0.82]], 'detenido': False, 'valor': 0.5},
    {'id': 28, 'activo': True, 'tipo': 'bajarVelocidad', 'posicion': [[0.47, 1.23], [0.62, 1.07]], 'detenido': False, 'valor': 0.2},
    {'id': 29, 'activo': True, 'tipo': 'bajarVelocidad', 'posicion': [[0.62, 1.23], [0.77, 1.07]], 'detenido': False, 'valor': 0.5},
    {'id': 30, 'activo': True, 'tipo': 'bajarVelocidad', 'posicion': [[2.90, 2.49], [3.05, 2.31]], 'detenido': False, 'valor': 0.2},
    {'id': 31, 'activo': True, 'tipo': 'bajarVelocidad', 'posicion': [[3.05, 2.49], [3.20, 2.31]], 'detenido': False, 'valor': 0.5},
    {'id': 32, 'activo': True, 'tipo': 'bajarVelocidad', 'posicion': [[1.7, 2.20], [1.55, 2.03]], 'detenido': False, 'valor': 0.2},
    {'id': 33, 'activo': True, 'tipo': 'bajarVelocidad', 'posicion': [[1.55, 2.20], [1.4, 2.03]], 'detenido': False, 'valor': 0.5},
    {'id': 34, 'activo': True, 'tipo': 'bajarVelocidad', 'posicion': [[2.28, 1.81], [2.45, 1.66]], 'detenido': False, 'valor': 0.2},
    {'id': 35, 'activo': True, 'tipo': 'bajarVelocidad', 'posicion': [[2.28, 1.66], [2.45, 1.51]], 'detenido': False, 'valor': 0.5},
    {'id': 36, 'activo': True, 'tipo': 'bajarVelocidad', 'posicion': [[0.03, 1.24], [0.2, 1.35]], 'detenido': False, 'valor': 0.5},
    {'id': 37, 'activo': True, 'tipo': 'bajarVelocidad', 'posicion': [[0.47, 1.23], [0.34, 1.07]], 'detenido': False, 'valor': 0.5},
    {'id': 38, 'activo': True, 'tipo': 'bajarVelocidad', 'posicion': [[2.01, 0.2], [2.18, 0.04]], 'detenido': False, 'valor': 0.5},
    {'id': 39, 'activo': True, 'tipo': 'bajarVelocidad', 'posicion': [[2.1, 0.52], [2.25, 0.4]], 'detenido': False, 'valor': 0.5},
    {'id': 40, 'activo': True, 'tipo': 'bajarVelocidad', 'posicion': [[2.45, 0.78], [2.28, 0.90]], 'detenido': False, 'valor': 0.5},
    {'id': 41, 'activo': True, 'tipo': 'bajarVelocidad', 'posicion': [[2.03, 0.88], [2.15, 1.04]], 'detenido': False, 'valor': 0.5},
    {'id': 42, 'activo': True, 'tipo': 'bajarVelocidad', 'posicion': [[2.1, 1.18], [2.26, 1.33]], 'detenido': False, 'valor': 0.7},
    {'id': 43, 'activo': True, 'tipo': 'bajarVelocidad', 'posicion': [[2.26, 1.5], [2.1, 1.33]], 'detenido': False, 'valor': 0.2},
    {'id': 44, 'activo': True, 'tipo': 'bajarVelocidad', 'posicion': [[3.447, 3.50], [3.26, 3.317]], 'detenido': False, 'valor': 0.6},
    {'id': 45, 'activo': True, 'tipo': 'bajarVelocidad', 'posicion': [[2.56, 3.64], [3.17, 3.96]], 'detenido': False, 'valor': 0.8},
    {'id': 46, 'activo': True, 'tipo': 'bajarVelocidad', 'posicion': [[2, 1.30271], [2.53, 0.7916]], 'detenido': False, 'valor': 0.8},
    {'id': 47, 'activo': True, 'tipo': 'bajarVelocidad', 'posicion': [[3.447, 3.5], [3.75, 3.1265]], 'detenido': False, 'valor': 0.8},
    {'id': 48, 'activo': True, 'tipo': 'semaforo', 'posicion': [1.1986, 0.9640], 'detenido': False, 'valor': 'Verde'}
]

# Escribir el diccionario a un archivo JSON
with open('senales.json', 'w') as archivo_json:
    json.dump(mi_diccionario, archivo_json, indent=4)

# Leer el archivo JSON y convertirlo a un diccionario
with open('senales.json', 'r') as archivo_json:
    mi_diccionario_cargado = json.load(archivo_json)

print(mi_diccionario_cargado)